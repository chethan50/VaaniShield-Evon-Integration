"""Authenticated, conversation-first assistant page."""

from __future__ import annotations

import base64
import hashlib
import logging
from uuid import UUID, uuid4

from frontend.api_client import (
    APIError,
    build_speech_turn_request,
    build_text_turn_request,
)
from frontend.components.conversation import (
    message_from_history_turn,
    messages_from_turn_response,
    render_messages,
)
from frontend.components.conversation_history import create_new_conversation
from frontend.components.voice_composer import render_language_controls, render_compact_voice, decode_voice_event
from frontend.navigation import FEATURE_SUBFEATURES, request_navigation
from frontend.session_state import (
    bind_pending_scam_assessment,
    clear_pending_scam_assessment,
    sign_out,
)

_voice_log = logging.getLogger(__name__ + ".voice")


def _record_voice_diagnostic(state, stage: str, event_id: str | None = None, kind: str | None = None) -> None:
    """Record privacy-safe voice lifecycle stages without transcripts, audio, or PII."""
    entry = {"stage": stage, "event_id": str(event_id) if event_id else None}
    if kind:
        entry["kind"] = str(kind)
    diagnostics = state.setdefault("voice_diagnostics", [])
    diagnostics.append(entry)
    if len(diagnostics) > 100:
        del diagnostics[:-100]
    if kind:
        _voice_log.info("voice stage=%s event_id=%s kind=%s", stage, event_id, kind)
    else:
        _voice_log.info("voice stage=%s event_id=%s", stage, event_id)


_SCAM_FOLLOW_UP_TERMS = (
    "scam", "fraud", "suspicious", "this assessment", "that assessment",
    "this result", "that result", "this warning", "that warning",
    "this message", "this call", "should i block them",
    "what should i do about it", "next steps for this", "report this",
    "\u0938\u094d\u0915\u0948\u092e", "\u091c\u094b\u0916\u093f\u092e", "\u0938\u0941\u0930\u0915\u094d\u0937\u093f\u0924",
    "\u0cb8\u0ccd\u0caf\u0cbe\u0cae\u0ccd", "\u0c85\u0caa\u0cbe\u0caf", "\u0cb8\u0cc1\u0cb0\u0c95\u0ccd\u0cb7\u0cbf\u0ca4",
)


def assessment_reference_for_text(state, conversation_id, text):
    assessment_id = state.get("pending_scam_assessment_id")
    if not assessment_id:
        return None
    if str(state.get("pending_scam_conversation_id")) != str(conversation_id):
        clear_pending_scam_assessment(state)
        return None
    normalized = (text or "").casefold()
    banking_terms = (
        "balance", "account", "accounts", "card", "cards", "loan", "loans",
        "transaction", "transactions", "spending", "spend", "savings",
        "खाता", "बैलेंस", "कार्ड", "लोन", "लेनदेन", "ಖಾತೆ", "ಬ್ಯಾಲೆನ್ಸ್", "ಕಾರ್ಡ್", "ಸಾಲ", "ವಹಿವಾಟು",
    )
    if any(term in normalized for term in banking_terms):
        clear_pending_scam_assessment(state)
        return None
    if not any(term.casefold() in normalized for term in _SCAM_FOLLOW_UP_TERMS):
        clear_pending_scam_assessment(state)
        return None
    return str(assessment_id)


def transaction_risk_reference_for_conversation(state, conversation_id):
    """Return a public assessment reference only in the conversation bound to it."""
    assessment_id = state.get("transaction_risk_follow_up_assessment_id")
    if not assessment_id:
        return None
    if str(state.get("transaction_risk_follow_up_conversation_id")) != str(conversation_id):
        state["transaction_risk_follow_up_assessment_id"] = None
        state["transaction_risk_follow_up_conversation_id"] = None
        return None
    try:
        return str(UUID(str(assessment_id)))
    except (ValueError, TypeError, AttributeError):
        state["transaction_risk_follow_up_assessment_id"] = None
        state["transaction_risk_follow_up_conversation_id"] = None
        return None


def _api_error(st, state, exc):
    if exc.status_code == 401:
        sign_out(state)
        st.warning("Your demo session expired. Sign in again to continue.")
        st.rerun()
    messages = {
        404: "This conversation or audio item is unavailable. Start a new conversation and try again.",
        413: "That recording is too large. Please record a shorter question.",
        415: "The recording format is not supported. Please use WAV, MP3, OGG, or WebM audio.",
        503: "The requested assistant service is temporarily unavailable. Your conversation remains available.",
        429: "The assistant provider is temporarily rate limited. Your message is saved and can be retried shortly.",
        502: "The assistant provider could not accept this request. Your message is saved and can be retried.",
        504: "The assistant provider timed out. Your message is saved and can be retried.",
        0: "The VaaniShield API could not be reached. Check that the backend is running.",
    }
    safe_provider_details = {
        "Speech recognition is not configured. Ask the administrator to check Gnani settings.",
        "Speech recognition timed out. Please try again.",
        "Speech recognition credentials were rejected. Ask the administrator to check Gnani settings.",
        "Speech recognition is temporarily rate limited. Please try again shortly.",
        "Speech recognition provider returned an error. Please try again later.",
        "Speech recognition could not connect to its provider. Please try again later.",
        "Speech recognition could not process the recording. Check Gnani configuration or try again later.",
        "The assistant took too long to respond. Please try again.",
        "The assistant could not respond right now. Your message is saved. Please try again.",
        "The Gemini SDK is not installed. Install the project requirements.",
        "Gemini is selected but GEMINI_API_KEY is not configured.",
        "The assistant request was too large for the provider, even after safely condensing context. Your message is saved; retry or ask a shorter question.",
        "The assistant service is temporarily rate limited. Your message is saved; please retry shortly.",
        "The assistant took too long to respond. Your message is saved; you can retry it.",
        "The assistant provider is temporarily unavailable. Your message is saved; you can retry it.",
        "The assistant could not connect to its provider. Your message is saved; you can retry it.",
        "The assistant provider rejected its configured credentials or permissions. Ask the administrator to check provider configuration.",
    }
    detail = str(exc)
    # The API's detail field contains only authored, provider-safe messages.
    # Show only known-safe details; do not surface unrecognized backend text.
    safe_transport_details = {
        "The VaaniShield API connection timed out.",
        "The VaaniShield API did not respond before the request timed out.",
        "The VaaniShield API could not be reached. Check that the backend is running.",
        "The VaaniShield API request could not be completed.",
    }
    st.error(
        detail if detail in safe_provider_details or detail in safe_transport_details
        else messages.get(exc.status_code, "The request could not be completed. Check the request and try again.")
    )


def _load_history(api, state):
    conversation_id = state.get("conversation_id")
    if not conversation_id or state.get("loaded_conversation_id") == conversation_id:
        return
    page = api.get_conversation_turns(conversation_id, limit=100)
    turns = page.get("items", [])
    state["conversation_messages"] = [message_from_history_turn(turn) for turn in turns]
    state["loaded_conversation_id"] = conversation_id
    state["conversation_cursor"] = page.get("next_cursor")
    state["conversation_has_more"] = bool(page.get("has_more"))


def _ensure_conversation(api, state):
    if not state.get("conversation_id"):
        create_new_conversation(api, state)


def _remember_failed_user_message(state, failed):
    client_id = failed.get("client_turn_id")
    local_id = f"failed-{client_id}"
    if client_id and not any(item.get("id") == local_id for item in state["conversation_messages"]):
        state["conversation_messages"].append({
            "id": local_id,
            "role": "user",
            "text": failed.get("text") or "Saved request",
            "input_mode": "speech" if failed.get("is_voice") else "text",
            "detected_languages": [],
            "is_code_mixed": False,
            "speech_audio_id": None,
            "response_language": None,
        })


def _send_turn(api, state, payload):
    is_voice_turn = payload.get("input_mode") == "speech" or bool(state.get("active_voice_component_event"))
    if is_voice_turn:
        _record_voice_diagnostic(state, "assistant_turn_started", payload.get("client_turn_id"), payload.get("input_mode"))
    try:
        response = api.send_assistant_turn(payload)
    except APIError as exc:
        failed = state.get("assistant_failed_turn") or {}
        if exc.status_code == 0:
            # The backend may have finished even though the response was lost.
            # Reuse the exact idempotency key once to retrieve its saved result.
            try:
                response = api.send_assistant_turn(payload)
            except APIError as retry_exc:
                exc = retry_exc
            else:
                exc = None
        if exc is not None:
            error_code = (exc.payload or {}).get("error_code") if isinstance(exc.payload, dict) else None
            if isinstance(error_code, str) and (error_code.startswith("SPEECH_") or error_code in {"EMPTY_TRANSCRIPT", "AUDIO_UPLOAD_UNAVAILABLE"}):
                failed["needs_audio_upload"] = True
            if exc.status_code == 0 or (exc.status_code == 409 and any(
                term in str(exc).casefold() for term in ("already processing", "in progress")
            )):
                failed["status"] = "processing"
                failed["recovery_attempted"] = True
                failed["error"] = None
                failed["status_message"] = (
                    "The connection was interrupted; Vaani may still be completing this request."
                    if exc.status_code == 0 else
                    "Vaani is still processing this saved request. Check its response in a moment."
                )
            else:
                failed["status"] = "failed"
                failed["error"] = str(exc) or "The assistant request could not be completed. Please retry."
            state["assistant_failed_turn"] = failed
            _remember_failed_user_message(state, failed)
            raise exc

    if not isinstance(response, dict):
        message = "The assistant returned an unreadable response. Retry to retrieve its saved result."
        failed = state.get("assistant_failed_turn") or {}
        failed.update(status="failed", error=message)
        state["assistant_failed_turn"] = failed
        _remember_failed_user_message(state, failed)
        raise APIError(502, message)
    if is_voice_turn:
        _record_voice_diagnostic(state, "assistant_turn_completed", payload.get("client_turn_id"), payload.get("input_mode"))
    application_error = response.get("error") or response.get("error_code")
    if application_error or not all(response.get(key) for key in (
        "conversation_id", "turn_id", "assistant_text",
    )):
        message = (
            str(response.get("message") or application_error)
            if application_error else
            "The assistant returned an incomplete response. Retry to retrieve its saved result."
        )
        failed = state.get("assistant_failed_turn") or {}
        failed.update(status="failed", error=message)
        state["assistant_failed_turn"] = failed
        _remember_failed_user_message(state, failed)
        raise APIError(502, message, response)
    return response


def _send_turn_with_spinner(api, st, state, payload):
    with st.spinner("Vaani is preparing your response…"):
        return _send_turn(api, state, payload)


def _submit(api, st, state, *, text=None, audio=None, retry=False):
    """Guard one composer submission against duplicate events and blank text."""
    if state.get("assistant_submission_in_progress"):
        return False
    if not retry and audio is None and not (text or "").strip() and not state.get("pending_attachments"):
        return False
    state["assistant_submission_in_progress"] = True
    try:
        return _submit_once(api, st, state, text=text, audio=audio, retry=retry)
    finally:
        state["assistant_submission_in_progress"] = False


def _submit_recorded_audio(api, st, state, audio, *, widget_epoch=None):
    """Submit one stopped recording, ignoring duplicate widget deliveries."""
    raw = audio.getvalue() if audio is not None else b""
    if not raw:
        st.warning("No audio was captured. Try recording again.")
        return False
    event_id = state.get("active_voice_component_event")
    digest = hashlib.sha256(raw).hexdigest()
    widget_epoch = int(state.get("audio_widget_epoch", 0) if widget_epoch is None else widget_epoch)
    submission_key = (event_id, digest) if event_id else (widget_epoch, digest)
    if (
        submission_key == state.get("last_audio_submission_key")
        or state.get("assistant_submission_in_progress")
        or widget_epoch != int(state.get("audio_widget_epoch", 0))
    ):
        _record_voice_diagnostic(state, "duplicate_event_rejected", event_id, "audio")
        return False
    state["last_audio_submission_key"] = submission_key
    state["last_audio_hash"] = digest
    try:
        result = _submit(api, st, state, audio=audio)
        return result is not False
    except Exception:
        # The audio widget key is advanced by its caller after failure. The
        # saved retry payload retains its upload ID and idempotency key.
        state["last_audio_hash"] = None
        state["audio_widget_epoch"] = int(state.get("audio_widget_epoch", 0)) + 1
        raise


def _submit_once(api, st, state, *, text=None, audio=None, retry=False):
    _ensure_conversation(api, state)
    mode = state.get("response_mode", "text")
    if (
        (audio is not None or state.get("active_voice_component_event"))
        and mode == "text"
        and not state.get("response_mode_explicit", False)
    ):
        mode = "speech_and_text"
    response_language = state["response_language"]
    assessment_id = None
    failed_turn = state.get("assistant_failed_turn") if retry else None
    pending_docs = list(state.get("pending_attachments") or [])
    attachments_payload = [
        {
            "attachment_id": doc.get("attachment_id"),
            "file_name": doc.get("file_name"),
            "file_type": doc.get("file_type"),
            "file_size_bytes": doc.get("file_size_bytes", 0),
            "extracted_text": doc.get("extracted_text", ""),
            "summary_metadata": doc.get("summary_metadata", {}),
        }
        for doc in pending_docs
    ] if pending_docs else None

    if retry:
        payload = state.get("assistant_retry_payload")
        if not isinstance(payload, dict):
            return
        # Retries must keep the original idempotency payload and render using
        # that same output mode, even if the selector changed after failure.
        mode = payload.get("output_mode", mode)
        text = (failed_turn or {}).get("text")
        assessment_id = payload.get("scam_assessment_id")
        # A failed STT request consumed its short-lived upload. Retry with the
        # same turn identity and a fresh upload, held only in session memory.
        if (failed_turn or {}).get("needs_audio_upload") and state.get("voice_retry_audio"):
            recording = state["voice_retry_audio"]
            _record_voice_diagnostic(state, "upload_started", payload.get("client_turn_id"), "audio_retry")
            uploaded = api.upload_audio(recording["name"], recording["raw"], recording["type"])
            _record_voice_diagnostic(state, "upload_completed", payload.get("client_turn_id"), "audio_retry")
            payload = {**payload, "audio_upload_id": uploaded["audio_upload_id"]}
            state["assistant_retry_payload"] = payload
            failed_turn["needs_audio_upload"] = False
        response = _send_turn_with_spinner(api, st, state, payload)
    elif audio is not None:
        # A new recording starts a new assistant input flow. We cannot inspect
        # its content without transcribing it, so do not attach stale context.
        clear_pending_scam_assessment(state)
        raw = audio.getvalue()
        if not raw:
            st.warning("No audio was captured. Try recording again.")
            return
        state["voice_retry_audio"] = {
            "raw": raw, "name": getattr(audio, "name", "question.wav") or "question.wav",
            "type": getattr(audio, "type", None) or "audio/wav",
        }
        # Establish the turn identity before upload so a failed upload can be
        # retried deliberately without creating a different assistant turn.
        payload = build_speech_turn_request(
            state["conversation_id"], str(uuid4()), mode,
            state["language_hint"], response_language,
            attachments=attachments_payload,
        )
        if state.get("active_voice_component_event"):
            payload["client_turn_id"] = state["active_voice_component_event"]
        _bind_account_review(state, payload)
        state["assistant_retry_payload"] = payload
        state["assistant_failed_turn"] = {
            "client_turn_id": payload["client_turn_id"],
            "text": "Voice message (transcript will be restored from the saved turn)",
            "is_voice": True, "error": None, "status": "processing",
            "recovery_attempted": False, "needs_audio_upload": True,
        }
        with st.status("Uploading voice audio…", expanded=False) as voice_status:
            state["voice_processing_stage"] = "upload"
            _record_voice_diagnostic(state, "upload_started", payload["client_turn_id"], "audio")
            try:
                uploaded = api.upload_audio(
                    getattr(audio, "name", "question.wav") or "question.wav",
                    raw,
                    getattr(audio, "type", None) or "audio/wav",
                )
            except APIError:
                state["assistant_failed_turn"].update(status="failed", error="Audio upload failed. Retry this message.")
                raise
            state["voice_processing_stage"] = "assistant"
            _record_voice_diagnostic(state, "upload_completed", payload["client_turn_id"], "audio")
            voice_status.update(label="Transcribing and preparing assistant response…", state="running", expanded=False)
            payload["audio_upload_id"] = uploaded["audio_upload_id"]
            state["assistant_failed_turn"]["needs_audio_upload"] = False
            response = _send_turn(api, state, payload)
            state.pop("voice_processing_stage", None)
            voice_status.update(label="Voice response ready", state="complete", expanded=False)
    else:
        state.pop("voice_retry_audio", None)
        assessment_id = assessment_reference_for_text(
            state, state["conversation_id"], text,
        )
        payload = build_text_turn_request(
            state["conversation_id"], text, mode,
            state["language_hint"], response_language,
            scam_assessment_id=assessment_id,
            attachments=attachments_payload,
        )
        if state.get("active_voice_component_event"):
            payload["client_turn_id"] = state["active_voice_component_event"]
        _bind_account_review(state, payload)
        state["assistant_retry_payload"] = payload
        state["assistant_failed_turn"] = {
            "client_turn_id": payload["client_turn_id"],
            "text": text.strip(),
            "is_voice": False,
            "error": None,
            "status": "processing",
            "recovery_attempted": False,
        }
        response = _send_turn_with_spinner(api, st, state, payload)

    persisted_input_text = text
    if pending_docs and text:
        from services.document_processing_service import format_document_context_for_prompt
        doc_ctx = format_document_context_for_prompt(pending_docs)
        if doc_ctx:
            persisted_input_text = f"{doc_ctx}\n\nCustomer Message:\n{text}"

    messages = messages_from_turn_response(response, input_text=persisted_input_text)
    state["pending_attachments"] = []
    state["uploaded_attachment_hashes"] = set()
    failed_id = (failed_turn or {}).get("client_turn_id")
    if failed_id:
        state["conversation_messages"] = [
            item for item in state["conversation_messages"]
            if item.get("id") != f"failed-{failed_id}"
        ]
    existing_message_ids = {
        item.get("id") for item in state["conversation_messages"]
    }
    state["conversation_messages"].extend(
        item for item in messages if item.get("id") not in existing_message_ids
    )
    state["assistant_retry_payload"] = None
    state.pop("voice_retry_audio", None)
    state["assistant_failed_turn"] = None
    assessment = response.get("scam_assessment")
    if isinstance(assessment, dict) and assessment.get("assessment_id"):
        state["latest_scam_assessment"] = assessment
        bind_pending_scam_assessment(
            state, assessment["assessment_id"], state["conversation_id"],
        )
    elif assessment_id:
        clear_pending_scam_assessment(state)
    for item in response.get("tool_results") or []:
        if not isinstance(item, dict):
            continue
        nav = (item.get("data") or {}).get("ui_navigation")
        if isinstance(nav, dict):
            subfeat = nav.get("subfeature")
            target_feat = None
            if subfeat in FEATURE_SUBFEATURES.get("Banking Information", ()):
                target_feat = "Banking Information"
            elif subfeat in FEATURE_SUBFEATURES.get("Account Opening Assistant", ()):
                target_feat = "Account Opening Assistant"
            if target_feat == "Account Opening Assistant":
                state["account_application_cache"] = None
                continue
            if target_feat and subfeat:
                request_navigation(
                    state, target_feat, subfeat, step=nav.get("step"),
                )
    state["loaded_conversation_id"] = state["conversation_id"]
    state["last_audio_hash"] = None
    state["audio_widget_epoch"] += 1
    audio_message = messages[-1]
    if mode in {"speech", "speech_and_text"} and not audio_message.get("speech_audio_id"):
        st.warning("Your text response is ready, but speech could not be generated.")
    if mode == "text":
        audio_message["speech_audio_id"] = None
    if mode in {"speech", "speech_and_text"} and audio_message.get("speech_audio_id"):
        try:
            audio_bytes, media_type = api.fetch_audio(audio_message["speech_audio_id"])
            audio_message["speech_media_type"] = media_type
            state["audio_cache"][audio_message["id"]] = audio_bytes
            state["audio_playback_visible"][audio_message["id"]] = True
            state["autoplay_audio_id"] = audio_message["id"]
            if mode == "speech":
                audio_message["text"] = ""
        except APIError as exc:
            st.warning("Your text response is ready, but its audio could not be loaded.")
            if exc.status_code == 401:
                _api_error(st, state, exc)
        except Exception:
            # The assistant turn and its text are already saved locally. A
            # playback retrieval problem is optional and must not fail the turn.
            st.warning("Your text response is ready, but its audio could not be loaded.")
    event_id = state.pop("active_voice_component_event", None)
    if event_id:
        cached_audio = state["audio_cache"].get(audio_message["id"])
        state["voice_component_response"] = {
            "id": event_id,
            "audio": base64.b64encode(cached_audio).decode("ascii") if cached_audio else None,
            "media_type": audio_message.get("speech_media_type", "audio/wav"),
        }
        if cached_audio:
            _record_voice_diagnostic(state, "response_playback_queued", event_id, "audio")
        state["autoplay_audio_id"] = None
    st.rerun()


def _bind_account_review(state, payload):
    binding = state.get("account_review_for_turn")
    if (isinstance(binding, dict) and not payload.get("attachments")
            and binding.get("conversation_id") == str(state.get("conversation_id"))):
        payload["account_review"] = {key: binding[key] for key in ("application_ref", "review_digest")}


def _handle_voice_event(api, st, state, event):
    event_id = event.get("id")
    processed = state.setdefault("voice_component_processed_ids", set())
    try:
        event_id = str(UUID(str(event_id)))
        if event_id in processed or event_id == state.get("voice_component_event_id"):
            _record_voice_diagnostic(state, "duplicate_event_rejected", event_id, event.get("kind"))
            return False
        if event.get("kind") not in {"wake", "audio"}:
            raise ValueError("Unknown voice event.")
        processed.add(event_id)
        _record_voice_diagnostic(state, "component_event_received", event_id, event["kind"])
        state["voice_component_event_id"] = event_id
        state["active_voice_component_event"] = event_id
        if event["kind"] == "wake":
            _submit(api, st, state, text="Hey Vaani")
        else:
            _submit_recorded_audio(api, st, state, decode_voice_event(event))
        return True
    except (APIError, ValueError) as exc:
        state.pop("active_voice_component_event", None)
        stage = state.pop("voice_processing_stage", None)
        code = (exc.payload or {}).get("error_code", "") if isinstance(exc, APIError) and isinstance(exc.payload, dict) else ""
        if stage == "upload":
            message = "Audio upload failed. Check your connection and record again."
        elif isinstance(code, str) and (code.startswith("SPEECH_") or code in {"EMPTY_TRANSCRIPT", "AUDIO_UPLOAD_UNAVAILABLE"}):
            message = "Transcription failed. Use Retry this message or record again."
        else:
            message = "Voice request failed. Use Retry this message or record again."
        state["voice_component_response"] = {"id": event_id, "error": True, "message": message}
        if isinstance(exc, APIError):
            _api_error(st, state, exc)
        else:
            st.warning(str(exc))
        st.rerun()


def _recover_interrupted_turn(api, st, state):
    """Reconcile a turn whose Streamlit run ended while its POST was pending."""
    pending = state.get("assistant_failed_turn") or {}
    if (
        pending.get("status") != "processing"
        or pending.get("recovery_attempted")
        or not state.get("assistant_retry_payload")
    ):
        return False
    pending["recovery_attempted"] = True
    try:
        _submit(api, st, state, retry=True)
        return True
    except APIError as exc:
        if exc.status_code == 401:
            _api_error(st, state, exc)
        return False
    except ValueError as exc:
        pending.update(status="failed", error=str(exc))
        return False


def _handle_file_upload(st, api, state, uploaded_file, is_image: bool = False):
    content = uploaded_file.getvalue()
    if not isinstance(content, (bytes, bytearray)):
        content = bytes(content) if content else b""
    if not content:
        st.warning("The selected file is empty.")
        return
    if len(content) > 10 * 1024 * 1024:
        st.error("File exceeds maximum size of 10MB.")
        return

    file_hash = hashlib.sha256(content).hexdigest()
    seen_hashes = state.setdefault("uploaded_attachment_hashes", set())
    if file_hash in seen_hashes:
        return

    _ensure_conversation(api, state)
    conv_id = state.get("conversation_id")
    try:
        with st.spinner("Processing attachment…"):
            result = api.upload_attachment(
                uploaded_file.name,
                content,
                uploaded_file.type or ("image/png" if is_image else "application/octet-stream"),
                conversation_id=conv_id,
            )
            seen_hashes.add(file_hash)
            state.setdefault("pending_attachments", []).append(result)
            if is_image:
                state["img_upload_epoch"] = state.get("img_upload_epoch", 0) + 1
            else:
                state["doc_upload_epoch"] = state.get("doc_upload_epoch", 0) + 1
            state["composer_menu_epoch"] = state.get("composer_menu_epoch", 0) + 1
            st.rerun()
    except APIError as exc:
        if exc.status_code == 401:
            sign_out(state)
            st.rerun()
        st.error(str(exc))
    except Exception as exc:
        st.error(f"Failed to process attachment: {exc}")


def render(st, api, state, *, compact=False):
    # Capture what the user saw BEFORE this interaction refreshes the review.
    binding = state.get("account_chat_confirmation")
    state["account_review_for_turn"] = dict(binding) if isinstance(binding, dict) else None
    customer = state.get("customer") or {}
    if compact:
        st.caption("Speak or type in Kannada, English, Hindi, or a code-mixed conversation.")
    else:
        st.markdown("<div class='section-eyebrow'>VAANISHIELD · VOICE BANKING ASSISTANT</div>", unsafe_allow_html=True)
        st.title("How can I help you today?")
        st.caption("Speak or type in Kannada, English, Hindi, or a code-mixed conversation.")
        left, right = st.columns([3, 1])
        with left:
            if customer.get("display_name"):
                st.markdown(f"**{customer['display_name']}** · Demo session")
        with right:
            st.caption("Voice and text responses are processed by the configured assistant services.")
        st.divider()

    try:
        _load_history(api, state)
    except APIError as exc:
        _api_error(st, state, exc)
        return

    # If Streamlit reran after the backend accepted a request but before this
    # run saved its response in session state, replay the same client_turn_id.
    _recover_interrupted_turn(api, st, state)

    risk_assessment_id = transaction_risk_reference_for_conversation(
        state, state.get("conversation_id"),
    )
    if risk_assessment_id and st.button(
        "Explain this transaction-risk assessment",
        key="assistant-explain-transaction-risk",
    ):
        state["transaction_risk_follow_up_assessment_id"] = None
        state["transaction_risk_follow_up_conversation_id"] = None
        try:
            _submit(
                api, st, state,
                text=f"Explain this transaction risk assessment {risk_assessment_id}",
            )
        except APIError as exc:
            _api_error(st, state, exc)
        except ValueError as exc:
            st.warning(str(exc))

    with st.container(key="copilot-chat-scroll-area"):
        submitted = state.get("account_application_last_submitted")
        submitted_ref = state.get("account_chat_submitted_ref") or (submitted.get("application_ref") if isinstance(submitted, dict) else None)
        if submitted_ref:
            try:
                status = api.get_account_application_status(application_ref=submitted_ref)
                if status.get("found") and status.get("status") == "SUBMITTED":
                    st.success(f"Application {status['application_ref']} was submitted to the demo workflow. No account has been opened.")
            except APIError:
                st.caption("Application submission status is temporarily unavailable.")
        render_messages(st, state["conversation_messages"], state, api=api)
        failed_turn = state.get("assistant_failed_turn")
        if failed_turn and state.get("assistant_retry_payload"):
            with st.container(border=True):
                is_processing = failed_turn.get("status") == "processing"
                if is_processing:
                    st.info(failed_turn.get("status_message") or "Vaani is still processing this saved request. Check its response in a moment.")
                else:
                    st.error(
                        failed_turn.get("error")
                        or "Vaani could not complete this turn. The original request is saved."
                    )
                retry_label = "Check response" if is_processing else "Retry this message"
                if st.button(retry_label, key="assistant-retry-saved-turn", type="primary"):
                    try:
                        _submit(api, st, state, retry=True)
                    except APIError as exc:
                        if exc.status_code != 409 or failed_turn.get("status") != "processing":
                            _api_error(st, state, exc)
                        st.rerun()
                    except ValueError as exc:
                        failed_turn["error"] = str(exc)
                        st.rerun()
        if not state["conversation_messages"]:
            st.markdown(
                """
                <div class="chatgpt-welcome-container">
                  <div class="chatgpt-welcome-avatar">🛡️</div>
                  <h2 class="chatgpt-welcome-title">How can I help you today?</h2>
                  <p class="chatgpt-welcome-subtitle">Your intelligent banking companion. Ask about accounts, transactions, loans, spending or security.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            prompts = [
                ("💳 Check my balance", "What is my account balance?"),
                ("📊 Analyze spending", "Analyze my spending this month and show key categories"),
                ("📄 View active loans", "Show my active loans and repayment progress"),
                ("🛡️ Security alerts", "Are there any recent security alerts on my account?"),
            ]
            prompt_cols = st.columns(2)
            for idx, (label, full_prompt) in enumerate(prompts):
                col = prompt_cols[idx % 2]
                key_id = f"copilot-dock-prompt-{label}" if compact else f"copilot-wide-prompt-{label}"
                if col.button(label, key=key_id, width="stretch"):
                    try:
                        _submit(api, st, state, text=full_prompt)
                    except APIError as exc:
                        _api_error(st, state, exc)
                        st.rerun()
                    except ValueError as exc:
                        st.warning(str(exc))
        if state.get("conversation_has_more") and st.button("Load earlier messages"):
            try:
                page = api.get_conversation_turns(
                    state["conversation_id"], limit=100,
                    cursor=state.get("conversation_cursor"),
                )
                earlier = [message_from_history_turn(turn) for turn in page.get("items", [])]
                state["conversation_messages"] = earlier + state["conversation_messages"]
                state["conversation_cursor"] = page.get("next_cursor")
                state["conversation_has_more"] = bool(page.get("has_more"))
                st.rerun()
            except APIError as exc:
                _api_error(st, state, exc)

    if not compact:
        with st.expander("⚙️ Assistant format & language settings", expanded=False):
            col1, col2 = st.columns([1.1, 2])
            with col1:
                from frontend.session_state import on_assistant_response_mode_change
                mode_label = st.selectbox(
                    "Response format", ["Speech", "Text", "Speech + Text"],
                    index={"speech": 0, "text": 1, "speech_and_text": 2}.get(state["response_mode"], 1),
                    key="assistant_response_mode",
                    on_change=on_assistant_response_mode_change,
                )
                state["response_mode"] = {
                    "Text": "text", "Speech": "speech", "Speech + Text": "speech_and_text",
                }[mode_label]
            with col2:
                render_language_controls(st, state)

    # Prominent Sticky Composer at bottom of Copilot
    with st.container(key="copilot-sticky-composer"):
        if state.get("kyc_chat_receipt"):
            st.caption(state["kyc_chat_receipt"])
            chat_application = state.get("kyc_chat_application")
            if isinstance(chat_application, dict) and chat_application.get("application_ref"):
                from frontend.components.account_application_chat import render_application_review
                render_application_review(st, api, state, chat_application["application_ref"], "kyc-upload")
        pending_attachments = state.get("pending_attachments") or []
        if pending_attachments:
            st.markdown("<div class='composer-attachments-preview'>", unsafe_allow_html=True)
            for idx, att in enumerate(pending_attachments):
                att_col, del_col = st.columns([5, 1])
                with att_col:
                    ftype = str(att.get("file_type", "doc")).upper()
                    icon = "🖼️" if ftype in {"JPG", "JPEG", "PNG", "WEBP", "IMAGE"} else "📄"
                    fname = att.get("file_name", "Attachment")
                    size_kb = max(1, att.get("file_size_bytes", 0) // 1024)
                    thumb_html = ""
                    if att.get("thumbnail_base64"):
                        thumb_html = f"<img src='data:image/png;base64,{att['thumbnail_base64']}' class='composer-chip-thumb' /> "
                    st.markdown(
                        f"<div class='composer-attachment-chip'>{thumb_html}{icon} <b>{fname}</b> <span class='chip-tag'>{ftype}</span> <span class='chip-size'>{size_kb} KB</span></div>",
                        unsafe_allow_html=True,
                    )
                with del_col:
                    if st.button("✕", key=f"btn_remove_pending_{idx}", help="Remove attachment"):
                        pending_attachments.pop(idx)
                        state["pending_attachments"] = pending_attachments
                        st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

        with st.container(key="copilot-composer-pill"):
            c_plus, c_input, c_mic, c_send = st.columns([0.45, 3.0, 0.5, 0.45])

            with c_plus:
                popover_key = f"composer_actions_popover_{state.get('composer_menu_epoch', 0)}"
                with st.popover("＋", help="Actions & attachments", key=popover_key):
                    st.markdown("<div class='popover-action-title'>Actions</div>", unsafe_allow_html=True)
                    if st.button("🔄 New conversation", key="popover_new_conversation", width="stretch"):
                        try:
                            conversation = create_new_conversation(api, state)
                            state["pending_attachments"] = []
                            state["floating-assistant-conversation"] = conversation["conversation_id"]
                            state["composer_menu_epoch"] = state.get("composer_menu_epoch", 0) + 1
                            st.rerun()
                        except APIError as exc:
                            if exc.status_code == 401:
                                sign_out(state)
                                st.rerun()
                            st.error("The conversation could not be created. Please try again.")

                    from frontend.components.account_application_chat import render_kyc_upload
                    render_kyc_upload(st, api, state)

                    st.divider()
                    st.markdown("<div class='popover-action-title'>📄 Add files</div>", unsafe_allow_html=True)
                    doc_file = st.file_uploader(
                        "Upload Document",
                        type=["pdf", "docx", "txt", "csv", "xlsx"],
                        key=f"composer_doc_file_{state.get('doc_upload_epoch', 0)}",
                        label_visibility="collapsed",
                        help="PDF, DOCX, TXT, CSV, or XLSX up to 10MB",
                    )
                    if doc_file is not None:
                        _handle_file_upload(st, api, state, doc_file, is_image=False)

                    st.divider()
                    st.markdown("<div class='popover-action-title'>🖼️ Add photos</div>", unsafe_allow_html=True)
                    img_file = st.file_uploader(
                        "Upload Photo",
                        type=["jpg", "jpeg", "png", "webp"],
                        key=f"composer_img_file_{state.get('img_upload_epoch', 0)}",
                        label_visibility="collapsed",
                        help="JPG, JPEG, PNG, or WEBP up to 10MB",
                    )
                    if img_file is not None:
                        _handle_file_upload(st, api, state, img_file, is_image=True)

            input_key = f"copilot_text_input_{state.get('input_epoch', 0)}"

            with c_input:
                entered_text = st.text_area(
                    "Message Vaani...",
                    key=input_key,
                    placeholder="Message Vaani...",
                    label_visibility="collapsed",
                    height=38,
                    disabled=bool(state.get("assistant_submission_in_progress")),
                )

            with c_mic:
                voice_event = render_compact_voice(state)

            with c_send:
                send_clicked = st.button("➤", key="copilot_send_button", type="primary", help="Send message")

        # Enter submits once; Shift+Enter retains the textarea's newline behavior.
        st.components.v1.html(
            """
            <script>
            (function() {
                const parentDoc = window.parent.document;
                const renderToken = '__COMPOSER_RENDER_TOKEN__';
                if (window.parent.__vaaniComposerRenderToken !== renderToken) {
                    window.parent.__vaaniComposerRenderToken = renderToken;
                    const previousButton = parentDoc.querySelector('.st-key-copilot_send_button button');
                    const previousArea = parentDoc.querySelector('.st-key-copilot-composer-pill textarea');
                    if (previousButton) {
                        delete previousButton.dataset.submitting;
                        previousButton.disabled = false;
                    }
                    if (previousArea) {
                        delete previousArea.dataset.submitting;
                        previousArea.disabled = false;
                    }
                }
                function setupEnter() {
                    const pill = parentDoc.querySelector('.st-key-copilot-composer-pill');
                    if (!pill) return;
                    const area = pill.querySelector('textarea');
                    if (area && !area.dataset.enterBound) {
                        area.dataset.enterBound = 'true';
                        area.addEventListener('keydown', function(e) {
                            if (e.key === 'Enter' && e.shiftKey) return;
                            if (e.key === 'Enter' && !e.shiftKey) {
                                e.preventDefault();
                                if (!area.value.trim() || area.dataset.submitting === 'true') return;
                                const sendBtn = parentDoc.querySelector('.st-key-copilot_send_button button');
                                if (sendBtn && !sendBtn.disabled) {
                                    sendBtn.click();
                                }
                            }
                        });
                    }
                    const sendBtn = parentDoc.querySelector('.st-key-copilot_send_button button');
                    if (area && sendBtn && !sendBtn.dataset.submitBound) {
                        sendBtn.dataset.submitBound = 'true';
                        sendBtn.addEventListener('click', function(e) {
                            if (sendBtn.dataset.submitting === 'true') {
                                e.preventDefault();
                                e.stopImmediatePropagation();
                                return;
                            }
                            const hasAttachment = !!parentDoc.querySelector('.composer-attachment-chip');
                            if (!area.value.trim() && !hasAttachment) return;
                            sendBtn.dataset.submitting = 'true';
                            area.dataset.submitting = 'true';
                            area.disabled = true;
                            setTimeout(function() { sendBtn.disabled = true; }, 0);
                        });
                    }
                }
                setupEnter();
                setTimeout(setupEnter, 300);
                new MutationObserver(setupEnter).observe(parentDoc.body, {childList: true, subtree: true});
            })();
            </script>
            """.replace("__COMPOSER_RENDER_TOKEN__", str(state.get("input_epoch", 0))),
            height=0,
        )

        if isinstance(voice_event, dict):
            _handle_voice_event(api, st, state, voice_event)

        if send_clicked:
            submitted_text = (entered_text or "").strip()
            has_pending = bool(state.get("pending_attachments"))
            if not submitted_text and has_pending:
                submitted_text = "Please analyze the attached document."
            if submitted_text:
                state["input_epoch"] = state.get("input_epoch", 0) + 1
                try:
                    _submit(api, st, state, text=submitted_text)
                except APIError as exc:
                    _api_error(st, state, exc)
                    st.rerun()
                except ValueError as exc:
                    st.warning(str(exc))

