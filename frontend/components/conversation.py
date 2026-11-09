"""Conversation conversion and rendering helpers."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from uuid import uuid4

from frontend.api_client import APIError
from frontend.components.fintech_ui import parse_markdown_table_to_compact_cards


_RISK_STYLE = {
    "LOW": ("🟢", "LOW"),
    "MEDIUM": ("🟡", "MEDIUM"),
    "HIGH": ("🟠", "HIGH"),
    "CRITICAL": ("🔴", "CRITICAL"),
    "UNKNOWN": ("⚪", "UNKNOWN"),
}


def render_scam_assessment(st, assessment: dict):
    """Render backend-authored scam fields as a distinct, cautious report card."""
    if not isinstance(assessment, dict):
        return
    risk = str(assessment.get("risk_level", "UNKNOWN")).upper()
    icon, label = _RISK_STYLE.get(risk, _RISK_STYLE["UNKNOWN"])
    source = {
        "typed_message": "Text message",
        "voice_call": "Voice / call transcript",
    }.get(assessment.get("source_type"), "Unknown source")
    created = assessment.get("created_at")

    with st.container(border=True):
        st.markdown("#### Security assessment")
        st.caption(f"Source: {source}" + (f"  |  {created}" if created else ""))
        st.caption("Automated screening only. This is not confirmation that fraud occurred or that a person is a criminal.")
        st.markdown(f"### {icon} {label} risk")

        st.markdown("**Explanation**")
        st.write(assessment.get("explanation") or "No explanation was returned.")

        st.markdown("**Detected indicators and evidence**")
        indicators = assessment.get("indicators") or []
        if not indicators:
            st.write("No specific indicators were identified.")
        for indicator in indicators:
            if not isinstance(indicator, dict):
                continue
            category = indicator.get("category", "Indicator")
            severity = str(indicator.get("severity", "UNKNOWN")).upper()
            st.markdown(f"**{category} · {severity}**")
            if indicator.get("explanation"):
                st.write(indicator["explanation"])
            for evidence in indicator.get("evidence") or []:
                st.text(f"Evidence: {evidence}")

        st.caption(f"Evidence strength: {assessment.get('evidence_strength', 'UNKNOWN')}")
        st.markdown("**Warning**")
        st.write(assessment.get("warning") or "No specific warning was returned.")
        st.markdown("**Recommended actions**")
        actions = assessment.get("recommended_steps") or []
        if actions:
            for action in actions:
                st.write(f"• {action}")
        else:
            st.write("No additional actions were returned.")


def render_transaction_risk_assessment(st, result: dict, transaction: dict | None = None):
    """Render authoritative transaction-risk fields without calculating or rewriting them."""
    if not isinstance(result, dict):
        return
    assessment = result.get("assessment")
    if not isinstance(assessment, dict):
        return
    risk = str(assessment.get("risk_level", "UNKNOWN")).upper()
    style = {
        "LOW": ("success", "LOW"),
        "MEDIUM": ("info", "MEDIUM"),
        "HIGH": ("warning", "HIGH"),
        "CRITICAL": ("error", "CRITICAL"),
        "UNKNOWN": ("info", "UNKNOWN"),
    }.get(risk, ("info", "UNKNOWN"))
    transaction = transaction or result.get("transaction") or {}
    created = result.get("created_at")
    ref = result.get("transaction_ref") or transaction.get("transaction_ref")

    with st.container(border=True):
        st.markdown("#### Transaction risk assessment")
        st.caption("Automated transaction screening. This is not a bank-authorized fraud determination or proof of fraud.")
        if created:
            st.caption(f"Assessed: {created}")
        if ref:
            st.caption(f"Transaction reference: {ref}")
        if transaction:
            merchant = transaction.get("merchant") or "Merchant unavailable"
            amount = transaction.get("amount")
            amount_text = f"₹{amount:,.2f}" if isinstance(amount, (int, float)) else "Amount unavailable"
            st.write(f"**{merchant}** · {amount_text}")
            st.caption(
                " · ".join(str(value) for value in (
                    transaction.get("transaction_date"),
                    transaction.get("transaction_type"),
                    transaction.get("status"),
                ) if value)
            )
        getattr(st, style[0])(f"{style[1]} screening level")
        st.metric(
            "Rule-based screening score (not a fraud probability)",
            assessment.get("risk_score", "UNKNOWN"),
        )

        st.markdown("**Explanation**")
        st.write(assessment.get("explanation") or "No explanation was returned.")
        st.markdown("**Indicators and evidence**")
        indicators = assessment.get("indicators") or []
        if not indicators:
            st.write("No specific indicators were returned.")
        for indicator in indicators:
            if not isinstance(indicator, dict):
                continue
            st.markdown(
                f"**{indicator.get('indicator_code', 'Indicator')} · "
                f"{indicator.get('severity', 'UNKNOWN')} severity**"
            )
            st.write(indicator.get("explanation") or "No indicator explanation was returned.")
            for evidence in indicator.get("evidence") or []:
                st.text(f"Evidence: {evidence}")
        evidence = assessment.get("evidence") or []
        if evidence:
            st.markdown("**Assessment evidence**")
            for item in evidence:
                st.text(str(item))
        st.caption(f"Evidence strength: {assessment.get('evidence_strength', 'UNKNOWN')}")
        sufficiency = assessment.get("history_sufficiency") or {}
        st.caption(
            "History sufficiency: "
            f"{sufficiency.get('level', 'UNKNOWN')} · "
            f"matching records: {sufficiency.get('matching_history_count', 'UNKNOWN')} · "
            f"amount-baseline minimum: {sufficiency.get('minimum_for_amount_baseline', 'UNKNOWN')} · "
            f"coverage days: {sufficiency.get('time_coverage_days', 'UNKNOWN')} · "
            f"frequency baseline ready: {sufficiency.get('frequency_baseline_ready', 'UNKNOWN')} · "
            f"excluded records: {sufficiency.get('excluded_history_count', 'UNKNOWN')}"
        )
        st.markdown("**Warning**")
        st.write(assessment.get("warning") or "No warning was returned.")
        st.markdown("**Recommended actions**")
        actions = assessment.get("recommended_actions") or []
        if actions:
            for action in actions:
                st.write(f"• {action}")
        else:
            st.write("No recommended actions were returned.")


def render_investigation_detail(st, detail: dict):
    """Render the persisted owner-facing investigation summary and event timeline."""
    if not isinstance(detail, dict):
        return
    with st.container(border=True):
        st.markdown("#### Personal security investigation")
        st.caption("Synthetic customer-managed tracking only; no bank staff review or resolution is represented.")
        st.markdown(f"### {detail.get('title', 'Investigation')} · {detail.get('status', 'UNKNOWN')}")
        st.write(detail.get("summary") or "No summary provided.")
        st.caption(f"Reference: {detail.get('investigation_ref')} · Category: {detail.get('category')} · Updated: {detail.get('updated_at')}")
        st.markdown("**Linked assessments**")
        evidence = detail.get("evidence") or []
        if not evidence:
            st.caption("No assessment records are linked.")
        for item in evidence:
            if isinstance(item, dict):
                st.write(
                    f"{item.get('evidence_type')} · {item.get('risk_level')} · "
                    f"{item.get('evidence_strength')} · {item.get('created_at')}"
                )
                st.caption(f"Assessment reference: {item.get('assessment_id')} · {item.get('summary')}")
        st.markdown("**Event timeline**")
        events = detail.get("events") or []
        if not events:
            st.caption("No events have been recorded.")
        for event in events:
            if not isinstance(event, dict):
                continue
            st.markdown(f"**{event.get('occurred_at')} · {event.get('event_type')}** — {event.get('summary')}")
            if event.get("from_status") or event.get("to_status"):
                st.caption(f"Status: {event.get('from_status') or '—'} → {event.get('to_status') or '—'}")
            if event.get("note"):
                st.write(event["note"])


def message_from_history_turn(turn: dict) -> dict:
    role = turn.get("role", "assistant")
    text = (
        turn.get("user_text_or_transcript")
        if role == "user"
        else turn.get("assistant_response_text")
    )
    return {
        "id": turn.get("turn_id") or str(uuid4()),
        "role": role,
        "text": text or "",
        "transcript": text if role == "user" and turn.get("input_type") == "audio" else None,
        "input_mode": "speech" if turn.get("input_type") == "audio" else "text",
        "detected_languages": turn.get("detected_languages") or ([turn["input_language"]] if turn.get("input_language") else []),
        "is_code_mixed": bool(turn.get("is_code_mixed")),
        "speech_audio_id": None,
        "speech_media_type": None,
        "response_language": turn.get("response_language"),
        "history_only": True,
        "conversation_id": turn.get("conversation_id"),
        "turn_id": turn.get("turn_id"),
    }


def messages_from_turn_response(response: dict, input_text: str | None = None):
    detected = [
        item.get("language")
        for item in response.get("detected_languages", [])
        if isinstance(item, dict) and item.get("language")
    ]
    user_text = response.get("transcript") or input_text or ""
    audio = response.get("speech_output") or {}
    turn_id = response.get("turn_id") or str(uuid4())
    conversation_id = response.get("conversation_id")
    ui_navigation = None
    account_review = None
    for item in response.get("tool_results") or []:
        if isinstance(item, dict) and isinstance((item.get("data") or {}).get("ui_navigation"), dict):
            ui_navigation = item["data"]["ui_navigation"]
        if isinstance(item, dict) and isinstance((item.get("data") or {}).get("account_review"), dict):
            account_review = item["data"]["account_review"]
    assistant_msg = {
        "id": f"assistant-{turn_id}",
        "role": "assistant",
        "text": response.get("assistant_text") or "",
        "transcript": None,
        "input_mode": None,
        "detected_languages": [],
        "is_code_mixed": False,
        "speech_audio_id": audio.get("audio_url"),
        "speech_media_type": audio.get("media_type", "audio/wav"),
        "response_language": response.get("response_language"),
        "history_only": False,
        "conversation_id": conversation_id,
        "scam_assessment": response.get("scam_assessment"),
        "pending_action": response.get("pending_action"),
        "account_review": account_review,
        "support_close_confirmation": response.get("support_close_confirmation"),
        "turn_id": turn_id,
    }
    if ui_navigation is not None:
        assistant_msg["ui_navigation"] = ui_navigation
    return [
        {
            "id": f"user-{turn_id}",
            "role": "user",
            "text": user_text,
            "transcript": response.get("transcript"),
            "input_mode": response.get("input_mode", "text"),
            "detected_languages": detected,
            "is_code_mixed": bool(response.get("is_code_mixed")),
            "speech_audio_id": None,
            "speech_media_type": None,
            "response_language": None,
            "history_only": False,
            "conversation_id": conversation_id,
            "turn_id": response.get("assistant_turn_id"),
        },
        assistant_msg,
    ]


def submit_support_confirmation(api, state, pending_action, *, confirmed):
    """Submit one explicit proposal decision, bound to its exact turn/context."""
    conversation_id = str(pending_action.get("conversation_id") or "")
    original_turn_id = str(pending_action.get("original_turn_id") or "")
    confirmation_id = str(pending_action.get("confirmation_id") or "")
    action = pending_action.get("action") or {}
    if (
        not confirmation_id or not conversation_id or not original_turn_id
        or action.get("action_type") != "create_support_ticket"
    ):
        raise ValueError("This support proposal is incomplete and cannot be submitted.")
    if conversation_id != str(state.get("conversation_id")):
        raise ValueError("This proposal belongs to another conversation. Reopen that conversation to act on it.")
    outcomes = state.setdefault("support_action_outcomes", {})
    key = f"create:{confirmation_id}"
    prior = outcomes.get(key)
    if isinstance(prior, dict) and prior.get("status") in {"executed", "declined", "expired"}:
        return prior
    if isinstance(prior, dict) and prior.get("status") == "processing":
        return prior
    outcomes[key] = {"status": "processing"}
    try:
        result = api.confirm_assistant_action(
            confirmation_id,
            confirmed=confirmed,
            conversation_id=conversation_id,
            original_turn_id=original_turn_id,
            action_type="create_support_ticket",
        )
    except Exception:
        outcomes[key] = {"status": "failed"}
        raise
    status = result.get("status") if isinstance(result, dict) else None
    if status not in {"executed", "declined", "expired"}:
        outcomes[key] = {"status": "failed"}
        raise ValueError("The server did not confirm the support action outcome.")
    outcomes[key] = result
    if status == "executed":
        ticket_ref = ((result.get("action_result") or {}).get("data") or {}).get("ticket_ref")
        if ticket_ref:
            state["support_selected_ticket_ref"] = ticket_ref
            state["support_ticket_filter_key"] = None
    return result


def submit_support_closure(api, state, confirmation):
    """Close only the owner-visible ticket described by this turn's proposal."""
    conversation_id = str(confirmation.get("conversation_id") or "")
    ticket_ref = confirmation.get("ticket_ref")
    if conversation_id != str(state.get("conversation_id")):
        raise ValueError("This closure request belongs to another conversation.")
    if confirmation.get("confirmation_method") != "button" or not ticket_ref:
        raise ValueError("This closure request is incomplete.")
    outcomes = state.setdefault("support_action_outcomes", {})
    key = f"close:{conversation_id}:{ticket_ref}"
    prior = outcomes.get(key)
    if isinstance(prior, dict) and prior.get("status") == "closed":
        return prior
    if isinstance(prior, dict) and prior.get("status") == "processing":
        return prior
    outcomes[key] = {"status": "processing"}
    try:
        ticket = api.close_support_ticket(ticket_ref)
    except Exception:
        outcomes[key] = {"status": "failed"}
        raise
    if not isinstance(ticket, dict) or ticket.get("status") != "CLOSED":
        outcomes[key] = {"status": "failed"}
        raise ValueError("The server did not confirm ticket closure.")
    outcomes[key] = {"status": "closed", "ticket": ticket}
    state["support_ticket_filter_key"] = None
    return outcomes[key]


def support_confirmation_state(pending_action, outcome=None, *, now=None):
    """Return pending only when a proposal has a valid, future server expiry."""
    status = outcome.get("status") if isinstance(outcome, dict) else None
    if status in {"executed", "declined", "expired", "rejected", "processing"}:
        return status
    if status in {"failed", "outcome_unknown"}:
        return "outcome_unknown"
    expires_at = pending_action.get("expires_at") if isinstance(pending_action, dict) else None
    if not isinstance(expires_at, str) or not expires_at.strip():
        return "invalid"
    try:
        expiry = datetime.fromisoformat(expires_at.strip().replace("Z", "+00:00"))
        if expiry.tzinfo is None:
            return "invalid"
        current = now or datetime.now(timezone.utc)
        if current.tzinfo is None:
            current = current.replace(tzinfo=timezone.utc)
        return "expired" if expiry <= current else "pending"
    except (TypeError, ValueError):
        return "invalid"


def _render_submitted_support_ticket(st, message, ticket_ref, state):
    if not ticket_ref:
        return
    message["pending_action"] = None
    message["support_submitted_ticket_ref"] = ticket_ref
    with st.container(border=True):
        st.markdown("#### Support ticket submitted")
        st.success(f"Ticket submitted: {ticket_ref}")
        if st.button(
            "📋 View in My Support Tickets",
            key=f"view-ticket-submitted-{ticket_ref}", type="primary",
        ):
            from frontend.navigation import request_navigation
            request_navigation(state, "Customer Support", "Track Complaints")
            state["support_selected_ticket_ref"] = ticket_ref
            state["support_ticket_filter_key"] = None
            st.rerun()


def _render_support_action(st, message, api, state):
    pending = message.get("pending_action")
    close_confirmation = message.get("support_close_confirmation")
    conversation_id = str(message.get("conversation_id") or "")
    if pending:
        action = pending.get("action") or {}
        if str(action.get("action_type", "")).endswith("investigation") or action.get("action_type") in {
            "create_investigation", "link_investigation_evidence", "add_investigation_note",
            "change_investigation_status", "close_investigation",
        }:
            _render_investigation_action(st, pending, api, state)
            return
        # Context is checked again by submit_support_confirmation. Hiding a
        # stale card here prevents buttons from appearing in another chat.
        if str(pending.get("conversation_id")) != str(state.get("conversation_id")):
            st.caption("This proposal belongs to another conversation and is no longer actionable here.")
            return
        outcome_key = f"create:{pending.get('confirmation_id')}"
        outcome = state.get("support_action_outcomes", {}).get(outcome_key, {})
        confirmation_state = support_confirmation_state(pending, outcome)
        if confirmation_state == "executed":
            result = outcome.get("action_result") or {}
            ticket_ref = ((result.get("data") or {}).get("ticket_ref"))
            _render_submitted_support_ticket(st, message, ticket_ref, state)
            return
        with st.container(border=True):
            st.markdown("#### Support ticket proposal")
            if confirmation_state == "pending":
                st.warning("This ticket has **not** been submitted.")
            st.write(f"**Category:** {str(action.get('category', '—')).replace('_', ' ').title()}")
            st.write(f"**Subject:** {action.get('subject', '—')}")
            st.write(f"**Priority:** {action.get('priority', '—')}")

            amount_val = action.get("amount")
            if amount_val is None:
                amt_match = re.search(r"(?:₹|rs\.?|amount:?\s*₹?)\s*([0-9]+(?:,[0-9]+)*(?:\.[0-9]{1,2})?)", str(action.get("description", "")), re.I)
                if amt_match:
                    amount_val = amt_match.group(1)
            if amount_val is not None:
                st.write(f"**Amount:** ₹{amount_val if isinstance(amount_val, str) else f'{amount_val:,.2f}'}")

            st.write(f"**Description:** {action.get('description') or 'No additional description provided.'}")
            st.caption("Synthetic demo support ticket. Confirming submits the proposal once.")
            outcome_key = f"create:{pending.get('confirmation_id')}"
            outcome = state.get("support_action_outcomes", {}).get(outcome_key, {})
            confirmation_state = support_confirmation_state(pending, outcome)
            if confirmation_state == "executed":
                result = outcome.get("action_result") or {}
                data = result.get("data") or {}
                ticket_ref = data.get("ticket_ref")
                st.success(f"Ticket submitted: {ticket_ref}" if ticket_ref else "The server confirmed ticket submission.")
                if ticket_ref:
                    if st.button("📋 View in My Support Tickets", key=f"view-ticket-{outcome_key}", type="primary"):
                        from frontend.navigation import request_navigation
                        request_navigation(state, "Customer Support", "Track Complaints")
                        state["support_selected_ticket_ref"] = ticket_ref
                        st.rerun()
            elif confirmation_state == "declined":
                st.info("Proposal declined. No ticket was created.")
            elif confirmation_state == "expired":
                st.info("This proposal expired. Ask Vaani AI to prepare a new one.")
            elif confirmation_state == "rejected":
                st.warning("This proposal is no longer available. Refresh the conversation before taking another action.")
            elif confirmation_state == "processing":
                st.info("Submitting your decision…")
            elif confirmation_state == "outcome_unknown":
                st.warning("We could not verify the action result. Check My Support Tickets before asking Vaani AI to prepare another proposal.")
            elif confirmation_state == "invalid":
                st.info("This proposal's expiry could not be verified, so it cannot be confirmed here. Ask Vaani AI to prepare a new proposal.")
            elif confirmation_state == "pending":
                confirm_col, decline_col = st.columns(2)
                if confirm_col.button("Confirm & Submit", key=f"support-confirm-{outcome_key}", type="primary"):
                    _run_support_confirmation(st, api, state, pending, True)
                if decline_col.button("Cancel / Decline", key=f"support-decline-{outcome_key}"):
                    _run_support_confirmation(st, api, state, pending, False)
    elif close_confirmation:
        if str(close_confirmation.get("conversation_id")) != str(state.get("conversation_id")):
            st.caption("This closure request belongs to another conversation and is no longer actionable here.")
            return
        ticket_ref = close_confirmation.get("ticket_ref", "")
        outcome_key = f"close:{conversation_id}:{ticket_ref}"
        outcome = state.get("support_action_outcomes", {}).get(outcome_key, {})
        with st.container(border=True):
            st.markdown("#### Confirm ticket closure")
            st.write(f"**Ticket:** {ticket_ref} · {close_confirmation.get('subject', '')}")
            st.write(f"Current status: **{close_confirmation.get('current_status', 'UNKNOWN')}**")
            st.caption("Closing is a customer-requested demo action. It does not represent a bank resolution.")
            if outcome.get("status") == "closed":
                st.success(f"Ticket {ticket_ref} was closed by the server.")
            elif outcome.get("status") == "processing":
                st.info("Submitting closure request…")
            elif outcome.get("status") == "rejected":
                st.warning("This closure request is no longer available. Refresh the ticket status.")
            elif st.button("Confirm ticket closure", key=f"support-close-{outcome_key}", type="primary"):
                try:
                    with st.spinner("Submitting ticket closure…"):
                        ticket = submit_support_closure(api, state, close_confirmation)["ticket"]
                    st.success(f"Ticket {ticket_ref} is now {ticket['status']}.")
                    st.rerun()
                except APIError as exc:
                    if exc.status_code in {404, 409}:
                        state.setdefault("support_action_outcomes", {})[outcome_key] = {"status": "rejected"}
                        st.warning("The ticket is unavailable or its status changed. Refresh the ticket before trying again.")
                    else:
                        _show_action_api_error(st, state, exc)
                except ValueError as exc:
                    st.warning(str(exc))


def _render_investigation_action(st, pending, api, state):
    action = pending.get("action") or {}
    conversation_id = str(pending.get("conversation_id") or "")
    if conversation_id != str(state.get("conversation_id")):
        st.caption("This proposal belongs to another conversation and is no longer actionable here.")
        return
    action_type = action.get("action_type", "")
    confirmation_id = str(pending.get("confirmation_id") or "")
    key = f"investigation:{confirmation_id}"
    outcomes = state.setdefault("investigation_action_outcomes", {})
    outcome = outcomes.get(key, {})
    status = support_confirmation_state(pending, outcome)
    with st.container(border=True):
        st.markdown("#### Personal investigation update proposal")
        st.caption("Synthetic demo tracking only. This does not contact bank staff or represent a bank investigation.")
        if action_type == "create_investigation":
            st.write(f"**Title:** {action.get('title', '')}")
            st.write(f"**Category:** {str(action.get('category', '')).replace('_', ' ').title()}")
            st.write(f"**Summary:** {action.get('summary', '')}")
            st.write(f"**Linked assessments:** {len(action.get('evidence') or [])}")
        elif action_type == "add_investigation_note":
            st.write(f"**Investigation:** {action.get('investigation_ref')}")
            st.write(f"**Proposed note:** {action.get('note', '')}")
        else:
            st.write(f"**Investigation:** {action.get('investigation_ref')}")
            if action_type == "link_investigation_evidence":
                evidence = action.get("evidence") or {}
                st.write(f"Link {evidence.get('evidence_type')} assessment {evidence.get('assessment_id')}")
            elif action_type == "change_investigation_status":
                st.write(f"Change status to **{action.get('status')}**")
            elif action_type == "close_investigation":
                st.write("Close this personal tracking record.")
        if status == "executed":
            data = ((outcome.get("action_result") or {}).get("data") or {})
            reference = data.get("investigation_ref") or action.get("investigation_ref")
            st.success(f"Confirmed. Investigation reference: {reference}" if reference else "The server confirmed the update.")
        elif status == "declined":
            st.info("Proposal declined. No investigation change was made.")
        elif status == "expired":
            st.info("This proposal expired. Ask Vaani AI to prepare it again.")
        elif status in {"rejected", "outcome_unknown", "invalid"}:
            st.warning("This proposal is no longer available. Refresh the investigation before trying again.")
        elif status == "processing":
            st.info("Submitting your decision…")
        elif status == "pending":
            yes, no = st.columns(2)
            if yes.button("Confirm update", key=f"confirm-{key}", type="primary"):
                _run_investigation_confirmation(st, api, state, pending, True)
            if no.button("Decline", key=f"decline-{key}"):
                _run_investigation_confirmation(st, api, state, pending, False)


def _run_investigation_confirmation(st, api, state, pending, confirmed):
    from frontend.api_client import APIError
    action = pending.get("action") or {}
    key = f"investigation:{pending.get('confirmation_id')}"
    outcomes = state.setdefault("investigation_action_outcomes", {})
    if str(pending.get("conversation_id")) != str(state.get("conversation_id")):
        st.warning("This proposal belongs to another conversation.")
        return
    outcomes[key] = {"status": "processing"}
    try:
        result = api.confirm_investigation_action(
            pending.get("confirmation_id"), confirmed=confirmed,
            conversation_id=pending.get("conversation_id"),
            original_turn_id=pending.get("original_turn_id"),
            action_type=action.get("action_type"),
        )
        outcomes[key] = result
        if result.get("status") == "executed":
            st.success("Investigation update confirmed by the server.")
        elif result.get("status") == "declined":
            st.info("Proposal declined. No change was made.")
        else:
            st.info("This proposal expired. No change was made.")
        state["investigation_items"] = None
        state["investigation_detail"] = None
        st.rerun()
    except APIError as exc:
        outcomes[key] = {"status": "rejected" if exc.status_code in {404, 409} else "failed"}
        if exc.status_code == 401:
            from frontend.session_state import sign_out
            sign_out(state)
            st.warning("Your session expired. Sign in again to continue.")
        else:
            st.warning("The proposal could not be confirmed. Refresh and try again.")


def _show_action_api_error(st, state, exc):
    from frontend.pages.assistant import _api_error
    _api_error(st, state, exc)


def _run_support_confirmation(st, api, state, pending, confirmed):
    from frontend.api_client import APIError
    try:
        with st.spinner("Sending your decision…"):
            result = submit_support_confirmation(api, state, pending, confirmed=confirmed)
        if result["status"] == "executed":
            data = ((result.get("action_result") or {}).get("data") or {})
            ref = data.get("ticket_ref")
            st.success(f"Ticket submitted: {ref}" if ref else "The server confirmed submission.")
        elif result["status"] == "declined":
            st.info("Proposal declined. No ticket was created.")
        else:
            st.info("This proposal has expired. Ask Vaani AI to prepare a new one.")
        st.rerun()
    except APIError as exc:
        if exc.status_code == 409:
            state.setdefault("support_action_outcomes", {})[
                f"create:{pending.get('confirmation_id')}"
            ] = {"status": "rejected"}
            st.warning("This proposal expired or was already used. No additional ticket was created.")
        elif exc.status_code == 404:
            state.setdefault("support_action_outcomes", {})[
                f"create:{pending.get('confirmation_id')}"
            ] = {"status": "rejected"}
            st.warning("This proposal is unavailable or no longer belongs to this session.")
        else:
            _show_action_api_error(st, state, exc)
    except ValueError as exc:
        st.warning(str(exc))


def _normalize_markdown_text(text: str) -> str:
    """Ensure CommonMark bold delimiters (**text**) render properly even around punctuation/currency/Indic scripts."""
    if not text:
        return ""
    cleaned = re.sub(r"\*\*\s+([^*\n]+?)\s+\*\*", r"**\1**", text)
    cleaned = re.sub(r"\*\*([^*\n]+?):\*\*(?=\S)", r"**\1:** ", cleaned)
    return cleaned


def _display_user_text_and_attachments(text: str) -> tuple[str, list[dict]]:
    """Separate attached document envelopes from the user's visible message."""
    att_pattern = re.compile(
        r"\[UNTRUSTED CUSTOMER ATTACHMENT START:\s*(.+?)\s*\((.+?)\)\]\s*[\s\S]*?\[UNTRUSTED CUSTOMER ATTACHMENT END:\s*.*?\]\s*",
        re.MULTILINE,
    )
    found_attachments = []
    for match in att_pattern.finditer(text):
        fname = match.group(1).strip()
        ftype = match.group(2).strip().upper()
        found_attachments.append({"file_name": fname, "file_type": ftype})
    clean_text = att_pattern.sub("", text).strip()
    return clean_text, found_attachments


def render_messages(st, messages: list[dict], state, api=None):
    for message in messages:
        role = message.get("role", "assistant")
        with st.chat_message("user" if role == "user" else "assistant"):
            text = _normalize_markdown_text(message.get("text") or "")
            if role == "user":
                clean_user_text, att_chips = _display_user_text_and_attachments(text)
                for att in att_chips:
                    icon = "🖼️" if att["file_type"] in {"JPG", "JPEG", "PNG", "WEBP", "IMAGE"} else "📄"
                    st.markdown(
                        f"""
                        <div class="user-attached-badge">
                          <span>{icon} <b>{att['file_name']}</b></span>
                          <small>({att['file_type']})</small>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                if message.get("transcript"):
                    st.markdown(f"**Transcript**\n\n{clean_user_text or text}")
                elif clean_user_text:
                    st.markdown(clean_user_text)
                elif not att_chips and text:
                    st.markdown(text)
            elif text:
                if role == "assistant":
                    converted, result_html = parse_markdown_table_to_compact_cards(text)
                    if converted:
                        st.markdown(result_html, unsafe_allow_html=True)
                    else:
                        st.markdown(text)
                else:
                    st.markdown(text)
            elif role == "assistant" and message.get("speech_audio_id"):
                st.caption("Voice response")

            languages = message.get("detected_languages") or []
            if languages:
                joined = " · ".join(languages)
                suffix = " · code-mixed" if message.get("is_code_mixed") else ""
                st.caption(f"Detected: {joined}{suffix}")

            audio_key = message.get("id")
            audio_url = message.get("speech_audio_id")
            audio_bytes = state["audio_cache"].get(audio_key)
            if audio_url and audio_bytes:
                if state["audio_playback_visible"].get(audio_key, True):
                    should_autoplay = bool(state.get("autoplay_audio_id") == audio_key)
                    st.audio(
                        audio_bytes,
                        format=message.get("speech_media_type", "audio/wav"),
                        autoplay=should_autoplay,
                        alt="VaaniShield assistant voice response",
                    )
                    if should_autoplay:
                        state["autoplay_audio_id"] = None
                    if st.button("Stop audio", key=f"stop-{audio_key}"):
                        state["audio_playback_visible"][audio_key] = False
                        st.rerun()
                elif st.button("Replay response", key=f"replay-{audio_key}"):
                    state["audio_playback_visible"][audio_key] = True
                    st.rerun()

            if message.get("response_language"):
                st.caption(f"Response language: {message['response_language']}")

            if role == "assistant" and message.get("scam_assessment"):
                render_scam_assessment(st, message["scam_assessment"])
            if role == "assistant" and isinstance(message.get("ui_navigation"), dict):
                nav = message["ui_navigation"]
                subfeature = nav.get("subfeature")
                show_navigation = bool(subfeature)
                if subfeature == "Review & Submit":
                    ref = nav.get("application_ref")
                    show_navigation = False
                    if api is not None and ref:
                        try:
                            from services.account_application_assistant import _ready_for_review
                            current = api.get_account_application(ref)
                            show_navigation = current.get("status") != "SUBMITTED" and _ready_for_review(current)
                        except (APIError, ValueError):
                            st.caption("Application readiness is temporarily unavailable.")
                if show_navigation:
                    nav_label = f"Go to {subfeature}"
                    if st.button(nav_label, key=f"nav-{message.get('id')}-{subfeature}"):
                        from frontend.navigation import request_navigation
                        if subfeature == "Review & Submit":
                            ref = nav["application_ref"]
                            try:
                                from services.account_application_assistant import _ready_for_review
                                current = api.get_account_application(ref)
                            except (APIError, ValueError):
                                st.warning("The application could not be checked. Please try again.")
                                continue
                            if not _ready_for_review(current) or current.get("status") == "SUBMITTED":
                                st.warning("The application has changed. Ask Vaani to check it again.")
                                continue
                            state["account_application_selected_ref"] = ref
                            state["account_application_review_target_ref"] = ref
                        request_navigation(
                            state, "Account Opening Assistant" if nav.get("target") == "account_services" else "Banking Information", subfeature, step=nav.get("step"),
                        )
                        st.rerun()
            if role == "assistant" and message.get("support_submitted_ticket_ref"):
                _render_submitted_support_ticket(
                    st, message, message["support_submitted_ticket_ref"], state,
                )
            elif role == "assistant" and api and (
                message.get("pending_action") or message.get("support_close_confirmation")
            ):
                action_type = ((message.get("pending_action") or {}).get("action") or {}).get("action_type")
                if action_type in {
                    "create_investigation", "link_investigation_evidence", "add_investigation_note",
                    "change_investigation_status", "close_investigation",
                }:
                    _render_investigation_action(st, message.get("pending_action"), api, state)
                else:
                    _render_support_action(st, message, api, state)


            if role == "assistant" and api and message.get("account_review"):
                from frontend.components.account_application_chat import render_application_review
                render_application_review(st, api, state, message["account_review"]["application_ref"], message["id"])

            # Deletion is explicit and server-side. The backend soft-deletes
            # only this turn and cancels actions bound to its originating turn.
            turn_id = message.get("turn_id") or (message.get("id", "").split("-", 1)[-1] if message.get("history_only") else None)
            conversation_id = message.get("conversation_id") or state.get("conversation_id")
            if api and turn_id and conversation_id:
                delete_key = f"{conversation_id}:{turn_id}"
                if state.get("delete_message_confirmation") == delete_key:
                    st.caption("Delete this message? Any confirmation action attached to it will be canceled.")
                    yes, no = st.columns(2)
                    if yes.button("Confirm delete", key=f"delete-confirm-{delete_key}"):
                        try:
                            api.delete_conversation_turn(conversation_id, turn_id)
                            state["conversation_messages"] = [
                                item for item in state.get("conversation_messages", [])
                                if (item.get("turn_id") or (item.get("id", "").split("-", 1)[-1] if item.get("history_only") else None)) != turn_id
                            ]
                            state.get("audio_cache", {}).pop(message.get("id"), None)
                            state.get("audio_playback_visible", {}).pop(message.get("id"), None)
                            state["delete_message_confirmation"] = None
                            st.rerun()
                        except APIError as exc:
                            state["delete_message_confirmation"] = None
                            st.warning("This message could not be deleted. Refresh the conversation and try again.")
                    if no.button("Keep message", key=f"delete-cancel-{delete_key}"):
                        state["delete_message_confirmation"] = None
                        st.rerun()
                elif st.button("Delete message", key=f"delete-{delete_key}"):
                    state["delete_message_confirmation"] = delete_key
                    st.rerun()
