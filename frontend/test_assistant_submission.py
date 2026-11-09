"""Offline assistant submission, recovery, and optional TTS regressions."""

import base64
import unittest
import inspect
from uuid import uuid4
from unittest.mock import MagicMock

from frontend.api_client import APIError
from frontend.pages.assistant import (
    _handle_voice_event,
    _recover_interrupted_turn,
    _submit_recorded_audio,
    _submit,
    render,
)
from frontend.session_state import reset_conversation_state


def response(*, speech=False):
    return {
        "conversation_id": "conversation-1",
        "turn_id": "user-turn-1",
        "assistant_turn_id": "assistant-turn-1",
        "input_mode": "speech",
        "output_mode": "speech_and_text" if speech else "text",
        "transcript": "Show me my last 5 transactions." if speech else None,
        "detected_languages": [],
        "assistant_text": "Here are your last 5 transactions.",
        "response_language": "en-IN",
        "speech_output": {"audio_url": "/api/v1/voice/responses/audio-1", "media_type": "audio/wav"} if speech else None,
        "tool_results": [],
    }


class AssistantSubmissionTests(unittest.TestCase):
    def setUp(self):
        self.state = {
            "conversation_id": "conversation-1",
            "conversation_messages": [],
            "response_mode": "text",
            "response_language": "en-IN",
            "language_hint": "auto",
            "pending_attachments": [],
            "audio_cache": {},
            "audio_playback_visible": {},
            "audio_widget_epoch": 0,
            "last_audio_hash": None,
        }
        self.st = MagicMock()
        self.api = MagicMock()

    def test_successful_assistant_response_is_saved_once(self):
        self.api.send_assistant_turn.return_value = response()

        _submit(self.api, self.st, self.state, text="Show me my last 5 transactions.")

        self.assertEqual(len(self.state["conversation_messages"]), 2)
        self.assertEqual(self.state["conversation_messages"][0]["text"], "Show me my last 5 transactions.")
        self.assertEqual(self.state["conversation_messages"][1]["text"], "Here are your last 5 transactions.")
        self.assertIsNone(self.state["assistant_failed_turn"])

    def test_enter_submits_but_shift_enter_inserts_a_newline(self):
        rendered_source = inspect.getsource(render)
        self.assertIn("if (e.key === 'Enter' && e.shiftKey) return;", rendered_source)
        self.assertIn("if (e.key === 'Enter' && !e.shiftKey)", rendered_source)
        self.assertIn("e.preventDefault();", rendered_source)
        self.assertIn("sendBtn.click();", rendered_source)

    def test_empty_and_whitespace_messages_are_not_submitted(self):
        for text in (None, "", "   \t\n"):
            with self.subTest(text=text):
                self.assertFalse(_submit(self.api, self.st, self.state, text=text))
        self.api.send_assistant_turn.assert_not_called()
        self.assertNotIn("assistant_retry_payload", self.state)

    def test_duplicate_submission_event_is_ignored_while_request_is_running(self):
        def send(payload):
            duplicate = _submit(
                self.api, self.st, self.state, text="Show me my last 5 transactions."
            )
            self.assertFalse(duplicate)
            return response()

        self.api.send_assistant_turn.side_effect = send
        _submit(self.api, self.st, self.state, text="Show me my last 5 transactions.")
        self.api.send_assistant_turn.assert_called_once()
        self.assertFalse(self.state["assistant_submission_in_progress"])

    def test_processing_indicator_closes_and_submission_unlocks_after_failure(self):
        self.api.send_assistant_turn.side_effect = APIError(
            503, "The assistant provider is temporarily unavailable."
        )
        with self.assertRaises(APIError):
            _submit(self.api, self.st, self.state, text="Show me my last 5 transactions.")
        self.st.spinner.assert_called_once()
        self.assertFalse(self.state["assistant_submission_in_progress"])
        self.assertEqual(self.state["assistant_failed_turn"]["status"], "failed")

    def test_response_modes_preserve_language_and_only_speak_when_requested(self):
        for mode, should_fetch_audio in (
            ("text", False), ("speech", True), ("speech_and_text", True),
        ):
            with self.subTest(mode=mode):
                self.state.update({
                    "response_mode": mode,
                    "response_language": "kn-IN",
                    "language_hint": "kn-IN",
                })
                self.state["conversation_messages"] = []
                self.api.reset_mock()
                self.api.send_assistant_turn.return_value = response(speech=True)
                self.api.fetch_audio.return_value = (b"audio", "audio/wav")

                _submit(self.api, self.st, self.state, text="ನನ್ನ ಖಾತೆ ತೋರಿಸಿ")

                payload = self.api.send_assistant_turn.call_args.args[0]
                self.assertEqual(payload["output_mode"], mode)
                self.assertEqual(payload["language_hint"], "kn-IN")
                self.assertEqual(payload["response_language"], "kn-IN")
                assistant_message = self.state["conversation_messages"][-1]
                if should_fetch_audio:
                    self.api.fetch_audio.assert_called_once()
                    self.assertTrue(assistant_message["speech_audio_id"])
                    self.assertEqual(
                        self.state["autoplay_audio_id"], assistant_message["id"]
                    )
                else:
                    self.api.fetch_audio.assert_not_called()
                    self.assertIsNone(assistant_message["speech_audio_id"])
                    self.assertIsNone(self.state.get("autoplay_audio_id"))
                if mode == "speech":
                    self.assertEqual(assistant_message["text"], "")
                else:
                    self.assertEqual(assistant_message["text"], "Here are your last 5 transactions.")

    def test_lost_success_response_is_retried_with_same_idempotency_key(self):
        calls = []
        backend_side_effects = set()

        def send(payload):
            calls.append(dict(payload))
            backend_side_effects.add(payload["client_turn_id"])
            if len(calls) == 1:
                # The server completed the turn, but the client lost its response.
                raise APIError(0, "The VaaniShield API did not respond before the request timed out.")
            return response()

        self.api.send_assistant_turn.side_effect = send
        _submit(self.api, self.st, self.state, text="Show me my last 5 transactions.")

        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0], calls[1])
        self.assertEqual(len(backend_side_effects), 1)
        self.assertEqual(len(self.state["conversation_messages"]), 2)
        self.assertIsNone(self.state["assistant_failed_turn"])

    def test_ambiguous_transport_failure_is_not_reported_as_turn_failure(self):
        self.api.send_assistant_turn.side_effect = APIError(0, "The VaaniShield API did not respond before the request timed out.")

        with self.assertRaises(APIError):
            _submit(self.api, self.st, self.state, text="Show me my last 5 transactions.")

        self.assertEqual(self.state["assistant_failed_turn"]["status"], "processing")
        self.assertIsNone(self.state["assistant_failed_turn"]["error"])
        self.assertTrue(any("Show me my last 5 transactions." in item["text"] for item in self.state["conversation_messages"]))

    def test_streamlit_rerun_recovers_pending_turn_in_same_conversation(self):
        self.state.update({
            "assistant_retry_payload": {
                "conversation_id": "conversation-1",
                "client_turn_id": "same-client-turn",
                "input_mode": "speech",
                "output_mode": "speech_and_text",
                "audio_upload_id": "saved-upload",
                "language_hint": "auto",
                "response_language": "en-IN",
            },
            "assistant_failed_turn": {
                "client_turn_id": "same-client-turn",
                "text": "Voice message (transcript will be restored from the saved turn)",
                "is_voice": True,
                "status": "processing",
                "recovery_attempted": False,
            },
        })
        self.api.send_assistant_turn.return_value = response(speech=True)

        recovered = _recover_interrupted_turn(self.api, self.st, self.state)

        self.assertTrue(recovered)
        sent_payload = self.api.send_assistant_turn.call_args.args[0]
        self.assertEqual(sent_payload["conversation_id"], "conversation-1")
        self.assertEqual(sent_payload["client_turn_id"], "same-client-turn")
        self.assertEqual(self.state["conversation_messages"][0]["text"], "Show me my last 5 transactions.")
        self.assertIsNone(self.state["assistant_failed_turn"])

    def test_tts_retrieval_failure_keeps_successful_text_response(self):
        self.api.send_assistant_turn.return_value = response(speech=True)
        self.api.fetch_audio.side_effect = RuntimeError("audio transport dropped")
        self.state["response_mode"] = "speech"

        _submit(self.api, self.st, self.state, text="Show me my last 5 transactions.")

        self.assertEqual(self.state["conversation_messages"][-1]["text"], "Here are your last 5 transactions.")
        self.assertIsNone(self.state["assistant_failed_turn"])
        self.st.warning.assert_any_call("Your text response is ready, but its audio could not be loaded.")

    def test_explicit_text_response_mode_is_respected_for_voice_input(self):
        self.state.update({"response_mode": "text", "response_mode_explicit": True, "wake_word_enabled": True})
        self.api.upload_audio.return_value = {"audio_upload_id": "upload-1"}
        self.api.send_assistant_turn.return_value = response()
        audio = MagicMock()
        audio.getvalue.return_value = b"wav-audio"
        audio.name = "question.wav"
        audio.type = "audio/wav"

        _submit(self.api, self.st, self.state, audio=audio)

        payload = self.api.send_assistant_turn.call_args.args[0]
        self.assertEqual(payload["input_mode"], "speech")
        self.assertEqual(payload["output_mode"], "text")

    def test_wake_word_enabled_state_does_not_change_typed_text_response_mode(self):
        self.state.update({"response_mode": "text", "wake_word_enabled": True})
        self.api.send_assistant_turn.return_value = response(speech=True)

        _submit(self.api, self.st, self.state, text="Show me my balance.")

        payload = self.api.send_assistant_turn.call_args.args[0]
        self.assertEqual(payload["output_mode"], "text")
        self.api.fetch_audio.assert_not_called()

    def test_retry_uses_the_original_output_mode_after_selector_changes(self):
        self.state.update({
            "response_mode": "speech_and_text",
            "assistant_retry_payload": {
                "conversation_id": "conversation-1",
                "client_turn_id": "same-turn",
                "input_mode": "text",
                "output_mode": "speech",
                "text": "Show me my balance.",
                "language_hint": "auto",
                "response_language": "en-IN",
            },
            "assistant_failed_turn": {
                "client_turn_id": "same-turn", "text": "Show me my balance.",
                "status": "failed", "error": "temporary failure",
            },
        })
        self.api.send_assistant_turn.return_value = response(speech=True)
        self.api.fetch_audio.return_value = (b"audio", "audio/wav")

        _submit(self.api, self.st, self.state, retry=True)

        self.assertEqual(
            self.api.send_assistant_turn.call_args.args[0]["output_mode"], "speech"
        )
        self.assertEqual(self.state["conversation_messages"][-1]["text"], "")
        self.api.fetch_audio.assert_called_once()

    def test_retry_does_not_duplicate_existing_assistant_message(self):
        saved = response()
        self.state.update({
            "assistant_retry_payload": {
                "conversation_id": "conversation-1",
                "client_turn_id": "same-client-turn",
                "input_mode": "text",
                "output_mode": "text",
                "text": "Show me my last 5 transactions.",
                "language_hint": "auto",
                "response_language": "en-IN",
            },
            "assistant_failed_turn": {
                "client_turn_id": "same-client-turn", "text": "Show me my last 5 transactions.",
                "is_voice": False, "status": "failed", "error": "temporary failure",
            },
            "conversation_messages": [
                {"id": "user-user-turn-1", "role": "user", "text": "Show me my last 5 transactions."},
                {"id": "assistant-user-turn-1", "role": "assistant", "text": saved["assistant_text"]},
            ],
        })
        self.api.send_assistant_turn.return_value = saved

        _submit(self.api, self.st, self.state, retry=True)

        self.assertEqual(len(self.state["conversation_messages"]), 2)
        self.assertEqual(sum(item["role"] == "assistant" for item in self.state["conversation_messages"]), 1)

    def test_genuine_backend_failure_preserves_message_and_useful_error(self):
        self.api.send_assistant_turn.side_effect = APIError(503, "The assistant provider is temporarily unavailable.")

        with self.assertRaises(APIError):
            _submit(self.api, self.st, self.state, text="Show me my last 5 transactions.")

        self.assertEqual(self.state["assistant_failed_turn"]["status"], "failed")
        self.assertEqual(self.state["assistant_failed_turn"]["error"], "The assistant provider is temporarily unavailable.")
        self.assertTrue(any("Show me my last 5 transactions." in item["text"] for item in self.state["conversation_messages"]))

    def test_failed_submission_unlocks_and_can_be_retried(self):
        self.api.send_assistant_turn.side_effect = [
            APIError(503, "The assistant provider is temporarily unavailable."),
            response(),
        ]

        with self.assertRaises(APIError):
            _submit(self.api, self.st, self.state, text="Show me my last 5 transactions.")
        original_payload = self.state["assistant_retry_payload"]
        self.assertFalse(self.state["assistant_submission_in_progress"])
        _submit(self.api, self.st, self.state, retry=True)

        self.assertEqual(self.api.send_assistant_turn.call_count, 2)
        self.assertEqual(self.api.send_assistant_turn.call_args.args[0], original_payload)
        self.assertIsNone(self.state["assistant_failed_turn"])
        self.assertIsNone(self.state["assistant_retry_payload"])
        self.assertFalse(self.state["assistant_submission_in_progress"])

    def test_same_audio_widget_delivery_is_submitted_once(self):
        class Audio:
            name = "voice.wav"
            type = "audio/wav"

            def getvalue(self):
                return b"recorded audio"

        self.api.upload_audio.return_value = {"audio_upload_id": "upload-1"}
        self.api.send_assistant_turn.return_value = response(speech=False)
        audio = Audio()

        self.assertTrue(_submit_recorded_audio(
            self.api, self.st, self.state, audio, widget_epoch=0,
        ))
        self.assertFalse(_submit_recorded_audio(
            self.api, self.st, self.state, audio, widget_epoch=0,
        ))

        self.api.upload_audio.assert_called_once()
        self.api.send_assistant_turn.assert_called_once()

    def test_new_audio_widget_epoch_accepts_same_recording_bytes(self):
        class Audio:
            name = "voice.wav"
            type = "audio/wav"

            def getvalue(self):
                return b"same content"

        self.api.upload_audio.return_value = {"audio_upload_id": "upload-1"}
        self.api.send_assistant_turn.return_value = response(speech=False)
        audio = Audio()
        _submit_recorded_audio(self.api, self.st, self.state, audio, widget_epoch=0)
        # A distinct recording may legitimately contain identical bytes.
        self.assertTrue(_submit_recorded_audio(
            self.api, self.st, self.state, audio, widget_epoch=1,
        ))
        self.assertEqual(self.api.upload_audio.call_count, 2)

    def test_voice_event_diagnostics_reject_duplicates_even_after_conversation_reset_and_allow_new_event(self):
        self.api.upload_audio.return_value = {"audio_upload_id": "upload-1"}
        self.api.send_assistant_turn.return_value = response(speech=True)
        self.api.fetch_audio.return_value = (b"RIFF\x00\x00\x00\x00WAVEfmt ", "audio/wav")
        wav_b64 = base64.b64encode(b"RIFF\x04\x00\x00\x00WAVEfmt ").decode("ascii")
        first_id = str(uuid4())
        second_id = str(uuid4())

        self.assertTrue(_handle_voice_event(
            self.api, self.st, self.state,
            {"id": first_id, "kind": "audio", "audio": wav_b64},
        ))
        # Duplicate delivery of the same component event is rejected and logged.
        self.assertFalse(_handle_voice_event(
            self.api, self.st, self.state,
            {"id": first_id, "kind": "audio", "audio": wav_b64},
        ))
        # Resetting conversation state must keep processed voice IDs so a mounted
        # component's last value is not re-submitted as a new turn.
        reset_conversation_state(self.state)
        self.state["conversation_id"] = "conversation-2"
        self.assertFalse(_handle_voice_event(
            self.api, self.st, self.state,
            {"id": first_id, "kind": "audio", "audio": wav_b64},
        ))
        # A later separate recording with a new event_id is accepted cleanly.
        self.assertTrue(_handle_voice_event(
            self.api, self.st, self.state,
            {"id": second_id, "kind": "audio", "audio": wav_b64},
        ))

        self.assertEqual(self.api.upload_audio.call_count, 2)
        self.assertEqual(self.api.send_assistant_turn.call_count, 2)
        first_stages = [
            d["stage"] for d in self.state["voice_diagnostics"]
            if d["event_id"] == first_id
        ]
        self.assertEqual(first_stages, [
            "component_event_received",
            "upload_started",
            "upload_completed",
            "assistant_turn_started",
            "assistant_turn_completed",
            "response_playback_queued",
            "duplicate_event_rejected",
            "duplicate_event_rejected",
        ])
        second_stages = [
            d["stage"] for d in self.state["voice_diagnostics"]
            if d["event_id"] == second_id
        ]
        self.assertEqual(second_stages, [
            "component_event_received",
            "upload_started",
            "upload_completed",
            "assistant_turn_started",
            "assistant_turn_completed",
            "response_playback_queued",
        ])
        for entry in self.state["voice_diagnostics"]:
            self.assertEqual(set(entry.keys()) - {"stage", "event_id", "kind"}, set())


if __name__ == "__main__":
    unittest.main()
