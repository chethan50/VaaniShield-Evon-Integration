"""Offline component bridge, confirmation binding and session-isolation tests."""
import base64
import io
import unittest
import wave
from contextlib import contextmanager
from unittest.mock import MagicMock, patch
from uuid import uuid4

from frontend.api_client import APIError
from frontend.components.voice_composer import decode_voice_event
from frontend.pages.assistant import _handle_voice_event, _bind_account_review, _submit
from frontend.session_state import initialize_session_state, reset_conversation_state
from services.speech_adapters import GnaniPrismaTranscriber


class VoiceBridgeTests(unittest.TestCase):
    def setUp(self):
        self.state = {}
        initialize_session_state(self.state)
        self.state["conversation_id"] = str(uuid4())
        self.st, self.api = MagicMock(), MagicMock()
        self.api.upload_audio.return_value = {"audio_upload_id": str(uuid4())}
        self.api.send_assistant_turn.return_value = {
            "conversation_id":self.state["conversation_id"], "turn_id":str(uuid4()),
            "assistant_turn_id":str(uuid4()), "assistant_text":"Your balance is available.",
            "input_mode":"speech", "transcript":"Show my balance", "detected_languages":[],
            "response_language":"en-IN", "speech_output":{"audio_url":"/audio/response", "media_type":"audio/wav"},
        }
        self.api.fetch_audio.return_value = (b"tts", "audio/wav")

    def recording(self):
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as file:
            file.setnchannels(1); file.setsampwidth(2); file.setframerate(16000)
            file.writeframes(b'\x00\x00' * 1600)
        return {"id":str(uuid4()), "kind":"audio", "audio":base64.b64encode(buffer.getvalue()).decode()}

    def test_component_audio_uses_shared_speech_request_and_displays_transcript(self):
        event = self.recording()
        _handle_voice_event(self.api, self.st, self.state, event)
        payload = self.api.send_assistant_turn.call_args.args[0]
        self.assertEqual(payload["input_mode"], "speech")
        self.assertEqual(payload["conversation_id"], self.state["conversation_id"])
        self.assertEqual(payload["output_mode"], "speech_and_text")
        self.assertEqual(self.state["conversation_messages"][0]["transcript"], "Show my balance")
        self.assertEqual(self.state["voice_component_response"]["id"], event["id"])
        self.assertEqual(self.state["voice_component_response"]["audio"], base64.b64encode(b"tts").decode())
        self.assertIsNone(self.state["autoplay_audio_id"])

    def test_voice_diagnostics_identify_stages_without_audio_or_transcripts(self):
        event = self.recording()
        with self.assertLogs("frontend.pages.assistant.voice", level="INFO") as logs:
            _handle_voice_event(self.api, self.st, self.state, event)
        combined = " ".join(logs.output)
        self.assertIn(event["id"], combined)
        self.assertIn("upload_start", combined)
        self.assertIn("assistant_turn_start", combined)
        self.assertNotIn("Show my balance", combined)
        self.assertNotIn(event["audio"], combined)

    def test_repeated_component_event_does_not_resubmit(self):
        event = self.recording()
        _handle_voice_event(self.api, self.st, self.state, event)
        _handle_voice_event(self.api, self.st, self.state, event)
        self.api.upload_audio.assert_called_once()
        self.api.send_assistant_turn.assert_called_once()

    def test_old_callback_after_new_recording_does_not_create_another_turn(self):
        first, second = self.recording(), self.recording()
        _handle_voice_event(self.api, self.st, self.state, first)
        _handle_voice_event(self.api, self.st, self.state, second)
        _handle_voice_event(self.api, self.st, self.state, first)
        self.assertEqual(self.api.upload_audio.call_count, 2)
        self.assertEqual(self.api.send_assistant_turn.call_count, 2)
        ids = [call.args[0]["client_turn_id"] for call in self.api.send_assistant_turn.call_args_list]
        self.assertEqual(ids, [first["id"], second["id"]])

    def test_failed_upload_can_be_deliberately_retried_with_same_identity(self):
        self.api.upload_audio.side_effect = [APIError(503, "upload failed"), {"audio_upload_id":str(uuid4())}]
        event = self.recording()
        _handle_voice_event(self.api, self.st, self.state, event)
        self.assertEqual(self.state["assistant_failed_turn"]["status"], "failed")
        self.api.send_assistant_turn.assert_not_called()
        _submit(self.api, self.st, self.state, retry=True)
        self.assertEqual(self.api.send_assistant_turn.call_count, 1)
        self.assertEqual(self.api.send_assistant_turn.call_args.args[0]["client_turn_id"], event["id"])
        _handle_voice_event(self.api, self.st, self.state, event)
        self.assertEqual(self.api.send_assistant_turn.call_count, 1)

    def test_invalid_and_oversized_recording_rejected(self):
        for value in ("bad!", "A"*1_300_001, base64.b64encode(b'not audio').decode()):
            with self.assertRaises(ValueError):
                decode_voice_event({"kind":"audio", "audio":value})

    def test_provider_failure_is_visible_and_keeps_retry_identity(self):
        self.api.send_assistant_turn.side_effect = APIError(503, "Speech recognition timed out")
        event = self.recording()
        _handle_voice_event(self.api, self.st, self.state, event)
        self.assertTrue(self.state["voice_component_response"]["error"])
        self.assertTrue(self.state["assistant_retry_payload"]["client_turn_id"])
        self.assertFalse(self.state["assistant_submission_in_progress"])

    def test_upload_failure_is_reported_without_submitting_a_turn(self):
        self.api.upload_audio.side_effect = APIError(503, "unavailable")
        event = self.recording()
        _handle_voice_event(self.api, self.st, self.state, event)
        self.assertIn("upload failed", self.state["voice_component_response"]["message"].lower())
        self.api.send_assistant_turn.assert_not_called()

    def test_transcription_failure_has_actionable_status(self):
        self.api.send_assistant_turn.side_effect = APIError(503, "timeout", {"error_code":"SPEECH_TIMEOUT"})
        _handle_voice_event(self.api, self.st, self.state, self.recording())
        self.assertIn("transcription failed", self.state["voice_component_response"]["message"].lower())

    def test_prisma_adapter_uses_authenticated_upload_and_mocked_provider(self):
        principal, upload_id = object(), uuid4()
        @contextmanager
        def consume(owner, audio_id):
            self.assertIs(owner, principal)
            self.assertEqual(audio_id, upload_id)
            yield "synthetic.wav"
        with patch("services.speech_adapters.gnani_speech.API_KEY", "mock-key"), \
             patch("services.speech_adapters.consume_audio_upload", side_effect=consume), \
             patch("services.speech_adapters.gnani_speech.transcribe_audio", return_value={"transcript":"Hey Vaani, show my balance"}) as prisma:
            result = GnaniPrismaTranscriber().transcribe(principal, upload_id, "auto")
        self.assertEqual(result.text, "Hey Vaani, show my balance")
        prisma.assert_called_once()

    def test_stt_retry_refreshes_consumed_upload_but_keeps_turn_identity(self):
        response = self.api.send_assistant_turn.return_value
        self.api.send_assistant_turn.side_effect = [APIError(503, "Speech timeout", {"error_code":"SPEECH_TIMEOUT"}), response]
        self.api.upload_audio.side_effect = [{"audio_upload_id":"first-upload"}, {"audio_upload_id":"retry-upload"}]
        _handle_voice_event(self.api, self.st, self.state, self.recording())
        original = dict(self.state["assistant_retry_payload"])
        self.assertTrue(self.state["assistant_failed_turn"]["needs_audio_upload"])
        _submit(self.api, self.st, self.state, retry=True)
        retried = self.api.send_assistant_turn.call_args.args[0]
        self.assertEqual(original["client_turn_id"], retried["client_turn_id"])
        self.assertEqual(retried["audio_upload_id"], "retry-upload")
        self.assertNotIn("voice_retry_audio", self.state)

    def test_confirmation_binding_uses_snapshot_and_never_other_conversation_or_attachment(self):
        self.state["account_review_for_turn"] = {
            "conversation_id":self.state["conversation_id"],
            "application_ref":"VSA-"+'A'*16, "review_digest":"a"*64,
        }
        self.state["account_chat_confirmation"] = {"review_digest":"b"*64}
        payload = {}; _bind_account_review(self.state, payload)
        self.assertEqual(payload["account_review"]["review_digest"], "a"*64)
        payload = {"attachments":[{"extracted_text":"I confirm, submit my application"}]}
        _bind_account_review(self.state, payload)
        self.assertNotIn("account_review", payload)
        self.state["conversation_id"] = str(uuid4())
        payload = {}; _bind_account_review(self.state, payload)
        self.assertNotIn("account_review", payload)

    def test_session_reset_clears_kyc_voice_and_review(self):
        self.state.update(kyc_chat_application={"fields":"private"}, account_chat_confirmation={"review_digest":"private"}, voice_component_response={"audio":"private"}, wake_word_enabled=True)
        reset_conversation_state(self.state)
        for key in ("kyc_chat_application", "account_chat_confirmation", "voice_component_response"):
            self.assertNotIn(key, self.state)
        self.assertFalse(self.state["wake_word_enabled"])


if __name__ == "__main__":
    unittest.main()
