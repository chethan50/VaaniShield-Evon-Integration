"""Offline account-opening consent, checklist and review regressions."""
import unittest
from unittest.mock import MagicMock, Mock, patch

from backend.test_account_opening import AccountOpeningTests, make_turn_request, VALID_PDF, VALID_PNG
from services import account_application_service as apps
from services.account_application_assistant import run_account_application_turn
from services.agent_tools import execute_tool
from services.assistant_service import AssistantOrchestrator
from services.llm_adapters import MockLLMProvider
from services.llm_interfaces import LLMResponse


class AccountCorrectionTests(unittest.TestCase):
    setUp = AccountOpeningTests.setUp
    tearDown = AccountOpeningTests.tearDown

    def agent(self, answers=()):
        return AssistantOrchestrator(llm_provider=MockLLMProvider(list(answers)), transcriber=Mock(), synthesizer=Mock())

    def turn(self, agent, text, conversation=None):
        return agent.process_turn(self.arjun, make_turn_request(text, conversation_id=conversation, response_language="en-IN"))

    def complete(self, ref):
        AccountOpeningTests._complete_arjun_details(self, ref)
        AccountOpeningTests._upload_required_docs(self, ref)

    def test_new_draft_and_existing_different_type_requires_consent(self):
        agent = self.agent()
        first = self.turn(agent, "Open a savings account")
        ref = apps.get_current_application(self.arjun)["application_ref"]
        self.assertEqual(first.tool_results[0].tool_name, "start_account_application")
        before = apps.get_application(self.arjun, ref)
        prompt = self.turn(agent, "Open a salary account", first.conversation_id)
        self.assertIn("Would you like to resume", prompt.assistant_text)
        self.assertIn("Savings", prompt.assistant_text)
        self.assertEqual(apps.get_application(self.arjun, ref), before)
        waiting = self.turn(agent, "salary account", first.conversation_id)
        self.assertIn("say yes", waiting.assistant_text)
        self.assertEqual(apps.get_application(self.arjun, ref), before)
        accepted = self.turn(agent, "yes", first.conversation_id)
        self.assertIn("Resuming", accepted.assistant_text)
        self.assertEqual(apps.get_application(self.arjun, ref), before)
        change = self.turn(agent, "I want a salary account", first.conversation_id)
        self.assertIn("remains unchanged", change.assistant_text)
        self.assertEqual(apps.get_application(self.arjun, ref), before)
        self.assertEqual(apps.start_application(self.arjun, "SALARY")["account_type"], "SAVINGS")
        self.assertEqual(apps.get_application(self.arjun, ref), before)
        self.assertEqual(len(apps.list_applications(self.arjun)), 1)

    def test_decline_preserves_pending_review_and_customer_isolation(self):
        agent = self.agent()
        first = self.turn(agent, "Open a savings account")
        ref = apps.get_current_application(self.arjun)["application_ref"]
        self.complete(ref)
        digest = apps.prepare_submission(self.arjun, ref)["review_digest"]
        before = apps.get_application(self.arjun, ref)
        prompt = self.turn(agent, "Open a current account", first.conversation_id)
        self.assertIn("awaiting final review", prompt.assistant_text)
        declined = self.turn(agent, "no", first.conversation_id)
        self.assertIn("unchanged", declined.assistant_text)
        self.assertEqual(apps.get_application(self.arjun, ref), before)
        self.assertEqual(before["review_digest"], digest)
        other = apps.start_application(self.priya, "SALARY")
        self.assertNotEqual(other["application_ref"], ref)
        with self.assertRaises(apps.ApplicationNotFound):
            apps.get_application(self.priya, ref)

    def test_pending_review_can_be_resumed_by_voice_without_changing_digest(self):
        from uuid import uuid4
        from services.speech_interfaces import Transcription
        agent = self.agent()
        first = self.turn(agent, "Open a savings account")
        ref = apps.get_current_application(self.arjun)["application_ref"]
        self.complete(ref)
        digest = apps.prepare_submission(self.arjun, ref)["review_digest"]
        self.turn(agent, "I want a new account", first.conversation_id)
        agent.transcriber.transcribe.return_value = Transcription(text="yes", languages=("en-IN",))
        spoken = agent.process_turn(self.arjun, make_turn_request(
            conversation_id=first.conversation_id, input_mode="speech", audio_upload_id=uuid4(), response_language="en-IN",
        ))
        self.assertIn("Resuming", spoken.assistant_text)
        current = apps.get_application(self.arjun, ref)
        self.assertEqual(current["status"], "PENDING_CONFIRMATION")
        self.assertEqual(current["review_digest"], digest)

    def test_checklist_past_tense_multilingual_missing_and_unverified(self):
        agent = self.agent()
        first = self.turn(agent, "Open a salary account")
        ref = apps.get_current_application(self.arjun)["application_ref"]
        apps.store_document(self.arjun, ref, "AADHAAR_DEMO", VALID_PDF, "application/pdf")
        for statement in ("I uploaded the documents.", "I have uploaded my PAN.", "I have my documents.", "मैंने दस्तावेज़ अपलोड कर दिए", "ನಾನು ದಾಖಲೆ ಅಪ್ಲೋಡ್ ಮಾಡಿದೆ"):
            with self.subTest(statement=statement):
                result = self.turn(agent, statement, first.conversation_id)
                self.assertEqual(result.tool_results[0].tool_name, "get_application_document_status")
                self.assertIn("not verified", result.assistant_text)
                self.assertIn("missing", result.assistant_text)
                self.assertNotIn("submitted successfully", result.assistant_text)
        apps.store_document(self.arjun, ref, "PAN_DEMO", VALID_PNG, "image/png")
        result = self.turn(agent, "I uploaded the documents", first.conversation_id)
        self.assertNotIn("missing", result.assistant_text)
        self.assertIn("not verified", result.assistant_text)
        self.assertNotEqual((result.tool_results[-1].data or {}).get("ui_navigation", {}).get("subfeature"), "Review & Submit")

    def test_contextual_upload_and_unavailable_checklist(self):
        apps.start_application(self.arjun, "SAVINGS")
        history = [{"role":"assistant", "assistant_response_text":"Please upload your KYC documents."}]
        result = run_account_application_turn(self.arjun, "I uploaded them", "en-IN", execute_tool, history=history)
        self.assertIn("missing", result[0])
        result = run_account_application_turn(self.arjun, "Are they ready?", "en-IN", execute_tool, history=history)
        self.assertIn("missing", result[0])
        with patch("services.agent_tools.account_applications.get_document_status", side_effect=RuntimeError("offline")):
            result = run_account_application_turn(self.arjun, "I uploaded my documents", "en-IN", execute_tool)
        self.assertIn("could not check", result[0])
        self.assertNotIn("all required documents have been received", result[0])

    def test_rejected_checklist_never_claims_ready(self):
        apps.start_application(self.arjun, "SAVINGS")
        def execute(name, principal, args):
            if name == "get_application_document_status":
                return {"account_type":"SAVINGS", "missing_documents":[], "documents":[
                    {"document_type":"PAN_DEMO", "label":"PAN", "required":True, "uploaded":True, "verification_status":"REJECTED"}]}
            return {"application_ref":"VSA-"+"A"*16, "validation":{"is_valid":True,"missing_documents":[]}, "documents":[]}
        text, results = run_account_application_turn(self.arjun, "I uploaded the documents", "en-IN", execute)
        self.assertIn("rejected", text)
        self.assertNotIn("ready for your review", text)
        self.assertNotEqual((results[-1].data or {}).get("ui_navigation", {}).get("subfeature"), "Review & Submit")

    def test_verification_states_are_reported_from_checklist(self):
        apps.start_application(self.arjun, "SAVINGS")
        for status, expected in (("VERIFIED", "uploaded and verified"),
                                 ("PENDING_VERIFICATION", "verification pending"),
                                 ("NOT_VERIFIED", "not verified"),
                                 ("REJECTED", "rejected")):
            with self.subTest(status=status):
                def execute(name, principal, args):
                    if name == "get_application_document_status":
                        return {"account_type":"SAVINGS", "missing_documents":[], "documents":[
                            {"document_type":"PAN_DEMO", "label":"PAN", "required":True, "uploaded":True, "verification_status":status}]}
                    return {"validation":{"is_valid":False}, "documents":[]}
                text, _ = run_account_application_turn(self.arjun, "I uploaded my PAN", "en-IN", execute)
                self.assertIn(expected, text)

    def test_fabricated_model_claim_is_blocked(self):
        agent = self.agent([LLMResponse(text="All required documents are verified and your application is submitted.")])
        result = self.turn(agent, "Tell me a story about a bank")
        self.assertIn("cannot confirm", result.assistant_text)
        self.assertNotIn("are verified", result.assistant_text)

    def test_unrecognized_application_followup_cannot_invent_implicit_success(self):
        agent = self.agent([LLMResponse(text="Everything is all set; you will hear from us soon.")])
        result = self.turn(agent, "I am wondering about my application")
        self.assertIn("cannot confirm", result.assistant_text)
        self.assertNotIn("all set", result.assistant_text)

    def test_review_navigation_does_not_submit_and_manual_submission_is_idempotent(self):
        agent = self.agent()
        first = self.turn(agent, "Open a savings account")
        ref = apps.get_current_application(self.arjun)["application_ref"]
        incomplete = self.turn(agent, "Review my application", first.conversation_id)
        self.assertFalse(any((r.data or {}).get("ui_navigation", {}).get("subfeature") == "Review & Submit" for r in incomplete.tool_results))
        self.complete(ref)
        ready = self.turn(agent, "Review my application", first.conversation_id)
        nav = ready.tool_results[-1].data["ui_navigation"]
        self.assertEqual(nav["subfeature"], "Review & Submit")
        self.assertEqual(nav["application_ref"], ref)
        self.assertEqual(apps.get_application(self.arjun, ref)["status"], "DRAFT")
        prepared = apps.prepare_submission(self.arjun, ref)
        self.turn(agent, "I confirm, submit my application", first.conversation_id)
        self.assertEqual(apps.get_application(self.arjun, ref)["status"], "PENDING_CONFIRMATION")
        result = apps.submit_application(self.arjun, ref, confirmed=True, confirmation_method="button", review_digest=prepared["review_digest"])
        replay = apps.submit_application(self.arjun, ref, confirmed=True, confirmation_method="button", review_digest=prepared["review_digest"])
        self.assertEqual(result["application_ref"], ref)
        self.assertTrue(replay["already_submitted"])
        status = self.turn(agent, "What is my application status?", first.conversation_id)
        self.assertIn(ref, status.assistant_text)
        self.assertIn("submitted", status.assistant_text.lower())
        self.assertNotIn("account has been opened", status.assistant_text.lower())

    def test_review_page_does_not_switch_to_another_application(self):
        from frontend.pages.account_opening import render
        ref = apps.start_application(self.arjun, "SAVINGS")["application_ref"]
        api = Mock()
        api.get_current_account_application.return_value = {"application": apps.get_application(self.arjun, ref)}
        st = MagicMock()
        state = {"account_application_review_target_ref":"VSA-" + "F" * 16}
        with patch("frontend.pages.account_opening._render_review_and_submit") as review:
            render(st, "Review & Submit", api, state)
        review.assert_not_called()
        st.warning.assert_called_once()

    def test_chat_review_link_rechecks_readiness_and_only_navigates(self):
        from frontend.components.conversation import render_messages
        ref = apps.start_application(self.arjun, "SAVINGS")["application_ref"]
        message = {"id":"review-message", "role":"assistant", "text":"Review", "ui_navigation": {
            "target":"account_services", "subfeature":"Review & Submit", "step":"review", "application_ref":ref,
        }}
        api, st = Mock(), MagicMock()
        api.get_account_application.side_effect = lambda _: apps.get_application(self.arjun, ref)
        st.button.side_effect = lambda label, **kw: label == "Go to Review & Submit"
        state = {"audio_cache": {}, "audio_playback_visible": {}}
        render_messages(st, [message], state, api=api)
        self.assertFalse(any(call.args and call.args[0] == "Go to Review & Submit" for call in st.button.call_args_list))
        self.complete(ref)
        render_messages(st, [message], state, api=api)
        self.assertEqual(state["pending_navigation"]["subfeature"], "Review & Submit")
        self.assertEqual(state["account_application_review_target_ref"], ref)
        api.submit_account_application.assert_not_called()
        self.assertEqual(apps.get_application(self.arjun, ref)["status"], "DRAFT")


if __name__ == "__main__":
    unittest.main()
