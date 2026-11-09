"""Offline Phase 3 flows using a temporary DB, mocked speech, and existing KYC APIs."""
import io
import unittest
from unittest.mock import Mock, patch
from uuid import uuid4

from backend.test_account_opening import AccountOpeningTests, make_turn_request, VALID_PDF
from frontend.components.account_application_chat import upload_kyc, render_application_review
from services import account_application_service as apps
from services.assistant_service import AssistantOrchestrator
from services.llm_adapters import MockLLMProvider
from services.speech_interfaces import Transcription


class UnifiedAgentTests(unittest.TestCase):
    setUp = AccountOpeningTests.setUp
    tearDown = AccountOpeningTests.tearDown

    def agent(self):
        llm = MockLLMProvider([])
        agent = AssistantOrchestrator(llm_provider=llm, transcriber=Mock(), synthesizer=Mock())
        return agent, llm

    def turn(self, agent, text, conversation=None, voice=False):
        if voice:
            agent.transcriber.transcribe.return_value = Transcription(text=text, languages=("en-IN",))
        return agent.process_turn(self.arjun, make_turn_request(
            text=None if voice else text, conversation_id=conversation,
            input_mode="speech" if voice else "text", audio_upload_id=uuid4() if voice else None,
            response_language="en-IN",
        ))

    def test_conversational_fields_resume_and_text_voice_share_state(self):
        agent, llm = self.agent()
        first = self.turn(agent, "Help me open a savings account")
        cid = first.conversation_id
        self.assertIn("date of birth", first.assistant_text)
        app = apps.get_current_application(self.arjun)
        self.assertEqual(app["fields"]["full_name"], "Arjun Sharma")
        second = self.turn(agent, "14 May 1996", cid, voice=True)
        self.assertEqual(second.transcript, "14 May 1996")
        self.assertIn("address", second.assistant_text)
        self.turn(agent, "42 MG Road, Indiranagar", cid)
        self.turn(agent, "Bengaluru", cid, voice=True)
        self.turn(agent, "Karnataka", cid)
        last = self.turn(agent, "560038", cid, voice=True)
        self.assertIn("upload", last.assistant_text.lower())
        updated = apps.get_current_application(self.arjun)
        self.assertEqual(updated["fields"]["date_of_birth"], "1996-05-14")
        self.assertEqual(updated["fields"]["city"], "Bengaluru")
        resumed = self.turn(agent, "continue my application")
        self.assertIn("would you like to resume", resumed.assistant_text.lower())
        resumed = self.turn(agent, "yes", resumed.conversation_id)
        self.assertIn("upload", resumed.assistant_text.lower())
        self.assertEqual(apps.get_current_application(self.arjun)["application_ref"], app["application_ref"])
        self.assertEqual(len(apps.list_applications(self.arjun)), 1)
        self.assertEqual(llm.requests, [])

    def test_all_account_types_and_missing_information_not_invented(self):
        agent, llm = self.agent()
        first = self.turn(agent, "Create a new bank account")
        for selected in ("salary", "current", "savings"):
            self.turn(agent, f"I want a {selected} account", first.conversation_id, voice=True)
            app = apps.get_current_application(self.arjun)
            self.assertEqual(app["account_type"], "SALARY")
            self.assertIsNone(app["fields"]["date_of_birth"])
            self.assertIsNone(app["fields"]["address_line"])
        self.assertEqual(llm.requests, [])

    def test_upload_chat_review_idempotency_and_tracker_single_source(self):
        agent, llm = self.agent()
        first = self.turn(agent, "Open a current account")
        cid = first.conversation_id
        self.turn(agent, "DOB: 1996-05-14; address: 42 MG Road; city: Bengaluru; state: Karnataka; PIN code: 560038", cid)
        self.turn(agent, "I am a business owner", cid)
        app = apps.get_current_application(self.arjun); ref = app["application_ref"]
        api = Mock()
        api.upload_account_application_document.side_effect = lambda ref, kind, data, media: apps.store_document(self.arjun, ref, kind, data, media)
        state = {}
        for kind in ("AADHAAR_DEMO", "PAN_DEMO", "ADDRESS_PROOF_DEMO"):
            file = io.BytesIO(VALID_PDF + kind.encode()); file.type = "application/pdf"
            app = upload_kyc(api, state, app, kind, file)
        self.assertTrue(all(doc["uploaded"] for doc in apps.get_document_status(self.arjun, ref)["documents"]))
        review = self.turn(agent, "Review my application", cid, voice=True)
        self.assertIn("ready for your review", review.assistant_text)
        self.assertEqual(review.tool_results[-1].data["ui_navigation"]["application_ref"], ref)
        self.assertEqual(apps.get_application(self.arjun, ref)["status"], "DRAFT")
        apps.prepare_submission(self.arjun, ref)
        prepared = apps.get_application(self.arjun, ref)
        self.assertEqual(prepared["status"], "PENDING_CONFIRMATION")
        for doc in prepared["documents"]:
            self.assertEqual(doc["verification_status"], "NOT_VERIFIED")
        before = list(apps.DOCUMENT_ROOT.iterdir())
        duplicate = apps.store_document(self.arjun, ref, "PAN_DEMO", VALID_PDF + b"PAN_DEMO", "application/pdf")
        self.assertEqual(duplicate["review_digest"], prepared["review_digest"])
        self.assertEqual(set(apps.DOCUMENT_ROOT.iterdir()), set(before))
        self.turn(agent, "Submit my application", cid, voice=True)
        self.assertEqual(apps.get_application(self.arjun, ref)["status"], "PENDING_CONFIRMATION")
        result = apps.submit_application(self.arjun, ref, confirmed=True, confirmation_method="button", review_digest=prepared["review_digest"])
        self.assertEqual(result["status"], "SUBMITTED")
        again = apps.submit_application(self.arjun, ref, confirmed=True, confirmation_method="button", review_digest=prepared["review_digest"])
        self.assertTrue(again["already_submitted"])
        self.assertEqual(apps.list_applications(self.arjun)[0]["status"], "SUBMITTED")
        with self.assertRaises(apps.ApplicationNotFound):
            apps.open_document(self.priya, ref, "PAN_DEMO")
        self.assertEqual(llm.requests, [])

    def test_invalid_fields_do_not_overwrite_or_leak_to_provider(self):
        agent, llm = self.agent()
        first = self.turn(agent, "Open a salary account")
        invalid = self.turn(agent, "DOB: 2099-01-01", first.conversation_id, voice=True)
        self.assertIn("future", invalid.assistant_text)
        self.assertIsNone(apps.get_current_application(self.arjun)["fields"]["date_of_birth"])
        self.assertEqual(llm.requests, [])

    def test_review_ui_only_navigates(self):
        from unittest.mock import MagicMock
        st, api = MagicMock(), Mock()
        app = {"application_ref":"VSA-"+'A'*16, "status":"DRAFT", "account_type":"SAVINGS", "validation":{"is_valid":True}, "documents":[
            {"document_type":"AADHAAR_DEMO", "uploaded":True}, {"document_type":"PAN_DEMO", "uploaded":True}]}
        api.get_account_application.return_value = app
        state = {}
        st.button.return_value = True
        render_application_review(st, api, state, app["application_ref"], "msg")
        api.submit_account_application.assert_not_called()
        self.assertEqual(state["pending_navigation"]["subfeature"], "Review & Submit")
        self.assertEqual(state["account_application_selected_ref"], app["application_ref"])

    def test_wake_greeting_has_no_llm_call(self):
        agent, llm = self.agent()
        result = self.turn(agent, "Hey Vaani")
        self.assertEqual(result.assistant_text, "Yes, how can I help you?")
        self.assertEqual(llm.requests, [])

    def test_voice_confirmation_does_not_submit(self):
        agent, llm = self.agent()
        first = self.turn(agent, "Open a savings account")
        ref = apps.get_current_application(self.arjun)["application_ref"]
        AccountOpeningTests._complete_arjun_details(self, ref)
        AccountOpeningTests._upload_required_docs(self, ref)
        prepared = apps.prepare_submission(self.arjun, ref)
        from backend.api_models import AccountReviewBinding
        agent.transcriber.transcribe.return_value = Transcription(text="I confirm, submit my application.", languages=("en-IN",))
        request = make_turn_request(conversation_id=first.conversation_id, input_mode="speech", audio_upload_id=uuid4())
        request.account_review = AccountReviewBinding(application_ref=ref, review_digest=prepared["review_digest"])
        result = agent.process_turn(self.arjun, request)
        self.assertIn("personally", result.assistant_text)
        self.assertEqual(apps.get_application(self.arjun, ref)["status"], "PENDING_CONFIRMATION")
        self.assertEqual(agent.process_turn(self.arjun, request).assistant_text, result.assistant_text)
        agent.transcriber.transcribe.assert_called_once()
        self.assertEqual(llm.requests, [])

    def test_conversational_submission_is_never_allowed(self):
        ref = apps.start_application(self.arjun, "SAVINGS")["application_ref"]
        for principal in (self.arjun, self.priya):
            with self.assertRaises(apps.ApplicationConflict):
                apps.submit_application_from_conversation(principal, ref, user_text="I confirm, submit my application", input_mode="speech", review_digest="a"*64)
        self.assertEqual(apps.get_application(self.arjun, ref)["status"], "DRAFT")


if __name__ == "__main__":
    unittest.main()
