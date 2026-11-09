"""Exercise chat KYC uploads through the authenticated application API."""

import io
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch
from uuid import uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.account_application_routes import router
from backend.auth import get_authenticated_principal
from frontend.api_client import APIError, VaaniShieldAPI
from frontend.components.account_application_chat import render_kyc_upload, render_application_review
from frontend.pages.account_opening import _render_kyc_documents
from services import account_application_service as applications, database
from services.principal import CustomerPrincipal


PDF = b"%PDF-1.7\n1 0 obj\n<<>>\nendobj\n%%EOF"


class KycChatUploadTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.db_patch = patch.object(database, "DB_PATH", root / "test.sqlite3")
        self.doc_patch = patch.object(applications, "DOCUMENT_ROOT", root / "documents")
        self.db_patch.start()
        self.doc_patch.start()
        self.addCleanup(self.temp_dir.cleanup)
        self.addCleanup(self.doc_patch.stop)
        self.addCleanup(self.db_patch.stop)
        database.initialize_database()
        conn = database.get_connection()
        try:
            conn.executemany(
                "INSERT INTO customers (customer_key, name, phone, email) VALUES (?, ?, ?, ?)",
                [("demo-arjun", "Arjun Demo", "9876543210", "arjun@example.test"),
                 ("demo-priya", "Priya Demo", "9123456789", "priya@example.test")],
            )
            conn.commit()
        finally:
            conn.close()
        now = datetime.now(timezone.utc)
        self.owner = CustomerPrincipal("demo-arjun", uuid4(), now - timedelta(minutes=1), now + timedelta(minutes=30))
        self.other = CustomerPrincipal("demo-priya", uuid4(), now - timedelta(minutes=1), now + timedelta(minutes=30))
        app = FastAPI()
        app.include_router(router)
        app.dependency_overrides[get_authenticated_principal] = lambda: self.owner
        self.app = app
        self.client = TestClient(app)
        self.addCleanup(self.client.close)
        self.api = VaaniShieldAPI(base_url="http://testserver", session=self.client)
        self.state = {"conversation_id": "existing-chat", "pending_navigation": {"keep": True}}

    @staticmethod
    def ui(*, load=False, save=False, file=None):
        st = MagicMock()
        st.button.side_effect = lambda *args, **kwargs: (
            load if kwargs.get("key") == "chat-load-kyc" else
            save if kwargs.get("key") == "chat-save-kyc" else False
        )
        st.selectbox.return_value = "AADHAAR_DEMO"
        st.file_uploader.return_value = file
        return st

    @staticmethod
    def pdf(content=PDF):
        file = io.BytesIO(content)
        file.type = "application/pdf"
        return file

    def test_wrapped_response_to_upload_and_existing_kyc_listing(self):
        application = applications.start_application(self.owner, "SAVINGS")
        ref = application["application_ref"]
        with patch.object(self.client, "request", wraps=self.client.request) as request:
            load_ui = self.ui(load=True)
            render_kyc_upload(load_ui, self.api, self.state)
            self.assertEqual(self.state["kyc_chat_application"]["application_ref"], ref)
            self.assertIn(f"Application {ref}", [call.args[0] for call in load_ui.caption.call_args_list])

            upload_ui = self.ui(save=True, file=self.pdf())
            render_kyc_upload(upload_ui, self.api, self.state)
            self.assertEqual(self.state["conversation_id"], "existing-chat")
            self.assertIn("not verified", self.state["kyc_chat_receipt"])
            calls = [(call.args[0], call.args[1]) for call in request.call_args_list]
            self.assertIn(("GET", "http://testserver/api/v1/me/account-applications/current"), calls)
            self.assertIn(("POST", f"http://testserver/api/v1/me/account-applications/{ref}/documents/AADHAAR_DEMO"), calls)

        current = self.api.get_current_account_application()["application"]
        listed = self.api.get_account_application_documents(ref)
        document = next(doc for doc in listed["documents"] if doc["document_type"] == "AADHAAR_DEMO")
        self.assertTrue(document["uploaded"])
        self.assertEqual(document["verification_status"], "NOT_VERIFIED")
        self.assertNotIn("AADHAAR_DEMO", current["validation"]["missing_documents"])
        self.assertEqual(applications.open_document(self.owner, ref, "AADHAAR_DEMO")[0], PDF)

        page_ui = self.ui()
        column1, column2 = MagicMock(), MagicMock()
        column1.button.return_value = column2.button.return_value = False
        page_ui.columns.return_value = (column1, column2)
        with patch("frontend.pages.account_opening._render_validation_checklist"):
            _render_kyc_documents(page_ui, self.api, self.state, current)
        markup = " ".join(str(call.args[0]) for call in page_ui.markdown.call_args_list if call.args)
        self.assertIn(document["label"], markup)
        self.assertIn("Uploaded</span>", markup)

        self.app.dependency_overrides[get_authenticated_principal] = lambda: self.other
        with self.assertRaises(APIError) as failure:
            self.api.get_account_application_documents(ref)
        self.assertEqual(failure.exception.status_code, 404)

    def test_missing_or_malformed_application_has_clear_message_and_no_stale_upload(self):
        for payload, message in (({"application": None}, "No active application"),
                                 ({}, "Could not read"),
                                 ({"application": {}}, "incomplete"),
                                 ({"application": "invalid"}, "incomplete")):
            with self.subTest(payload=payload):
                self.state["kyc_chat_application"] = {"application_ref": "stale"}
                st = self.ui(load=True, save=True, file=self.pdf())
                api = MagicMock()
                api.get_current_account_application.return_value = payload
                render_kyc_upload(st, api, self.state)
                api.upload_account_application_document.assert_not_called()
                self.assertNotIn("kyc_chat_application", self.state)
                displayed = [str(call.args[0]) for call in st.info.call_args_list + st.warning.call_args_list]
                self.assertTrue(any(message in text for text in displayed), displayed)

    def test_actual_empty_application_response_is_handled(self):
        st = self.ui(load=True)
        render_kyc_upload(st, self.api, self.state)
        st.info.assert_any_call("No active application found. Start an account application first.")
        self.assertNotIn("kyc_chat_application", self.state)

    def test_load_and_upload_errors_remain_in_chat(self):
        applications.start_application(self.owner, "SAVINGS")
        st = self.ui(load=True)
        api = MagicMock()
        api.get_current_account_application.side_effect = APIError(503, "Service unavailable")
        render_kyc_upload(st, api, self.state)
        st.warning.assert_called_once_with("Could not load the application: Service unavailable")

        render_kyc_upload(self.ui(load=True), self.api, self.state)
        st = self.ui(save=True, file=self.pdf(b"not a PDF"))
        render_kyc_upload(st, self.api, self.state)
        self.assertIn("Could not upload the document", st.error.call_args.args[0])
        self.assertFalse(next(doc for doc in self.api.get_account_application_documents(
            self.state["kyc_chat_application"]["application_ref"])["documents"]
            if doc["document_type"] == "AADHAAR_DEMO")["uploaded"])

    def test_bad_checklist_is_reported_without_upload(self):
        api = MagicMock()
        api.get_current_account_application.return_value = {"application": {
            "application_ref": "VSA-" + "A" * 16, "account_type": "SAVINGS", "documents": None,
        }}
        st = self.ui(load=True, save=True, file=self.pdf())
        render_kyc_upload(st, api, self.state)
        st.warning.assert_called_once_with("The document checklist is unavailable. Please load the application again.")
        api.upload_account_application_document.assert_not_called()

    def test_chat_upload_keeps_or_invalidates_review_as_backend_requires(self):
        ref = applications.start_application(self.owner, "SAVINGS")["application_ref"]
        applications.update_application(self.owner, ref, {
            "date_of_birth": "1996-05-14", "address_line": "42 Example Road",
            "city": "Bengaluru", "state": "Karnataka", "postal_code": "560038",
        })
        applications.store_document(self.owner, ref, "AADHAAR_DEMO", PDF, "application/pdf")
        applications.store_document(self.owner, ref, "PAN_DEMO", PDF + b"other", "application/pdf")
        digest = applications.prepare_submission(self.owner, ref)["review_digest"]
        render_kyc_upload(self.ui(load=True), self.api, self.state)
        render_kyc_upload(self.ui(save=True, file=self.pdf()), self.api, self.state)
        unchanged = self.api.get_account_application(ref)
        self.assertEqual(unchanged["status"], "PENDING_CONFIRMATION")
        self.assertEqual(unchanged["review_digest"], digest)

        render_kyc_upload(self.ui(save=True, file=self.pdf(PDF + b"replacement")), self.api, self.state)
        self.assertEqual(self.api.get_account_application(ref)["status"], "DRAFT")
        with self.assertRaises(APIError) as failure:
            self.api.submit_account_application(ref, confirmed=True, review_digest=digest)
        self.assertEqual(failure.exception.status_code, 409)

    def test_kyc_page_upload_confirmation_does_not_claim_verification(self):
        current = applications.start_application(self.owner, "SAVINGS")
        st = self.ui()
        st.button.side_effect = lambda *args, **kwargs: kwargs.get("key") == "kyc-upload-btn-AADHAAR_DEMO"
        st.file_uploader.side_effect = lambda *args, **kwargs: self.pdf() if kwargs.get("key", "").endswith("-AADHAAR_DEMO") else None
        left, right = MagicMock(), MagicMock()
        left.button.return_value = right.button.return_value = False
        st.columns.return_value = (left, right)
        with patch("frontend.pages.account_opening._render_validation_checklist"):
            _render_kyc_documents(st, self.api, self.state, current)
        self.assertIn("has not been verified", st.success.call_args.args[0])

    def test_ready_chat_application_link_navigates_without_submission(self):
        ref = applications.start_application(self.owner, "SAVINGS")["application_ref"]
        applications.update_application(self.owner, ref, {
            "date_of_birth":"1996-05-14", "address_line":"42 Example Road", "city":"Bengaluru", "state":"Karnataka", "postal_code":"560038",
        })
        applications.store_document(self.owner, ref, "AADHAAR_DEMO", PDF, "application/pdf")
        applications.store_document(self.owner, ref, "PAN_DEMO", PDF + b"pan", "application/pdf")
        st = MagicMock()
        st.button.return_value = True
        render_application_review(st, self.api, self.state, ref, "test")
        self.assertEqual(self.state["pending_navigation"]["subfeature"], "Review & Submit")
        self.assertEqual(self.state["account_application_review_target_ref"], ref)
        self.assertEqual(applications.get_application(self.owner, ref)["status"], "DRAFT")


if __name__ == "__main__":
    unittest.main()
