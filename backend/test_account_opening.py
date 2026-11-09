"""Comprehensive offline tests for the VaaniShield Account Opening Assistant."""

from __future__ import annotations

import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import Mock, patch
from uuid import uuid4

from fastapi import HTTPException

from backend import account_application_routes, auth, main
from backend.api_models import (
    AccountApplicationStartRequest,
    AccountApplicationSubmitRequest,
    AccountApplicationUpdateRequest,
    AssistantTurnRequest,
    SpeechOutput,
)
from frontend.api_client import VaaniShieldAPI
from frontend.navigation import FEATURE_SUBFEATURES, ensure_subfeature, set_subfeature
from frontend.session_state import (
    clear_account_application_cache,
    initialize_session_state,
    mark_session_valid,
    sign_out,
)
from services import account_application_service, database
from services.account_application_assistant import classify_account_intent
from services.account_application_service import (
    ApplicationConflict,
    ApplicationNotFound,
    ApplicationValidationError,
    DocumentRejected,
    MAX_DOCUMENT_BYTES,
    delete_document,
    get_application,
    get_application_status,
    get_current_application,
    open_document,
    prepare_submission,
    return_to_draft,
    start_application,
    store_document,
    submit_application,
    update_application,
    validate_application,
)
from services.agent_tools import execute_tool, get_available_tools
from services.assistant_service import (
    AssistantOrchestrator,
    _required_tools_for_question,
)
from services.llm_adapters import MockLLMProvider
from services.llm_interfaces import LLMResponse, LLMToolCall
from services.principal import CustomerPrincipal
from services.speech_interfaces import Transcription


VALID_PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
VALID_PDF = b"%PDF-1.7\n1 0 obj\n<<>>\nendobj\n%%EOF"
VALID_JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 64 + b"\xff\xd9"


def make_principal(customer_key: str = "demo-arjun") -> CustomerPrincipal:
    now = datetime.now(timezone.utc)
    return CustomerPrincipal(
        customer_key=customer_key,
        session_id=uuid4(),
        issued_at=now - timedelta(minutes=1),
        expires_at=now + timedelta(minutes=30),
    )


def make_turn_request(
    text: str | None = None,
    *,
    conversation_id=None,
    input_mode: str = "text",
    output_mode: str = "text",
    audio_upload_id=None,
    language_hint: str = "auto",
    response_language: str | None = None,
) -> AssistantTurnRequest:
    return AssistantTurnRequest(
        conversation_id=conversation_id,
        client_turn_id=uuid4(),
        input_mode=input_mode,
        output_mode=output_mode,
        text=text,
        audio_upload_id=audio_upload_id,
        language_hint=language_hint,
        response_language=response_language,
    )


class AccountOpeningTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        base = Path(self.temp_dir.name)
        self.db_patch = patch.object(database, "DB_PATH", base / "account_opening.sqlite3")
        self.doc_patch = patch.object(
            account_application_service, "DOCUMENT_ROOT", base / "kyc_documents"
        )
        self.env_patch = patch.dict(
            os.environ,
            {"VAANISHIELD_DEMO_SESSION_SECRET": "test-account-opening-secret-key-0123456789"},
        )
        self.db_patch.start()
        self.doc_patch.start()
        self.env_patch.start()
        database.initialize_database()
        conn = database.get_connection()
        try:
            conn.executemany(
                "INSERT INTO customers (customer_key, name, phone, email, monthly_income) VALUES (?, ?, ?, ?, ?)",
                [
                    ("demo-arjun", "Arjun Sharma", "9876543210", "arjun@example.com", 85000.0),
                    ("demo-priya", "Priya Nair", "9123456789", "priya@example.com", 72000.0),
                ],
            )
            conn.execute(
                "INSERT INTO accounts (customer_id, account_number, account_type, balance, status) "
                "VALUES (1, 'VS1000001234', 'SAVINGS', 142500.0, 'ACTIVE')"
            )
            conn.commit()
        finally:
            conn.close()
        self.arjun = make_principal("demo-arjun")
        self.priya = make_principal("demo-priya")

    def tearDown(self):
        self.env_patch.stop()
        self.doc_patch.stop()
        self.db_patch.stop()
        self.temp_dir.cleanup()

    def _complete_arjun_details(self, application_ref: str, account_type: str = "SAVINGS"):
        return update_application(
            self.arjun,
            application_ref,
            {
                "account_type": account_type,
                "date_of_birth": "1996-05-14",
                "phone": "9876543210",
                "address_line": "42 MG Road, Indiranagar",
                "city": "Bengaluru",
                "state": "Karnataka",
                "postal_code": "560038",
                "employment_category": "SALARIED" if account_type != "CURRENT" else "BUSINESS_OWNER",
                "nominee_name": "Meera Sharma",
                "nominee_relationship": "SPOUSE",
            },
        )

    def _upload_required_docs(self, application_ref: str, account_type: str = "SAVINGS"):
        store_document(self.arjun, application_ref, "AADHAAR_DEMO", VALID_PDF, "application/pdf")
        out = store_document(self.arjun, application_ref, "PAN_DEMO", VALID_PNG, "image/png")
        if account_type == "CURRENT":
            out = store_document(
                self.arjun, application_ref, "ADDRESS_PROOF_DEMO", VALID_JPEG, "image/jpeg"
            )
        return out

    # 1. Account-opening intent recognition
    def test_01_account_opening_intent_recognition_multilingual(self):
        en_intent = classify_account_intent(
            "Vani, I want to open another savings account. Can you help me complete the process?"
        )
        self.assertIsNotNone(en_intent)
        self.assertEqual(en_intent["action"], "OPEN")
        self.assertEqual(en_intent["args"], {"account_type": "SAVINGS"})

        hi_intent = classify_account_intent("मुझे नया बचत खाता खोलना है")
        self.assertIsNotNone(hi_intent)
        self.assertEqual(hi_intent["action"], "OPEN")
        self.assertEqual(hi_intent["args"], {"account_type": "SAVINGS"})

        kn_intent = classify_account_intent("ನಾನು ಹೊಸ ಉಳಿತಾಯ ಖಾತೆ ತೆರೆಯಲು ಬಯಸುತ್ತೇನೆ")
        self.assertIsNotNone(kn_intent)
        self.assertEqual(kn_intent["action"], "OPEN")
        self.assertEqual(kn_intent["args"], {"account_type": "SAVINGS"})

        options_intent = classify_account_intent("What types of accounts can I open?")
        self.assertIsNotNone(options_intent)
        self.assertEqual(options_intent["action"], "OPTIONS")

        self.assertEqual(
            _required_tools_for_question(
                "Vani, I want to open another savings account. Can you help me complete the process?"
            ),
            frozenset({"start_account_application"}),
        )

    # 2. Ordinary account-balance queries remain correctly routed
    def test_02_ordinary_account_balance_queries_remain_routed_to_balance(self):
        for query in (
            "What is my savings account balance?",
            "Check my account balance",
            "मेरे खाते का बैलेंस कितना है?",
            "ನನ್ನ ಖಾತೆಯ ಬ್ಯಾಲೆನ್ಸ್ ಎಷ್ಟು?",
        ):
            self.assertIsNone(classify_account_intent(query))
            self.assertEqual(
                _required_tools_for_question(query),
                frozenset({"get_account_balance"}),
            )

        provider = MockLLMProvider([
            LLMResponse(tool_calls=(LLMToolCall("get_account_balance", {}, "bal-call-1"),)),
            LLMResponse(text="Your savings account balance is ₹142,500.00."),
        ])
        orchestrator = AssistantOrchestrator(llm_provider=provider)
        response = orchestrator.process_turn(
            self.arjun, make_turn_request("What is my savings account balance?")
        )
        self.assertEqual(response.tool_results[0].tool_name, "get_account_balance")
        self.assertIn("142,500", response.assistant_text)

    # 3. Starting a new application
    def test_03_start_new_application_creates_draft_and_prefills_profile(self):
        app_view = start_application(self.arjun, "SAVINGS")
        self.assertRegex(app_view["application_ref"], r"^VSA-[A-F0-9]{16}$")
        self.assertEqual(app_view["status"], "DRAFT")
        self.assertEqual(app_view["account_type"], "SAVINGS")
        self.assertFalse(app_view["resumed"])
        self.assertEqual(app_view["fields"]["full_name"], "Arjun Sharma")
        self.assertEqual(app_view["fields"]["email"], "arjun@example.com")
        self.assertEqual(app_view["fields"]["phone"], "9876543210")

    # 4. Account-type selection
    def test_04_account_type_selection_updates_required_documents(self):
        app_view = start_application(self.arjun)
        ref = app_view["application_ref"]
        self.assertIsNone(app_view["account_type"])
        self.assertEqual(app_view["progress"]["current_step"], "account_type")

        savings = update_application(self.arjun, ref, {"account_type": "SAVINGS"})
        self.assertEqual(savings["account_type"], "SAVINGS")
        self.assertEqual(
            savings["validation"]["missing_documents"],
            ["AADHAAR_DEMO", "PAN_DEMO"],
        )

        current = update_application(self.arjun, ref, {"account_type": "CURRENT"})
        self.assertEqual(current["account_type"], "CURRENT")
        self.assertEqual(
            current["validation"]["missing_documents"],
            ["AADHAAR_DEMO", "PAN_DEMO", "ADDRESS_PROOF_DEMO"],
        )

    # 5. Form validation
    def test_05_form_validation_rejects_underage_bad_pin_phone_and_ineligible_employment(self):
        app_view = start_application(self.arjun, "SALARY")
        ref = app_view["application_ref"]

        with self.assertRaises(ApplicationValidationError) as underage_ctx:
            update_application(self.arjun, ref, {"date_of_birth": "2012-05-01"})
        self.assertEqual(underage_ctx.exception.errors[0]["code"], "UNDER_MINIMUM_AGE")

        with self.assertRaises(ApplicationValidationError) as pin_ctx:
            update_application(self.arjun, ref, {"postal_code": "01234"})
        self.assertIn(pin_ctx.exception.errors[0]["code"], {"INVALID_LENGTH", "INVALID_FORMAT"})

        with self.assertRaises(ApplicationValidationError) as phone_ctx:
            update_application(self.arjun, ref, {"phone": "1234567890"})
        self.assertEqual(phone_ctx.exception.errors[0]["code"], "INVALID_FORMAT")

        with self.assertRaises(ApplicationValidationError) as emp_ctx:
            update_application(self.arjun, ref, {"employment_category": "STUDENT"})
        self.assertEqual(emp_ctx.exception.errors[0]["code"], "EMPLOYMENT_NOT_ELIGIBLE")

    # 6. Missing fields detection
    def test_06_missing_fields_and_documents_detected_deterministically(self):
        app_view = start_application(self.arjun, "SAVINGS")
        ref = app_view["application_ref"]
        validated = validate_application(self.arjun, ref)
        missing_fields = set(validated["validation"]["missing_fields"])
        self.assertEqual(
            missing_fields,
            {"date_of_birth", "address_line", "city", "state", "postal_code"},
        )
        self.assertEqual(
            validated["validation"]["missing_documents"],
            ["AADHAAR_DEMO", "PAN_DEMO"],
        )
        self.assertFalse(validated["validation"]["is_valid"])

    # 7. Document upload validation
    def test_07_valid_document_uploads_stored_safely_and_never_sent_to_llm(self):
        app_view = start_application(self.arjun, "CURRENT")
        ref = app_view["application_ref"]
        self._complete_arjun_details(ref, "CURRENT")
        updated = self._upload_required_docs(ref, "CURRENT")
        self.assertTrue(updated["validation"]["is_valid"])
        self.assertEqual(updated["validation"]["missing_documents"], [])

        raw_bytes, media_type = open_document(self.arjun, ref, "AADHAAR_DEMO")
        self.assertEqual(raw_bytes, VALID_PDF)
        self.assertEqual(media_type, "application/pdf")

        # Tool projection must never contain document bytes, storage paths, or personal fields
        tool_proj = execute_tool("get_account_application", self.arjun, {"application_ref": ref})
        self.assertNotIn("fields", tool_proj)
        self.assertNotIn("review_digest", tool_proj)
        self.assertNotIn("Arjun Sharma", str(tool_proj))

        # Replacing or deleting a document updates validation state cleanly
        store_document(self.arjun, ref, "AADHAAR_DEMO", VALID_PNG, "image/png")
        new_bytes, new_media = open_document(self.arjun, ref, "AADHAAR_DEMO")
        self.assertEqual(new_bytes, VALID_PNG)
        self.assertEqual(new_media, "image/png")
        after_delete = delete_document(self.arjun, ref, "AADHAAR_DEMO")
        self.assertIn("AADHAAR_DEMO", after_delete["validation"]["missing_documents"])

    # 8. Invalid file rejection
    def test_08_invalid_files_rejected_by_size_and_magic_bytes(self):
        app_view = start_application(self.arjun, "SAVINGS")
        ref = app_view["application_ref"]

        with self.assertRaises(DocumentRejected):
            store_document(self.arjun, ref, "AADHAAR_DEMO", b"", "application/pdf")

        with self.assertRaises(DocumentRejected):
            store_document(
                self.arjun,
                ref,
                "AADHAAR_DEMO",
                VALID_PDF + b"0" * (MAX_DOCUMENT_BYTES + 10),
                "application/pdf",
            )

        with self.assertRaises(DocumentRejected):
            store_document(
                self.arjun,
                ref,
                "AADHAAR_DEMO",
                b"MZ\x90\x00not-a-pdf-or-image",
                "application/pdf",
            )

        with self.assertRaises(DocumentRejected):
            store_document(self.arjun, ref, "PAN_DEMO", VALID_PNG, "application/pdf")

    # 9. Customer ownership & cross-customer isolation
    def test_09_customer_ownership_prevents_cross_customer_access(self):
        arjun_app = start_application(self.arjun, "SAVINGS")
        ref = arjun_app["application_ref"]
        store_document(self.arjun, ref, "AADHAAR_DEMO", VALID_PDF, "application/pdf")

        with self.assertRaises(ApplicationNotFound):
            get_application(self.priya, ref)
        with self.assertRaises(ApplicationNotFound):
            update_application(self.priya, ref, {"city": "Mumbai"})
        with self.assertRaises(ApplicationNotFound):
            open_document(self.priya, ref, "AADHAAR_DEMO")
        with self.assertRaises(ApplicationNotFound):
            prepare_submission(self.priya, ref)

        priya_status = get_application_status(self.priya, application_ref=ref)
        self.assertFalse(priya_status["found"])

    # 10 & 11. Application saving as DRAFT and resumption
    def test_10_11_application_saved_as_draft_and_resumed_cleanly(self):
        first = start_application(self.arjun, "SAVINGS")
        ref = first["application_ref"]
        update_application(self.arjun, ref, {"city": "Bengaluru", "state": "Karnataka"})

        # Starting again while an active draft exists resumes the same application
        resumed = start_application(self.arjun)
        self.assertEqual(resumed["application_ref"], ref)
        self.assertTrue(resumed["resumed"])
        self.assertEqual(resumed["fields"]["city"], "Bengaluru")

        current = get_current_application(self.arjun)
        self.assertIsNotNone(current)
        self.assertEqual(current["application_ref"], ref)

    # 12. Application review and review invalidation on edit
    def test_12_prepare_review_sets_pending_confirmation_and_edit_invalidates_digest(self):
        app_view = start_application(self.arjun, "SAVINGS")
        ref = app_view["application_ref"]
        self._complete_arjun_details(ref, "SAVINGS")
        self._upload_required_docs(ref, "SAVINGS")

        prepared = prepare_submission(self.arjun, ref)
        self.assertTrue(prepared["ready"])
        self.assertEqual(prepared["application"]["status"], "PENDING_CONFIRMATION")
        digest = prepared["review_digest"]
        self.assertIsNotNone(digest)
        self.assertEqual(len(digest), 64)

        # Any subsequent edit returns the application to DRAFT
        edited = update_application(self.arjun, ref, {"city": "Mysuru"})
        self.assertEqual(edited["status"], "DRAFT")

        # Old digest can no longer submit
        with self.assertRaises(ApplicationConflict):
            submit_application(
                self.arjun, ref, confirmed=True, confirmation_method="button", review_digest=digest
            )

        # Re-prepare and test explicit return_to_draft
        prepared_again = prepare_submission(self.arjun, ref)
        self.assertEqual(prepared_again["application"]["status"], "PENDING_CONFIRMATION")
        draft_again = return_to_draft(self.arjun, ref)
        self.assertEqual(draft_again["status"], "DRAFT")

    # 13. Confirmation required before submission
    def test_13_submission_requires_explicit_button_confirmation_and_matching_digest(self):
        # Verify no submit tool is exposed in get_available_tools
        tool_names = {t["name"] for t in get_available_tools()}
        self.assertNotIn("submit_account_application", tool_names)
        with self.assertRaises(ValueError):
            execute_tool("submit_account_application", self.arjun, {})

        app_view = start_application(self.arjun, "SAVINGS")
        ref = app_view["application_ref"]
        self._complete_arjun_details(ref, "SAVINGS")
        self._upload_required_docs(ref, "SAVINGS")

        # Cannot submit directly from DRAFT before prepare_submission
        with self.assertRaises(ApplicationConflict):
            submit_application(
                self.arjun,
                ref,
                confirmed=True,
                confirmation_method="button",
                review_digest="a" * 64,
            )

        prepared = prepare_submission(self.arjun, ref)
        digest = prepared["review_digest"]

        # Unconfirmed rejected
        with self.assertRaises(ApplicationConflict):
            submit_application(
                self.arjun, ref, confirmed=False, confirmation_method="button", review_digest=digest
            )

        # Non-button method rejected
        with self.assertRaises(ApplicationConflict):
            submit_application(
                self.arjun, ref, confirmed=True, confirmation_method="voice", review_digest=digest
            )

        # Wrong digest rejected
        with self.assertRaises(ApplicationConflict):
            submit_application(
                self.arjun, ref, confirmed=True, confirmation_method="button", review_digest="0" * 64
            )

    # 14. Submission idempotency
    def test_14_submission_is_idempotent_and_records_audit_trail(self):
        app_view = start_application(self.arjun, "SAVINGS")
        ref = app_view["application_ref"]
        self._complete_arjun_details(ref, "SAVINGS")
        self._upload_required_docs(ref, "SAVINGS")
        prepared = prepare_submission(self.arjun, ref)
        digest = prepared["review_digest"]

        first_submit = submit_application(
            self.arjun, ref, confirmed=True, confirmation_method="button", review_digest=digest
        )
        self.assertEqual(first_submit["status"], "SUBMITTED")
        self.assertIsNotNone(first_submit["submitted_at"])
        self.assertFalse(first_submit["already_submitted"])

        second_submit = submit_application(
            self.arjun, ref, confirmed=True, confirmation_method="button", review_digest=digest
        )
        self.assertEqual(second_submit["status"], "SUBMITTED")
        self.assertTrue(second_submit["already_submitted"])
        self.assertEqual(second_submit["submitted_at"], first_submit["submitted_at"])

        # Submitted application cannot be modified
        with self.assertRaises(ApplicationConflict):
            update_application(self.arjun, ref, {"city": "Chennai"})

    # 15 & 16. Application status retrieval and REST API + unauthorized rejection
    def test_15_16_rest_api_workflow_status_and_unauthorized_rejection(self):
        # Verify router and route registration on FastAPI application
        included_routers = [
            getattr(route, "original_router", None) for route in main.app.routes
        ]
        self.assertIn(account_application_routes.router, included_routers)
        registered_paths = {
            getattr(route, "path", "")
            for route in account_application_routes.router.routes
        }
        self.assertIn("/api/v1/me/account-applications", registered_paths)
        self.assertIn("/api/v1/me/account-applications/options", registered_paths)
        self.assertIn("/api/v1/me/account-applications/current", registered_paths)
        self.assertIn("/api/v1/me/account-applications/status", registered_paths)
        self.assertIn("/api/v1/me/account-applications/{application_ref}", registered_paths)
        self.assertIn(
            "/api/v1/me/account-applications/{application_ref}/documents/{document_type}",
            registered_paths,
        )
        self.assertIn(
            "/api/v1/me/account-applications/{application_ref}/prepare-review",
            registered_paths,
        )
        self.assertIn(
            "/api/v1/me/account-applications/{application_ref}/submit",
            registered_paths,
        )

        # Unauthenticated request is rejected with 401
        unauth_req = Mock()
        unauth_req.cookies = {}
        with self.assertRaises(HTTPException) as unauth_ctx:
            auth.get_authenticated_principal(unauth_req)
        self.assertEqual(unauth_ctx.exception.status_code, 401)

        # Sign in as Arjun via signed session cookie and resolve principal
        now_ts = int(datetime.now(timezone.utc).timestamp())
        arjun_token = auth._encode_session_token("demo-arjun", now_ts, now_ts + 3600)
        arjun_req = Mock()
        arjun_req.cookies = {auth.SESSION_COOKIE_NAME: arjun_token}
        principal = auth.get_authenticated_principal(arjun_req)
        self.assertEqual(principal.customer_key, "demo-arjun")

        options_resp = account_application_routes.get_options_endpoint(principal=principal)
        self.assertEqual(len(options_resp["account_types"]), 3)

        created_resp = account_application_routes.start_application_endpoint(
            payload=AccountApplicationStartRequest(account_type="SAVINGS"),
            principal=principal,
        )
        ref = created_resp["application_ref"]

        patch_resp = account_application_routes.update_application_endpoint(
            payload=AccountApplicationUpdateRequest(
                date_of_birth="1996-05-14",
                phone="9876543210",
                address_line="42 MG Road, Indiranagar",
                city="Bengaluru",
                state="Karnataka",
                postal_code="560038",
                employment_category="SALARIED",
            ),
            application_ref=ref,
            principal=principal,
        )
        self.assertEqual(patch_resp["fields"]["city"], "Bengaluru")

        class _StubUpload:
            def __init__(self, data: bytes, content_type: str):
                self._data = data
                self.content_type = content_type
                self.closed = False

            async def read(self, _size: int = -1) -> bytes:
                return self._data

            async def close(self) -> None:
                self.closed = True

        def _run_coro(coro):
            try:
                coro.send(None)
            except StopIteration as stop:
                return stop.value
            raise AssertionError("Coroutine did not complete synchronously")

        up1_file = _StubUpload(VALID_PDF, "application/pdf")
        up1 = _run_coro(
            account_application_routes.upload_document_endpoint(
                document_type="AADHAAR_DEMO",
                file=up1_file,
                application_ref=ref,
                principal=principal,
            )
        )
        self.assertTrue(up1_file.closed)
        self.assertTrue(
            any(
                d["document_type"] == "AADHAAR_DEMO" and d["uploaded"]
                for d in up1["documents"]
            )
        )

        up2_file = _StubUpload(VALID_PNG, "image/png")
        up2 = _run_coro(
            account_application_routes.upload_document_endpoint(
                document_type="PAN_DEMO",
                file=up2_file,
                application_ref=ref,
                principal=principal,
            )
        )
        self.assertTrue(up2["validation"]["is_valid"])

        prep = account_application_routes.prepare_review_endpoint(
            application_ref=ref, principal=principal
        )
        digest = prep["review_digest"]
        current_pending = account_application_routes.get_current_application_endpoint(
            principal=principal
        )
        self.assertEqual(current_pending["application"]["review_digest"], digest)

        own_doc_resp = account_application_routes.get_document_content_endpoint(
            document_type="AADHAAR_DEMO", application_ref=ref, principal=principal
        )
        self.assertEqual(own_doc_resp.body, VALID_PDF)
        self.assertEqual(own_doc_resp.media_type, "application/pdf")

        sub = account_application_routes.submit_application_endpoint(
            payload=AccountApplicationSubmitRequest(
                confirmed=True, confirmation_method="button", review_digest=digest
            ),
            application_ref=ref,
            principal=principal,
        )
        self.assertEqual(sub["status"], "SUBMITTED")

        status_resp = account_application_routes.get_application_status_endpoint(
            application_ref=ref, account_type=None, principal=principal
        )
        self.assertEqual(status_resp["status"], "SUBMITTED")

        # Switch session to Priya and confirm Arjun's application is inaccessible (404)
        priya_token = auth._encode_session_token("demo-priya", now_ts, now_ts + 3600)
        priya_req = Mock()
        priya_req.cookies = {auth.SESSION_COOKIE_NAME: priya_token}
        priya_principal = auth.get_authenticated_principal(priya_req)
        with self.assertRaises(HTTPException) as forbidden_ctx:
            account_application_routes.get_application_endpoint(
                application_ref=ref, principal=priya_principal
            )
        self.assertEqual(forbidden_ctx.exception.status_code, 404)
        with self.assertRaises(HTTPException) as forbidden_doc_ctx:
            account_application_routes.get_document_content_endpoint(
                document_type="AADHAAR_DEMO",
                application_ref=ref,
                principal=priya_principal,
            )
        self.assertEqual(forbidden_doc_ctx.exception.status_code, 404)

    # 17. Voice and text continuity across AssistantOrchestrator
    def test_17_voice_and_text_continuity_across_account_opening_workflow(self):
        spoken_calls = []

        class StubTranscriber:
            def __init__(self):
                self.transcripts = [
                    Transcription(
                        "I am salaried",
                        ("en-IN",),
                        False,
                    ),
                    Transcription(
                        "What is the status of my savings account application?",
                        ("en-IN",),
                        False,
                    ),
                ]

            def transcribe(self, principal, upload_id, language_hint):
                return self.transcripts.pop(0)

        class StubSynthesizer:
            def synthesize(self, principal, text, language):
                spoken_calls.append((text, language))
                return SpeechOutput(
                    audio_url="/api/v1/voice/responses/voice-1",
                    media_type="audio/wav",
                    language=language,
                    expires_at=datetime.now(timezone.utc) + timedelta(minutes=10),
                )

        orchestrator = AssistantOrchestrator(
            llm_provider=MockLLMProvider([]),
            transcriber=StubTranscriber(),
            synthesizer=StubSynthesizer(),
        )

        # Step 1 (Text): Start account application via Vaani AI
        turn1 = orchestrator.process_turn(
            self.arjun,
            make_turn_request(
                "Vani, I want to open another savings account. Can you help me complete the process?"
            ),
        )
        self.assertEqual(turn1.tool_results[0].tool_name, "start_account_application")
        self.assertEqual(turn1.tool_results[0].status, "succeeded")
        nav1 = turn1.tool_results[0].data["ui_navigation"]
        self.assertEqual(nav1["subfeature"], "Open New Account")
        ref = turn1.tool_results[0].data["result"]["application_ref"]
        conv_id = turn1.conversation_id

        # Step 2 (Voice): Provide employment category over speech in the same conversation
        turn2 = orchestrator.process_turn(
            self.arjun,
            make_turn_request(
                conversation_id=conv_id,
                input_mode="speech",
                output_mode="speech_and_text",
                audio_upload_id=uuid4(),
            ),
        )
        self.assertEqual(turn2.transcript, "I am salaried")
        self.assertEqual(turn2.tool_results[0].tool_name, "update_account_application")
        self.assertIn("salaried", turn2.assistant_text.casefold())
        self.assertEqual(len(spoken_calls), 1)
        self.assertIsNotNone(turn2.speech_output)

        # Step 3 (Text): Ask what is remaining
        turn3 = orchestrator.process_turn(
            self.arjun,
            make_turn_request("What is remaining?", conversation_id=conv_id),
        )
        self.assertEqual(turn3.tool_results[0].tool_name, "validate_account_application")
        self.assertIn("date of birth", turn3.assistant_text.casefold())

        # Step 4 (Text): Ask where to upload Aadhaar
        turn4 = orchestrator.process_turn(
            self.arjun,
            make_turn_request("Where do I upload my Aadhaar?", conversation_id=conv_id),
        )
        self.assertEqual(turn4.tool_results[0].tool_name, "get_application_document_status")
        self.assertEqual(
            turn4.tool_results[0].data["ui_navigation"]["subfeature"], "KYC Documents"
        )

        # Complete details and documents, then ask Vaani AI to review/submit
        self._complete_arjun_details(ref, "SAVINGS")
        self._upload_required_docs(ref, "SAVINGS")
        turn5 = orchestrator.process_turn(
            self.arjun,
            make_turn_request("Review and submit my application", conversation_id=conv_id),
        )
        self.assertEqual(
            turn5.tool_results[0].tool_name, "validate_account_application"
        )
        self.assertEqual(
            turn5.tool_results[0].data["ui_navigation"]["subfeature"], "Review & Submit"
        )
        # Review navigation does not prepare or submit the application.
        current_after_review = get_application(self.arjun, ref)
        self.assertEqual(current_after_review["status"], "DRAFT")

        # Step 6 (Voice): Ask for application status over voice
        turn6 = orchestrator.process_turn(
            self.arjun,
            make_turn_request(
                conversation_id=conv_id,
                input_mode="speech",
                output_mode="speech_and_text",
                audio_upload_id=uuid4(),
            ),
        )
        self.assertEqual(turn6.tool_results[0].tool_name, "get_account_application_status")
        self.assertIn(ref, turn6.assistant_text)
        self.assertEqual(
            turn6.tool_results[0].data["ui_navigation"]["subfeature"], "Track Applications"
        )

    # 18. Frontend API client, navigation & session state integration
    def test_18_frontend_client_navigation_and_session_state_isolation(self):
        self.assertEqual(len(FEATURE_SUBFEATURES), 5)
        self.assertEqual(FEATURE_SUBFEATURES["Banking Information"][0], "Overview")
        for sub in ("Open New Account", "KYC Documents", "Review & Submit", "Track Applications"):
            self.assertIn(sub, FEATURE_SUBFEATURES["Account Opening Assistant"])

        state = {}
        initialize_session_state(state, api_factory=lambda: Mock())
        mark_session_valid(state, {"customer_key": "demo-arjun", "display_name": "Arjun"})
        set_subfeature(state, "Account Opening Assistant", "Open New Account")
        self.assertEqual(ensure_subfeature(state, "Account Opening Assistant"), "Open New Account")

        state["account_application_cache"] = {"application_ref": "VSA-0123456789ABCDEF"}
        mark_session_valid(state, {"customer_key": "demo-priya", "display_name": "Priya"})
        self.assertIsNone(state["account_application_cache"])

        state["account_application_selected_ref"] = "VSA-FEDCBA9876543210"
        clear_account_application_cache(state)
        self.assertIsNone(state["account_application_selected_ref"])

        sign_out(state, api_factory=lambda: Mock())
        self.assertIsNone(state["account_application_cache"])

        session = Mock()
        resp = Mock()
        resp.status_code = 200
        resp.json.return_value = {"found": True}
        session.request.return_value = resp
        api = VaaniShieldAPI("http://localhost:8000", session=session)
        api.get_account_application("VSA-0123456789ABCDEF")
        self.assertEqual(
            session.request.call_args.args[1],
            "http://localhost:8000/api/v1/me/account-applications/VSA-0123456789ABCDEF",
        )
        with self.assertRaises(ValueError):
            api.get_account_application("../../bad-ref")


if __name__ == "__main__":
    unittest.main()
