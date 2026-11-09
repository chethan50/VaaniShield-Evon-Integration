"""Authenticated client for the VaaniShield FastAPI endpoints."""

from __future__ import annotations

import os
import re
import logging
from urllib.parse import urljoin, urlparse
from uuid import UUID, uuid4

import requests

logger = logging.getLogger(__name__)


class APIError(RuntimeError):
    def __init__(self, status_code: int, message: str, payload=None):
        super().__init__(message)
        self.status_code = status_code
        self.payload = payload


def build_text_turn_request(
    conversation_id,
    text: str,
    output_mode: str,
    language_hint: str = "auto",
    response_language: str | None = None,
    scam_assessment_id: str | None = None,
    attachments: list[dict] | None = None,
):
    if not conversation_id:
        raise ValueError("A server-created conversation is required.")
    if not text or not text.strip():
        raise ValueError("Enter a message before sending.")
    if output_mode not in {"text", "speech", "speech_and_text"}:
        raise ValueError("Unsupported assistant output mode.")
    payload = {
        "conversation_id": str(conversation_id),
        "client_turn_id": str(uuid4()),
        "input_mode": "text",
        "output_mode": output_mode,
        "text": text.strip(),
        "language_hint": language_hint,
    }
    if response_language:
        payload["response_language"] = response_language
    if scam_assessment_id:
        payload["scam_assessment_id"] = str(scam_assessment_id)
    if attachments:
        payload["attachments"] = attachments
    return payload


def build_speech_turn_request(
    conversation_id,
    audio_upload_id,
    output_mode: str,
    language_hint: str = "auto",
    response_language: str | None = None,
    attachments: list[dict] | None = None,
):
    if not conversation_id:
        raise ValueError("A server-created conversation is required.")
    if not audio_upload_id:
        raise ValueError("An uploaded audio ID is required.")
    if output_mode not in {"text", "speech", "speech_and_text"}:
        raise ValueError("Unsupported assistant output mode.")
    payload = {
        "conversation_id": str(conversation_id),
        "client_turn_id": str(uuid4()),
        "input_mode": "speech",
        "output_mode": output_mode,
        "audio_upload_id": str(audio_upload_id),
        "language_hint": language_hint,
    }
    if response_language:
        payload["response_language"] = response_language
    if attachments:
        payload["attachments"] = attachments
    return payload


class VaaniShieldAPI:
    """Holds the demo-session cookie server-side in a requests.Session."""

    def __init__(self, base_url: str | None = None, session=None, timeout: int = 20):
        self.base_url = (base_url or os.getenv(
            "VAANISHIELD_API_BASE", "http://127.0.0.1:8000"
        )).rstrip("/")
        self.session = session or requests.Session()
        self.timeout = timeout

    def _url(self, path: str) -> str:
        if path.startswith(("http://", "https://")):
            candidate = urlparse(path)
            base = urlparse(self.base_url)
            if (candidate.scheme, candidate.netloc) != (base.scheme, base.netloc):
                raise ValueError("API requests must stay on the configured backend host.")
            return path
        return urljoin(self.base_url + "/", path.lstrip("/"))

    @staticmethod
    def _read_payload(response):
        try:
            return response.json()
        except (ValueError, requests.exceptions.JSONDecodeError):
            return None

    def _request(self, method: str, path: str, *, expected=(200,), **kwargs):
        request_timeout = kwargs.pop("timeout", self.timeout)
        try:
            response = self.session.request(
                method,
                self._url(path),
                timeout=request_timeout,
                **kwargs,
            )
        except requests.exceptions.ConnectTimeout:
            logger.warning("frontend_api_transport_failure kind=connect_timeout")
            raise APIError(0, "The VaaniShield API connection timed out.", {"transport": "connect_timeout"}) from None
        except requests.exceptions.ReadTimeout:
            logger.warning("frontend_api_transport_failure kind=read_timeout")
            raise APIError(0, "The VaaniShield API did not respond before the request timed out.", {"transport": "read_timeout"}) from None
        except requests.exceptions.ConnectionError:
            logger.warning("frontend_api_transport_failure kind=connection_error")
            raise APIError(0, "The VaaniShield API could not be reached. Check that the backend is running.", {"transport": "connection_error"}) from None
        except requests.exceptions.RequestException:
            logger.warning("frontend_api_transport_failure kind=request_error")
            raise APIError(0, "The VaaniShield API request could not be completed.", {"transport": "request_error"}) from None

        if response.status_code not in expected:
            payload = self._read_payload(response)
            error_code = response.headers.get("X-Vaani-Error-Code")
            if isinstance(payload, dict) and isinstance(error_code, str) and error_code:
                payload["error_code"] = error_code
            # Only API-authored detail messages are surfaced. Transport exception
            # strings and request headers (including cookies) are never included.
            detail = payload.get("detail") if isinstance(payload, dict) else None
            if isinstance(detail, dict):
                message = detail.get("message")
            else:
                message = detail
            if not isinstance(message, str) or not message:
                message = "The request could not be completed."
            raise APIError(response.status_code, message, payload)
        if response.status_code == 204:
            return None
        return response

    def _json(self, method: str, path: str, *, expected=(200,), **kwargs):
        response = self._request(method, path, expected=expected, **kwargs)
        if response is None:
            return None
        return self._read_payload(response)

    def create_demo_session(self, customer_key: str):
        return self._json(
            "POST", "/api/v1/auth/demo/session", expected=(200,),
            json={"customer_key": customer_key},
        )

    def demo_login(self, email: str, password: str):
        return self._json(
            "POST", "/api/v1/auth/demo/login", expected=(200,),
            json={"email": email, "password": password},
        )

    def get_security_activity(self):
        return self._json("GET", "/api/v1/auth/demo/security-activity")

    def create_android_sms_pairing_code(self, device_name: str | None = None):
        """Create a short-lived Android pairing code with the current demo session."""
        kwargs = {"json": {"device_name": device_name}} if device_name else {}
        return self._json(
            "POST",
            "/api/v1/me/scam/devices/pair-token",
            expected=(201,),
            **kwargs,
        )

    def get_demo_session(self):
        return self._json("GET", "/api/v1/auth/demo/session")

    def delete_demo_session(self):
        return self._json(
            "DELETE", "/api/v1/auth/demo/session", expected=(200, 204)
        )

    def list_conversations(self, limit=25, cursor=None):
        params = {"limit": limit}
        if cursor:
            params["cursor"] = cursor
        return self._json(
            "GET", "/api/v1/assistant/conversations", params=params
        )

    def create_conversation(self, title: str | None = None):
        return self._json(
            "POST", "/api/v1/assistant/conversations", expected=(201,),
            json={"title": title} if title else {},
        )

    def get_conversation_turns(self, conversation_id, limit=25, cursor=None):
        params = {"limit": limit}
        if cursor:
            params["cursor"] = cursor
        return self._json(
            "GET",
            f"/api/v1/assistant/conversations/{conversation_id}/turns",
            params=params,
        )

    def delete_conversation_turn(self, conversation_id, turn_id):
        return self._json(
            "DELETE",
            f"/api/v1/assistant/conversations/{conversation_id}/turns/{turn_id}",
            expected=(204,),
        )

    def upload_audio(self, filename: str, content: bytes, media_type="audio/wav"):
        return self._json(
            "POST", "/api/v1/voice/uploads", expected=(201,),
            files={"file": (filename, content, media_type)},
        )

    def send_assistant_turn(self, payload: dict):
        return self._json(
            "POST", "/api/v1/assistant/turns", expected=(200,), json=payload,
            # Complex tool turns execute sequentially and can legitimately
            # exceed the shorter dashboard request timeout.
            timeout=max(self.timeout, 300),
        )

    def assess_scam_message(self, message_text: str, language_hint="auto"):
        return self._json(
            "POST", "/api/v1/me/scam/assessments/messages", expected=(201,),
            json={"message_text": message_text, "language_hint": language_hint},
        )

    def assess_scam_call(self, audio_upload_id, language_hint="auto"):
        return self._json(
            "POST", "/api/v1/me/scam/assessments/calls", expected=(201,),
            json={"audio_upload_id": str(audio_upload_id), "language_hint": language_hint},
        )

    def list_scam_assessments(self, limit=25, cursor=None):
        params = {"limit": limit}
        if cursor:
            params["cursor"] = cursor
        return self._json("GET", "/api/v1/me/scam/assessments", params=params)

    def list_scam_reports(
        self, limit=25, cursor=None, *, source_type=None, risk_level=None,
        date_from=None, date_to=None, search=None,
    ):
        """List unified reports through the current authenticated session."""
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 100:
            raise ValueError("Report limit must be between 1 and 100.")
        params = {"limit": limit}
        if cursor:
            params["cursor"] = cursor
        if source_type:
            params["source_type"] = source_type
        if risk_level:
            params["risk_level"] = risk_level
        if date_from:
            params["from"] = str(date_from)
        if date_to:
            params["to"] = str(date_to)
        if search:
            params["search"] = search
        return self._json("GET", "/api/v1/me/scam/reports", params=params)

    def get_scam_report(self, source_type: str, report_id: str):
        allowed = {"MANUAL_MESSAGE", "INCOMING_SMS", "VOICE_CALL"}
        source = str(source_type).upper()
        if source not in allowed:
            raise ValueError("Unsupported report source type.")
        try:
            canonical_id = str(UUID(str(report_id)))
        except (ValueError, TypeError, AttributeError):
            raise ValueError("A valid report identifier is required.") from None
        return self._json(
            "GET", f"/api/v1/me/scam/reports/{source}/{canonical_id}"
        )

    def get_my_context(self, year=None, month=None):
        params = {}
        if year is not None:
            params["year"] = year
        if month is not None:
            params["month"] = month
        return self._json("GET", "/api/v1/me/context", params=params or None)

    def get_my_accounts(self):
        return self._json("GET", "/api/v1/me/accounts")

    def get_my_balance(self):
        return self._json("GET", "/api/v1/me/balance")

    def get_my_transactions(self, limit=100):
        return self._json("GET", "/api/v1/me/transactions", params={"limit": limit})

    def assess_transaction_risk(self, transaction_ref: str):
        if not isinstance(transaction_ref, str) or not re.fullmatch(r"TR-[A-F0-9]{48}", transaction_ref):
            raise ValueError("A valid public transaction reference is required.")
        return self._json(
            "POST", "/api/v1/me/scam/assessments/transactions",
            expected=(200, 201), json={"transaction_ref": str(transaction_ref)},
        )

    def list_transaction_risk_assessments(self, limit=25, cursor=None):
        params = {"limit": limit}
        if cursor:
            params["cursor"] = cursor
        return self._json(
            "GET", "/api/v1/me/scam/transaction-assessments", params=params
        )

    def get_transaction_risk_assessment(self, assessment_id: str):
        try:
            assessment_id = str(UUID(str(assessment_id)))
        except (ValueError, TypeError, AttributeError):
            raise ValueError("A valid public assessment identifier is required.") from None
        return self._json(
            "GET", f"/api/v1/me/scam/transaction-assessments/{assessment_id}"
        )

    def search_my_transactions(self, params=None):
        return self._json(
            "GET", "/api/v1/me/transactions/search", params=params or None
        )

    def get_my_cards(self):
        return self._json("GET", "/api/v1/me/cards")

    def get_my_loans(self):
        return self._json("GET", "/api/v1/me/loans")

    def get_my_financial_summary(self):
        return self._json("GET", "/api/v1/me/financial/summary")

    def get_my_financial_categories(self):
        return self._json("GET", "/api/v1/me/financial/categories")

    def get_my_financial_merchants(self, limit=5):
        return self._json(
            "GET", "/api/v1/me/financial/merchants", params={"limit": limit}
        )

    def get_my_monthly_comparison(self, year=None, month=None):
        params = {}
        if year is not None:
            params["year"] = year
        if month is not None:
            params["month"] = month
        return self._json(
            "GET", "/api/v1/me/financial/monthly-comparison", params=params or None
        )

    def get_my_financial_advisory(self, year=None, month=None):
        params = {}
        if year is not None:
            params["year"] = year
        if month is not None:
            params["month"] = month
        return self._json(
            "GET", "/api/v1/me/financial/advisory", params=params or None
        )

    def get_notifications(self, limit: int = 25):
        return self._json("GET", "/api/v1/me/notifications", params={"limit": limit})

    def mark_notification_read(self, notification_id: str):
        return self._json("POST", f"/api/v1/me/notifications/{notification_id}/read")

    def mark_all_notifications_read(self):
        return self._json("POST", "/api/v1/me/notifications/read-all")

    def upload_attachment(self, filename: str, content: bytes, media_type: str = "application/octet-stream", conversation_id: str | None = None):
        files = {"file": (filename, content, media_type)}
        data = {"conversation_id": conversation_id} if conversation_id else {}
        return self._json("POST", "/api/v1/assistant/attachments", files=files, data=data)

    def create_support_ticket(self, payload: dict):
        return self._json(
            "POST", "/api/v1/me/support/tickets", expected=(201,), json=payload
        )

    def list_support_tickets(
        self, limit=25, cursor=None, *, status=None, category=None, priority=None
    ):
        params = {"limit": limit}
        if cursor:
            params["cursor"] = cursor
        if status:
            params["status"] = status
        if category:
            params["category"] = category
        if priority:
            params["priority"] = priority
        return self._json("GET", "/api/v1/me/support/tickets", params=params)

    def get_support_ticket(self, ticket_ref: str):
        return self._json(
            "GET", f"/api/v1/me/support/tickets/{ticket_ref}"
        )

    def get_support_ticket_events(self, ticket_ref: str):
        return self._json(
            "GET", f"/api/v1/me/support/tickets/{ticket_ref}/events"
        )

    def close_support_ticket(self, ticket_ref: str):
        return self._json(
            "PATCH", f"/api/v1/me/support/tickets/{ticket_ref}/status",
            json={
                "status": "CLOSED",
                "confirmed": True,
                "confirmation_method": "button",
            },
        )

    def confirm_assistant_action(
        self, confirmation_id, *, confirmed, conversation_id, original_turn_id,
        action_type="create_support_ticket",
    ):
        return self._json(
            "POST", f"/api/v1/assistant/confirmations/{confirmation_id}",
            json={
                "confirmed": bool(confirmed),
                "confirmation_method": "button",
                "conversation_id": str(conversation_id),
                "original_turn_id": str(original_turn_id),
                "action_type": action_type,
            },
        )

    def confirm_investigation_action(
        self, confirmation_id, *, confirmed, conversation_id, original_turn_id,
        action_type,
    ):
        try:
            confirmation_id = str(UUID(str(confirmation_id)))
            conversation_id = str(UUID(str(conversation_id)))
            original_turn_id = str(UUID(str(original_turn_id)))
        except (ValueError, TypeError, AttributeError):
            raise ValueError("The investigation confirmation reference is invalid.") from None
        return self._json(
            "POST", f"/api/v1/assistant/investigation-confirmations/{confirmation_id}",
            json={
                "confirmed": bool(confirmed),
                "confirmation_method": "button",
                "conversation_id": conversation_id,
                "original_turn_id": original_turn_id,
                "action_type": action_type,
            },
        )

    def list_investigations(self, limit=25, cursor=None, *, status=None, category=None, search=None):
        params = {"limit": limit}
        if cursor:
            params["cursor"] = cursor
        if status:
            params["status"] = status
        if category:
            params["category"] = category
        if search:
            params["search"] = search
        return self._json("GET", "/api/v1/me/scam/investigations", params=params)

    def get_investigation(self, investigation_ref: str):
        return self._json("GET", f"/api/v1/me/scam/investigations/{investigation_ref}")

    def create_investigation(self, payload: dict):
        return self._json(
            "POST", "/api/v1/me/scam/investigations", expected=(201,), json=payload,
        )

    def link_investigation_evidence(self, investigation_ref: str, evidence: dict):
        return self._json(
            "POST", f"/api/v1/me/scam/investigations/{investigation_ref}/evidence",
            expected=(200,), json={"evidence": evidence, "confirmed": True},
        )

    def add_investigation_note(self, investigation_ref: str, note: str):
        return self._json(
            "POST", f"/api/v1/me/scam/investigations/{investigation_ref}/notes",
            expected=(200,), json={"note": note, "confirmed": True},
        )

    def update_investigation_status(self, investigation_ref: str, status: str):
        return self._json(
            "PATCH", f"/api/v1/me/scam/investigations/{investigation_ref}/status",
            json={"status": status, "confirmed": True},
        )

    def close_investigation(self, investigation_ref: str):
        return self._json(
            "POST", f"/api/v1/me/scam/investigations/{investigation_ref}/close",
            json={"confirmed": True},
        )

    def get_account_application_options(self):
        return self._json("GET", "/api/v1/me/account-applications/options")

    def list_account_applications(self, limit=20):
        return self._json(
            "GET", "/api/v1/me/account-applications", params={"limit": limit}
        )

    def start_account_application(self, account_type: str | None = None):
        payload = {"account_type": account_type} if account_type else {}
        return self._json(
            "POST", "/api/v1/me/account-applications", expected=(201,), json=payload,
        )

    def get_current_account_application(self):
        return self._json("GET", "/api/v1/me/account-applications/current")

    def get_account_application(self, application_ref: str):
        if not isinstance(application_ref, str) or not re.fullmatch(r"VSA-[A-F0-9]{16}", application_ref):
            raise ValueError("A valid public application reference is required.")
        return self._json("GET", f"/api/v1/me/account-applications/{application_ref}")

    def update_account_application(self, application_ref: str, changes: dict):
        if not isinstance(application_ref, str) or not re.fullmatch(r"VSA-[A-F0-9]{16}", application_ref):
            raise ValueError("A valid public application reference is required.")
        return self._json(
            "PATCH", f"/api/v1/me/account-applications/{application_ref}", json=changes,
        )

    def validate_account_application(self, application_ref: str):
        if not isinstance(application_ref, str) or not re.fullmatch(r"VSA-[A-F0-9]{16}", application_ref):
            raise ValueError("A valid public application reference is required.")
        return self._json(
            "POST", f"/api/v1/me/account-applications/{application_ref}/validate",
        )

    def get_account_application_documents(self, application_ref: str):
        if not isinstance(application_ref, str) or not re.fullmatch(r"VSA-[A-F0-9]{16}", application_ref):
            raise ValueError("A valid public application reference is required.")
        return self._json(
            "GET", f"/api/v1/me/account-applications/{application_ref}/documents",
        )

    def upload_account_application_document(
        self, application_ref: str, document_type: str, content: bytes, media_type: str,
    ):
        if not isinstance(application_ref, str) or not re.fullmatch(r"VSA-[A-F0-9]{16}", application_ref):
            raise ValueError("A valid public application reference is required.")
        return self._json(
            "POST",
            f"/api/v1/me/account-applications/{application_ref}/documents/{document_type}",
            files={"file": ("document", content, media_type)},
        )

    def delete_account_application_document(
        self, application_ref: str, document_type: str,
    ):
        if not isinstance(application_ref, str) or not re.fullmatch(r"VSA-[A-F0-9]{16}", application_ref):
            raise ValueError("A valid public application reference is required.")
        return self._json(
            "DELETE",
            f"/api/v1/me/account-applications/{application_ref}/documents/{document_type}",
        )

    def prepare_account_application_review(self, application_ref: str):
        if not isinstance(application_ref, str) or not re.fullmatch(r"VSA-[A-F0-9]{16}", application_ref):
            raise ValueError("A valid public application reference is required.")
        return self._json(
            "POST", f"/api/v1/me/account-applications/{application_ref}/prepare-review",
        )

    def return_account_application_to_draft(self, application_ref: str):
        if not isinstance(application_ref, str) or not re.fullmatch(r"VSA-[A-F0-9]{16}", application_ref):
            raise ValueError("A valid public application reference is required.")
        return self._json(
            "POST", f"/api/v1/me/account-applications/{application_ref}/return-to-draft",
        )

    def submit_account_application(
        self, application_ref: str, *, confirmed: bool, review_digest: str,
    ):
        if not isinstance(application_ref, str) or not re.fullmatch(r"VSA-[A-F0-9]{16}", application_ref):
            raise ValueError("A valid public application reference is required.")
        return self._json(
            "POST",
            f"/api/v1/me/account-applications/{application_ref}/submit",
            json={
                "confirmed": bool(confirmed),
                "confirmation_method": "button",
                "review_digest": str(review_digest),
            },
        )

    def get_account_application_status(
        self, *, application_ref: str | None = None, account_type: str | None = None,
    ):
        params = {}
        if application_ref:
            params["application_ref"] = application_ref
        if account_type:
            params["account_type"] = account_type
        return self._json(
            "GET", "/api/v1/me/account-applications/status", params=params or None,
        )

    def fetch_audio(self, audio_url: str):
        if not isinstance(audio_url, str) or not audio_url:
            raise ValueError("Assistant audio URL is missing.")
        url = self._url(audio_url)
        expected_path = "/api/v1/voice/responses/"
        if not urlparse(url).path.startswith(expected_path):
            raise ValueError("Unsupported assistant audio URL.")
        response = self._request("GET", url, expected=(200,))
        return response.content, response.headers.get("Content-Type", "audio/wav")

    def legacy_get(self, path: str, params=None):
        """Read-only compatibility calls for explicitly labeled demo dashboards."""
        return self._json("GET", path, params=params)

    def close(self):
        self.session.close()
