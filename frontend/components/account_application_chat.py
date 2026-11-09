"""In-chat views of the existing owner-scoped account application API."""

import hashlib

from frontend.api_client import APIError
from frontend.session_state import clear_account_application_cache


def upload_kyc(api, state, application, document_type, uploaded):
    """Use the same store and validation as KYC Documents; never general attachments."""
    raw = uploaded.getvalue()
    if not raw or len(raw) > 2 * 1024 * 1024:
        raise ValueError("Choose a nonempty PDF, PNG or JPEG up to 2 MB.")
    ref = application["application_ref"]
    digest = hashlib.sha256(raw).hexdigest()
    result = api.upload_account_application_document(
        ref, document_type, raw, getattr(uploaded, "type", None) or "application/octet-stream",
    )
    clear_account_application_cache(state)
    state["account_application_cache"] = result
    state["kyc_chat_receipt"] = f"Document saved to application {ref}. Visible in KYC Documents; not verified."
    state["kyc_chat_upload_key"] = (ref, document_type, digest)
    state["kyc_chat_epoch"] = state.get("kyc_chat_epoch", 0) + 1
    return result


def render_kyc_upload(st, api, state):
    with st.expander("KYC document", expanded=False):
        st.caption("PDF, PNG or JPEG · up to 2 MB · synthetic demo documents only. Upload is not verification.")
        # Fetch only after explicit selection, so ordinary chat does not add API calls.
        if st.button("Load my application", key="chat-load-kyc"):
            state.pop("kyc_chat_application", None)
            try:
                payload = api.get_current_account_application()
                if not isinstance(payload, dict) or "application" not in payload:
                    st.warning("Could not read the application response. Please try loading it again.")
                    return
                application = payload["application"]
                if application is None:
                    st.info("No active application found. Start an account application first.")
                    return
                if not isinstance(application, dict) or not isinstance(application.get("application_ref"), str) or not application["application_ref"]:
                    st.warning("The application details are incomplete. Please try loading them again.")
                    return
                state["kyc_chat_application"] = application
            except APIError as exc:
                st.warning("Start an account application first." if exc.status_code == 404 else f"Could not load the application: {exc}")
                return
        application = state.get("kyc_chat_application")
        if not application:
            return
        if not isinstance(application, dict) or not isinstance(application.get("application_ref"), str) or not application["application_ref"]:
            state.pop("kyc_chat_application", None)
            st.warning("The application details are incomplete. Please load the application again.")
            return
        st.caption(f"Application {application['application_ref']}")
        documents = application.get("documents")
        if not isinstance(documents, list):
            st.warning("The document checklist is unavailable. Please load the application again.")
            return
        choices = {doc["document_type"]: doc["label"] for doc in documents if isinstance(doc, dict) and doc.get("document_type") and doc.get("label")}
        if not application.get("account_type") or not choices:
            st.info("Choose an account type in chat, then load the application again.")
            return
        doc_type = st.selectbox("Document type", list(choices), format_func=choices.get, key="chat-kyc-type")
        uploaded = st.file_uploader("KYC document", type=["pdf", "png", "jpg", "jpeg"], key=f"chat-kyc-{state.get('kyc_chat_epoch', 0)}")
        if uploaded is not None and st.button("Save KYC document", key="chat-save-kyc"):
            try:
                state["kyc_chat_application"] = upload_kyc(api, state, application, doc_type, uploaded)
                st.rerun()
            except (APIError, ValueError, OSError) as exc:
                st.error(f"Could not upload the document: {exc}")


def render_application_review(st, api, state, ref, message_id):
    """Offer navigation only while the same owner-scoped application is ready."""
    from frontend.navigation import request_navigation
    from services.account_application_assistant import _ready_for_review
    try:
        application = api.get_account_application(ref)
    except APIError as exc:
        st.warning(str(exc))
        return
    if application.get("status") == "SUBMITTED":
        st.success(f"Application {ref} submitted to the demo workflow. No account has been opened.")
    elif _ready_for_review(application):
        if st.button("Go to Review & Submit", key=f"chat-review-nav-{message_id}"):
            state["account_application_selected_ref"] = ref
            state["account_application_review_target_ref"] = ref
            request_navigation(state, "Account Opening Assistant", "Review & Submit", step="review")
            st.rerun()
    else:
        st.info("Complete the required application fields and documents before review.")
