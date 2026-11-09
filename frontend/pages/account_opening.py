"""Streamlit UI for the VaaniShield Agentic Account Opening workflow."""

from __future__ import annotations

import html

from frontend.api_client import APIError
from frontend.components.fintech_ui import render_stepper_html
from frontend.navigation import (
    get_previous_navigation,
    navigate_back,
    request_navigation,
)
from frontend.session_state import sign_out


ACCOUNT_SUBFEATURES = (
    "Open New Account",
    "KYC Documents",
    "Review & Submit",
    "Track Applications",
)

ACCOUNT_TYPE_OPTIONS = (
    ("SAVINGS", "Savings Account (Synthetic Demo)"),
    ("SALARY", "Salary Account (Synthetic Demo)"),
    ("CURRENT", "Current Account (Synthetic Demo)"),
)

EMPLOYMENT_OPTIONS = (
    ("SALARIED", "Salaried"),
    ("GOVERNMENT_EMPLOYEE", "Government Employee"),
    ("SELF_EMPLOYED", "Self-Employed"),
    ("BUSINESS_OWNER", "Business Owner"),
    ("STUDENT", "Student"),
    ("RETIRED", "Retired"),
    ("OTHER", "Other"),
)

NOMINEE_RELATIONSHIP_OPTIONS = (
    ("", "None"),
    ("SPOUSE", "Spouse"),
    ("PARENT", "Parent"),
    ("CHILD", "Child"),
    ("SIBLING", "Sibling"),
    ("OTHER", "Other"),
)


def _status_pill_html(status: str) -> str:
    safe_status = html.escape(str(status or "DRAFT"))
    css_key = safe_status.lower()
    return f"<span class='acct-status-pill status-{css_key}'>{safe_status}</span>"


def _synthetic_demo_pdf_bytes(doc_type: str, ref: str) -> bytes:
    header = (
        f"%PDF-1.4\n"
        f"% VaaniShield 2.0 Synthetic Demonstration KYC Document\n"
        f"% Application Ref: {ref} | Document Type: {doc_type}\n"
        f"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n"
        f"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n"
        f"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >> endobj\n"
        f"xref\n0 4\n0000000000 65535 f \n"
        f"trailer << /Size 4 /Root 1 0 R >>\nstartxref\n190\n%%EOF\n"
    )
    return header.encode("utf-8")


def _handle_api_error(st, state, exc: APIError):
    if exc.status_code == 401:
        sign_out(state)
        st.warning("Your demo session expired. Sign in again to continue.")
        st.rerun()
    payload = exc.payload if isinstance(exc.payload, dict) else {}
    detail = payload.get("detail") if isinstance(payload, dict) else None
    if isinstance(detail, dict):
        st.error(detail.get("message") or str(exc))
        for item in detail.get("errors") or []:
            if isinstance(item, dict) and item.get("message"):
                field = item.get("field", "field")
                st.caption(f"• **{field}**: {item['message']}")
    else:
        st.error(str(exc) or "The account application request could not be completed.")


def _navigate_to(st, state, subfeature: str):
    request_navigation(state, "Account Opening Assistant", subfeature)
    st.rerun()


def _render_step_header(st, state, current_subfeature: str, current_app: dict | None):
    st.markdown(
        "<div class='section-eyebrow'>ACCOUNT SERVICES · SYNTHETIC DEMO ONBOARDING</div>",
        unsafe_allow_html=True,
    )
    prev_feature, prev_subfeature = get_previous_navigation(state)
    back_label = f"← Back to {prev_feature}" if prev_feature != "Account Opening Assistant" else f"← Back to {prev_subfeature}"

    top_back, top_left, top_right = st.columns([1.1, 2.3, 1.0])
    with top_back:
        if st.button(
            back_label,
            key="account-opening-back-btn",
            help="Return to previous section without losing application progress or draft data",
            width="stretch",
        ):
            navigate_back(state)
            st.rerun()
    with top_left:
        st.caption(
            "Synthetic demonstration account opening only. Eligibility rules, KYC uploads, "
            "and submission statuses are simulated and never open a real bank account."
        )
    with top_right:
        is_open = bool(state.get("assistant_open"))
        if st.button(
            "✕ Close Vaani AI" if is_open else "✨ Ask Vaani AI",
            key="account-open-vaani-btn",
            width="stretch",
        ):
            state["assistant_open"] = not is_open
            st.rerun()

    step_idx = ACCOUNT_SUBFEATURES.index(current_subfeature) if current_subfeature in ACCOUNT_SUBFEATURES else 0
    st.markdown(render_stepper_html(list(ACCOUNT_SUBFEATURES), step_idx), unsafe_allow_html=True)

    nav_cols = st.columns(len(ACCOUNT_SUBFEATURES))
    for idx, label in enumerate(ACCOUNT_SUBFEATURES):
        is_current = label == current_subfeature
        button_label = f"● {label}" if is_current else label
        if nav_cols[idx].button(
            button_label,
            key=f"account-step-nav-{idx}",
            type="primary" if is_current else "secondary",
            width="stretch",
        ):
            if not is_current:
                _navigate_to(st, state, label)

    if current_app:
        progress = current_app.get("progress") or {}
        pct = int(progress.get("percent") or 0)
        status = current_app.get("status", "DRAFT")
        type_label = current_app.get("account_type_label") or "Account type not selected"
        ref = current_app.get("application_ref", "—")
        with st.container(border=True):
            c1, c2, c3, c4 = st.columns([1.45, 1.45, 1.25, 1.05])
            c1.markdown(f"**Application Ref**\n\n`{ref}`")
            c2.markdown(f"**Account Type**\n\n{type_label}")
            c3.markdown(
                f"**Workflow Status**\n\n{_status_pill_html(status)}",
                unsafe_allow_html=True,
            )
            c4.markdown(f"**Completion**\n\n**{pct}%**")
            st.progress(min(max(pct / 100.0, 0.0), 1.0))


def _render_validation_checklist(st, validation: dict):
    missing_fields = validation.get("missing_fields") or []
    missing_docs = validation.get("missing_documents") or []
    errors = validation.get("errors") or []
    ready = bool(validation.get("is_valid"))
    with st.container(border=True):
        st.markdown("#### Deterministic Validation Checklist")
        if ready:
            st.success("All required fields, eligibility rules, and synthetic KYC documents are complete.")
            return
        if missing_fields:
            st.warning("Missing required fields: " + ", ".join(missing_fields))
        if missing_docs:
            st.warning("Missing required demo documents: " + ", ".join(missing_docs))
        for err in errors:
            if isinstance(err, dict) and err.get("message"):
                st.caption(f"• **{err.get('field', 'field')}**: {err['message']}")


def _render_open_new_account(st, api, state, current_app: dict | None, options_data: dict):
    st.markdown("## Open New Account")
    options = options_data.get("account_types") or []
    if options:
        cols = st.columns(len(options))
        for idx, opt in enumerate(options):
            with cols[idx]:
                with st.container(border=True):
                    st.markdown(f"#### {opt.get('label', opt.get('account_type'))}")
                    st.write(opt.get("description", ""))
                    docs = ", ".join(
                        d.get("label", d.get("document_type", ""))
                        for d in (opt.get("required_documents") or [])
                    )
                    st.caption(f"Required demo KYC: {docs}")
                    st.caption(opt.get("employment_rule", ""))

    if current_app is None:
        with st.container(border=True):
            st.markdown("#### Start a Synthetic Account Application")
            st.write(
                "Select an account type below or ask Vaani AI: "
                "*\"Vani, I want to open another savings account. Can you help me complete the process?\"*"
            )
            type_labels = {code: label for code, label in ACCOUNT_TYPE_OPTIONS}
            chosen_type = st.selectbox(
                "Account type",
                [code for code, _ in ACCOUNT_TYPE_OPTIONS],
                format_func=lambda code: type_labels.get(code, code),
                key="account-start-type-select",
            )
            if st.button("Start Application Draft", key="account-start-btn", type="primary"):
                try:
                    created = api.start_account_application(chosen_type)
                    state["account_application_cache"] = created
                    st.success(f"Draft application `{created['application_ref']}` started.")
                    st.rerun()
                except APIError as exc:
                    _handle_api_error(st, state, exc)
        return

    ref = current_app["application_ref"]
    fields = current_app.get("fields") or {}
    current_type = current_app.get("account_type") or "SAVINGS"

    with st.container(border=True):
        st.markdown("#### Authenticated Applicant Profile (Pre-filled from Demo Identity)")
        p1, p2, p3 = st.columns(3)
        p1.text_input("Full Name (Authenticated)", value=fields.get("full_name") or "", disabled=True)
        p2.text_input("Email Address", value=fields.get("email") or "", disabled=True)
        p3.text_input("Profile Reference", value=ref, disabled=True)

    with st.container(border=True):
        form_hdr_left, form_hdr_right = st.columns([3, 1.35])
        with form_hdr_left:
            st.markdown("#### Application Details Form")
            st.caption("Your progress is saved as a draft so you can resume at any time.")
        with form_hdr_right:
            if st.button(
                "Fill Demo Applicant Details",
                key="account-fill-demo-btn",
                width="stretch",
                help="Populate synthetic demo applicant details for quick demonstration",
            ):
                demo_changes = {
                    "account_type": current_type or "SAVINGS",
                    "date_of_birth": fields.get("date_of_birth") or "1996-05-14",
                    "phone": fields.get("phone") or "9876543210",
                    "employment_category": fields.get("employment_category") or "SALARIED",
                    "address_line": fields.get("address_line") or "42 MG Road, Indiranagar",
                    "city": fields.get("city") or "Bengaluru",
                    "state": fields.get("state") or "Karnataka",
                    "postal_code": fields.get("postal_code") or "560038",
                    "nominee_name": fields.get("nominee_name") or "Meera Sharma",
                    "nominee_relationship": fields.get("nominee_relationship") or "SPOUSE",
                }
                try:
                    updated = api.update_account_application(ref, demo_changes)
                    state["account_application_cache"] = updated
                    st.success("Synthetic demo applicant details saved to draft.")
                    st.rerun()
                except APIError as exc:
                    _handle_api_error(st, state, exc)

        type_codes = [code for code, _ in ACCOUNT_TYPE_OPTIONS]
        type_labels = {code: label for code, label in ACCOUNT_TYPE_OPTIONS}
        type_index = type_codes.index(current_type) if current_type in type_codes else 0

        emp_codes = [code for code, _ in EMPLOYMENT_OPTIONS]
        emp_labels = {code: label for code, label in EMPLOYMENT_OPTIONS}
        current_emp = fields.get("employment_category")
        emp_index = emp_codes.index(current_emp) if current_emp in emp_codes else 0

        rel_codes = [code for code, _ in NOMINEE_RELATIONSHIP_OPTIONS]
        rel_labels = {code: label for code, label in NOMINEE_RELATIONSHIP_OPTIONS}
        current_rel = fields.get("nominee_relationship") or ""
        rel_index = rel_codes.index(current_rel) if current_rel in rel_codes else 0

        with st.form("account_application_details_form"):
            selected_type = st.selectbox(
                "Account Type",
                type_codes,
                index=type_index,
                format_func=lambda code: type_labels.get(code, code),
            )
            f1, f2, f3 = st.columns(3)
            with f1:
                dob = st.text_input(
                    "Date of Birth (YYYY-MM-DD)",
                    value=fields.get("date_of_birth") or "",
                    placeholder="1996-05-14",
                )
            with f2:
                phone = st.text_input(
                    "Indian Mobile Number (10 digits)",
                    value=fields.get("phone") or "",
                    placeholder="9876543210",
                )
            with f3:
                employment = st.selectbox(
                    "Employment Category",
                    emp_codes,
                    index=emp_index,
                    format_func=lambda code: emp_labels.get(code, code),
                )

            address_line = st.text_input(
                "Residential Address Line",
                value=fields.get("address_line") or "",
                placeholder="42 MG Road, Indiranagar",
            )
            c1, c2, c3 = st.columns(3)
            with c1:
                city = st.text_input("City", value=fields.get("city") or "", placeholder="Bengaluru")
            with c2:
                state_name = st.text_input("State", value=fields.get("state") or "", placeholder="Karnataka")
            with c3:
                postal_code = st.text_input(
                    "6-Digit PIN Code",
                    value=fields.get("postal_code") or "",
                    placeholder="560038",
                )

            n1, n2 = st.columns(2)
            with n1:
                nominee_name = st.text_input(
                    "Nominee Name (Optional)",
                    value=fields.get("nominee_name") or "",
                    placeholder="Meera Sharma",
                )
            with n2:
                nominee_relationship = st.selectbox(
                    "Nominee Relationship (Optional)",
                    rel_codes,
                    index=rel_index,
                    format_func=lambda code: rel_labels.get(code, code),
                )

            save_col, next_col = st.columns(2)
            save_clicked = save_col.form_submit_button("Save Draft Progress", type="primary", width="stretch")
            save_and_kyc = next_col.form_submit_button("Save & Continue to KYC Documents →", width="stretch")

        if save_clicked or save_and_kyc:
            changes = {
                "account_type": selected_type,
                "employment_category": employment,
            }
            if dob.strip():
                changes["date_of_birth"] = dob.strip()
            if phone.strip():
                changes["phone"] = phone.strip()
            if address_line.strip():
                changes["address_line"] = address_line.strip()
            if city.strip():
                changes["city"] = city.strip()
            if state_name.strip():
                changes["state"] = state_name.strip()
            if postal_code.strip():
                changes["postal_code"] = postal_code.strip()
            if nominee_name.strip():
                changes["nominee_name"] = nominee_name.strip()
            elif fields.get("nominee_name"):
                changes["nominee_name"] = None
            if nominee_relationship:
                changes["nominee_relationship"] = nominee_relationship
            elif fields.get("nominee_relationship"):
                changes["nominee_relationship"] = None

            try:
                updated = api.update_account_application(ref, changes)
                state["account_application_cache"] = updated
                st.success("Application draft saved.")
                if save_and_kyc:
                    _navigate_to(st, state, "KYC Documents")
                else:
                    st.rerun()
            except APIError as exc:
                _handle_api_error(st, state, exc)

    _render_validation_checklist(st, current_app.get("validation") or {})


def _render_kyc_documents(st, api, state, current_app: dict | None):
    st.markdown("## KYC Demonstration Documents")
    if current_app is None:
        st.info("No active account application draft found. Start an application in Open New Account first.")
        if st.button("Go to Open New Account", key="kyc-go-open-btn", type="primary"):
            _navigate_to(st, state, "Open New Account")
        return

    ref = current_app["application_ref"]
    if not current_app.get("account_type"):
        st.warning("Select an account type in Open New Account before uploading demonstration documents.")
        if st.button("Go to Open New Account", key="kyc-select-type-btn", type="primary"):
            _navigate_to(st, state, "Open New Account")
        return

    st.info(
        "Upload synthetic demonstration files only (PDF, PNG, or JPEG up to 2 MB). "
        "Files are validated by magic bytes, stored under random server identifiers, "
        "never sent to any language model, and never verified against any government registry."
    )

    documents = [d for d in (current_app.get("documents") or []) if d.get("required") or d.get("uploaded")]
    for doc in documents:
        doc_type = doc["document_type"]
        label = doc.get("label", doc_type)
        uploaded = bool(doc.get("uploaded"))
        with st.container(border=True):
            header_left, header_right = st.columns([3, 1.2])
            with header_left:
                badge_html = (
                    "<span class='acct-kyc-badge uploaded'>✓ Uploaded</span>"
                    if uploaded
                    else "<span class='acct-kyc-badge required'>⏳ Required</span>"
                )
                st.markdown(
                    f"#### {html.escape(str(label))} &nbsp; {badge_html}",
                    unsafe_allow_html=True,
                )
                st.caption("Synthetic demonstration KYC document only; not verified with any external authority.")
            with header_right:
                if uploaded:
                    st.caption(f"Type: `{doc.get('media_type')}`\n\nSize: {doc.get('size_bytes')} bytes")
                    if st.button("Remove Document", key=f"kyc-delete-{doc_type}", width="stretch"):
                        try:
                            updated = api.delete_account_application_document(ref, doc_type)
                            state["account_application_cache"] = updated
                            st.info(f"Removed {label}.")
                            st.rerun()
                        except APIError as exc:
                            _handle_api_error(st, state, exc)
                else:
                    if st.button(
                        f"Use Synthetic {doc_type} Sample",
                        key=f"kyc-sample-btn-{doc_type}",
                        width="stretch",
                        help="Generate and upload a valid synthetic demonstration PDF for this KYC slot",
                    ):
                        try:
                            sample_bytes = _synthetic_demo_pdf_bytes(doc_type, ref)
                            updated = api.upload_account_application_document(
                                ref, doc_type, sample_bytes, "application/pdf",
                            )
                            state["account_application_cache"] = updated
                            st.success(f"Synthetic {label} sample uploaded. It has not been verified.")
                            st.rerun()
                        except APIError as exc:
                            _handle_api_error(st, state, exc)

            file_obj = st.file_uploader(
                f"Select {label} file from computer (PDF, PNG, JPEG · Max 2 MB)",
                type=["pdf", "png", "jpg", "jpeg"],
                key=f"kyc-uploader-{ref}-{doc_type}",
            )
            if file_obj is not None:
                if st.button(f"Upload Selected {label}", key=f"kyc-upload-btn-{doc_type}", type="primary"):
                    try:
                        raw = file_obj.getvalue()
                        media_type = getattr(file_obj, "type", None) or "application/octet-stream"
                        updated = api.upload_account_application_document(
                            ref, doc_type, raw, media_type,
                        )
                        state["account_application_cache"] = updated
                        st.success(f"{label} uploaded. It has not been verified.")
                        st.rerun()
                    except APIError as exc:
                        _handle_api_error(st, state, exc)

    _render_validation_checklist(st, current_app.get("validation") or {})
    nav_back, nav_next = st.columns(2)
    if nav_back.button("← Back to Open New Account", key="kyc-back-btn", width="stretch"):
        _navigate_to(st, state, "Open New Account")
    if nav_next.button("Continue to Review & Submit →", key="kyc-next-btn", type="primary", width="stretch"):
        _navigate_to(st, state, "Review & Submit")


def _render_review_and_submit(st, api, state, current_app: dict | None):
    st.markdown("## Review & Confirm Submission")
    last_submitted = state.get("account_application_last_submitted")
    if isinstance(last_submitted, dict) and last_submitted.get("status") == "SUBMITTED":
        with st.container(border=True):
            st.success(
                f"Application `{last_submitted['application_ref']}` has been submitted "
                f"at {last_submitted.get('submitted_at')}."
            )
            st.caption(last_submitted.get("status_note", ""))
            if st.button("Go to Track Applications", key="review-go-track-btn", type="primary"):
                state["account_application_last_submitted"] = None
                _navigate_to(st, state, "Track Applications")

    if current_app is None:
        st.info("No active draft application is waiting for review.")
        if st.button("Go to Track Applications", key="review-empty-track-btn"):
            _navigate_to(st, state, "Track Applications")
        return

    ref = current_app["application_ref"]
    fields = current_app.get("fields") or {}
    validation = current_app.get("validation") or {}
    docs = current_app.get("documents") or []
    status = current_app.get("status", "DRAFT")
    uploaded_labels = [d.get("label", d["document_type"]) for d in docs if d.get("uploaded")]
    city_state_pin = ", ".join(
        part for part in (fields.get("city"), fields.get("state"), fields.get("postal_code")) if part
    ) or "—"
    nominee_display = (
        f"{fields.get('nominee_name')} ({fields.get('nominee_relationship') or 'Unspecified'})"
        if fields.get("nominee_name")
        else "None specified"
    )
    kyc_display = ", ".join(uploaded_labels) if uploaded_labels else "None uploaded"

    summary_items = [
        ("Application Reference", ref),
        ("Account Product", current_app.get("account_type_label") or "Not selected"),
        ("Applicant Name", fields.get("full_name") or "—"),
        ("Email Address", fields.get("email") or "—"),
        ("Mobile Number", fields.get("phone") or "—"),
        ("Date of Birth", fields.get("date_of_birth") or "—"),
        ("Employment Category", fields.get("employment_category") or "—"),
        ("Residential Address", fields.get("address_line") or "—"),
        ("City, State & PIN", city_state_pin),
        ("Nominee Details", nominee_display),
        ("Uploaded Demo KYC (not verified)", kyc_display),
        ("Workflow Status", status),
    ]
    grid_cells = "".join(
        f"<div class='acct-summary-item'>"
        f"<span class='acct-label'>{html.escape(str(lbl))}</span>"
        f"<span class='acct-value'>{html.escape(str(val))}</span>"
        f"</div>"
        for lbl, val in summary_items
    )

    with st.container(border=True):
        hdr_left, hdr_right = st.columns([3, 1])
        with hdr_left:
            st.markdown("#### Application Review Summary")
            st.caption("Verify all applicant details and synthetic KYC records before final confirmation.")
        with hdr_right:
            st.markdown(
                f"<div style='text-align:right;margin-top:.35rem;'>{_status_pill_html(status)}</div>",
                unsafe_allow_html=True,
            )
        st.markdown(f"<div class='acct-summary-grid'>{grid_cells}</div>", unsafe_allow_html=True)

    _render_validation_checklist(st, validation)

    if not validation.get("is_valid"):
        st.warning("Complete all required form fields and upload the required demo KYC documents before preparing review.")
        b1, b2 = st.columns(2)
        if b1.button("Go to Open New Account", key="review-fix-details-btn", width="stretch"):
            _navigate_to(st, state, "Open New Account")
        if b2.button("Go to KYC Documents", key="review-fix-kyc-btn", width="stretch"):
            _navigate_to(st, state, "KYC Documents")
        return

    review_digest = current_app.get("review_digest") or state.get("account_application_review_digest")

    if status != "PENDING_CONFIRMATION" or not review_digest:
        with st.container(border=True):
            st.markdown("#### Step 1: Lock Application for Final Review")
            st.write(
                "Preparing review verifies all fields and generates a cryptographic review digest "
                "bound to the exact application state above."
            )
            if st.button("Prepare Application for Confirmation", key="review-prepare-btn", type="primary"):
                try:
                    prepared = api.prepare_account_application_review(ref)
                    state["account_application_cache"] = prepared.get("application")
                    state["account_application_review_digest"] = prepared.get("review_digest")
                    st.rerun()
                except APIError as exc:
                    _handle_api_error(st, state, exc)
    else:
        with st.container(border=True):
            st.markdown("#### Step 2: Explicit Customer Confirmation Required")
            st.warning(
                "Review the summary carefully. Vaani AI cannot submit this application for you. "
                "Pressing **Confirm & Submit Application** below records your explicit confirmation."
            )
            c_submit, c_back = st.columns(2)
            if c_submit.button(
                "Confirm & Submit Application",
                key="review-confirm-submit-btn",
                type="primary",
                width="stretch",
            ):
                try:
                    submitted_app = api.submit_account_application(
                        ref, confirmed=True, review_digest=review_digest,
                    )
                    state["account_application_last_submitted"] = submitted_app
                    state["account_chat_submitted_ref"] = submitted_app.get("application_ref")
                    state["account_application_review_target_ref"] = None
                    state["account_application_cache"] = None
                    state["account_application_review_digest"] = None
                    state["account_application_selected_ref"] = submitted_app.get("application_ref")
                    st.success(f"Application `{submitted_app['application_ref']}` submitted!")
                    st.rerun()
                except APIError as exc:
                    _handle_api_error(st, state, exc)
            if c_back.button("Return to Draft to Make Changes", key="review-return-draft-btn", width="stretch"):
                try:
                    draft_app = api.return_account_application_to_draft(ref)
                    state["account_application_cache"] = draft_app
                    state["account_application_review_digest"] = None
                    st.info("Application returned to DRAFT status.")
                    st.rerun()
                except APIError as exc:
                    _handle_api_error(st, state, exc)


def _render_track_applications(st, api, state, current_app: dict | None):
    st.markdown("## Track Account Applications")
    try:
        listing = api.list_account_applications(limit=20)
    except APIError as exc:
        _handle_api_error(st, state, exc)
        return

    items = listing.get("items") or []
    if not items:
        st.info("No account applications have been created for this demo profile yet.")
        if st.button("Go to Open New Account", key="track-start-new-btn", type="primary"):
            _navigate_to(st, state, "Open New Account")
        return

    refs = [item["application_ref"] for item in items]
    selected_ref = state.get("account_application_selected_ref")
    if selected_ref not in refs:
        selected_ref = refs[0]
        state["account_application_selected_ref"] = selected_ref

    for item in items:
        ref = item["application_ref"]
        status = item.get("status", "DRAFT")
        type_label = item.get("account_type_label") or "Account type not chosen"
        progress = (item.get("progress") or {}).get("percent", 0)
        with st.container(border=True):
            c1, c2, c3, c4 = st.columns([1.6, 1.5, 1.3, 1.2])
            c1.markdown(f"**`{ref}`**\n\n{type_label}")
            c2.markdown(
                f"**Status**\n\n{_status_pill_html(status)} &nbsp; **{progress}%**",
                unsafe_allow_html=True,
            )
            c3.caption(f"Created: {item.get('created_at')}\n\nSubmitted: {item.get('submitted_at') or 'Not submitted'}")
            with c4:
                if status in {"DRAFT", "PENDING_CONFIRMATION"}:
                    target = "Review & Submit" if status == "PENDING_CONFIRMATION" else "Open New Account"
                    if st.button(f"Go to {target}", key=f"track-resume-{ref}", type="primary", width="stretch"):
                        state["account_application_cache"] = item
                        _navigate_to(st, state, target)
                if st.button("View Audit Timeline", key=f"track-detail-{ref}", width="stretch"):
                    state["account_application_selected_ref"] = ref
                    st.rerun()

    if selected_ref:
        try:
            detail = api.get_account_application_status(application_ref=selected_ref)
        except APIError as exc:
            _handle_api_error(st, state, exc)
            return
        with st.container(border=True):
            st.markdown(f"#### Application Audit Timeline · `{selected_ref}`")
            st.caption(detail.get("synthetic_notice", ""))
            events = detail.get("events") or []
            if not events:
                st.caption("No events recorded.")
            for event in events:
                st.markdown(
                    f"**{event.get('occurred_at')} · `{event.get('event_type')}`** — {event.get('summary')}"
                )


def render(st, subfeature: str, api, state):
    try:
        current_payload = api.get_current_account_application()
        current_app = current_payload.get("application") if isinstance(current_payload, dict) else None
        state["account_application_cache"] = current_app
    except APIError as exc:
        _handle_api_error(st, state, exc)
        return

    review_target = state.get("account_application_review_target_ref")
    if subfeature == "Review & Submit" and review_target and (
        not current_app or current_app.get("application_ref") != review_target
    ):
        st.warning("The requested application is no longer the active draft. Check Track Applications before continuing.")
        return
    _render_step_header(st, state, subfeature, current_app)

    if subfeature == "Open New Account":
        try:
            options_data = api.get_account_application_options()
        except APIError as exc:
            _handle_api_error(st, state, exc)
            return
        _render_open_new_account(st, api, state, current_app, options_data)
    elif subfeature == "KYC Documents":
        _render_kyc_documents(st, api, state, current_app)
    elif subfeature == "Review & Submit":
        _render_review_and_submit(st, api, state, current_app)
    else:
        _render_track_applications(st, api, state, current_app)
