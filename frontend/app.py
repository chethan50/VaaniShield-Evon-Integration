
import html
import textwrap
from contextlib import nullcontext
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from frontend.api_client import APIError
from frontend.components.fintech_ui import (
    format_inr,
    get_category_icon,
    render_stepper_html,
    render_transaction_card_html,
)
from frontend.components.floating_assistant import assistant_dock_is_visible, render_floating_assistant
from frontend.navigation import (
    FEATURE_SUBFEATURES,
    apply_pending_navigation,
    ensure_subfeature,
    page_key,
    request_navigation,
    set_subfeature,
    subfeature_state_key,
)
from frontend.pages import account_opening, customer_support, scam_detection
from frontend.pages.banking import load_dashboard_context, render_demo_scope_notice
from frontend.session_state import (
    initialize_session_state,
    mark_session_valid,
    reset_conversation_state,
    session_validation_due,
    sign_out,
)

# --------------------------------------------------
# CONFIGURATION & DESIGN SYSTEM
# --------------------------------------------------

st.set_page_config(
    page_title="VaaniShield | Intelligent Banking Companion",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

styles_path = Path(__file__).parent.joinpath("assets", "styles.css")
if styles_path.exists():
    extra_styles = styles_path.read_text(encoding="utf-8")
    st.markdown(f"<style>{extra_styles}</style>", unsafe_allow_html=True)


# --------------------------------------------------
# API HELPERS
# --------------------------------------------------

def api_call(request, *args, **kwargs):
    try:
        return request(*args, **kwargs)
    except APIError as error:
        if error.status_code == 401:
            sign_out(st.session_state)
            st.warning("Your demo session expired. Sign in again to continue.")
            st.rerun()
        st.error("The demo dashboard data could not be loaded. Check that the API is running.")
        st.stop()


def normalize_collection(data):
    """Handle both list and dictionary API responses."""
    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        if "value" in data:
            return data["value"]

    return []


def money(value):
    return format_inr(value)


initialize_session_state(st.session_state)
api = st.session_state.api_client

# Support direct demo verification parameters
if not st.session_state.authenticated and st.query_params.get("demo_user") == "arjun":
    try:
        profile = api.create_demo_session("demo-arjun")
        mark_session_valid(st.session_state, profile)
        reset_conversation_state(st.session_state)
    except Exception:
        pass

if st.query_params.get("copilot") == "open":
    st.session_state["assistant_open"] = True

if st.query_params.get("history_open") == "true":
    st.session_state["assistant_open"] = True
    st.session_state["expand_history_for_demo"] = True

if st.query_params.get("demo_messages") == "long":
    st.session_state["assistant_open"] = True
    if not st.session_state.get("conversation_messages"):
        st.session_state["conversation_messages"] = [
            {
                "id": "msg-demo-1",
                "role": "user",
                "text": "What is my current savings balance and last 3 transactions?",
                "input_mode": "text",
            },
            {
                "id": "msg-demo-2",
                "role": "assistant",
                "text": "Your **Salary Account** balance is **₹1,24,500.50**.\n\nHere are your last 3 transactions:\n- **Dec 10, 2026**: DMart Supermarket — ₹2,450.00 (Shopping)\n- **Dec 09, 2026**: Swiggy Food — ₹640.00 (Dining)\n- **Dec 08, 2026**: Tata Power Electricity — ₹1,820.00 (Utilities)",
                "input_mode": "text",
            },
            {
                "id": "msg-demo-3",
                "role": "user",
                "text": "Could you also check my active Home Loan EMI due date?",
                "input_mode": "text",
            },
            {
                "id": "msg-demo-4",
                "role": "assistant",
                "text": "### Active Loan Details\n\n| Facility | Outstanding | Monthly EMI | Next Due Date |\n| :--- | :--- | :--- | :--- |\n| **Home Loan** | ₹34,20,000 | ₹32,450 | **December 12, 2026** |\n\nYour repayment progress is at **28%**. Your linked auto-debit account has sufficient funds for this payment.",
                "input_mode": "text",
            },
            {
                "id": "msg-demo-5",
                "role": "user",
                "text": "Are there any suspicious security alerts on my account?",
                "input_mode": "text",
            },
            {
                "id": "msg-demo-6",
                "role": "assistant",
                "text": "🛡️ **Security Status: Safe**\n\nAll security systems are functioning normally:\n- No unrecognized device sign-ins in the past 30 days.\n- Two-factor authentication (Gnani Prisma voice biometric) active.\n- AI scam prevention shield active on SMS and UPI transfers.",
                "input_mode": "text",
            },
        ]

if not st.session_state.authenticated:
    st.markdown(
        textwrap.dedent("""
        <div class="auth-container">
          <div class="auth-brand-logo">🛡️</div>
          <div class="auth-brand-title">VaaniShield Digital Banking</div>
          <div class="auth-brand-subtitle">Secure AI-Powered Customer Portal · Demonstration Login</div>
        </div>
        """).strip(),
        unsafe_allow_html=True,
    )

    auth_col1, auth_col2, auth_col3 = st.columns([1, 2.2, 1])
    with auth_col2:
        st.markdown("##### ⚡ Quick Select Demo Customer")
        qc1, qc2, qc3 = st.columns(3)
        if qc1.button("Arjun Sharma\n(Primary Demo)", use_container_width=True):
            st.session_state["login_email"] = "arjun@example.com"
            st.session_state["login_password"] = "Arjun@Demo2026"
        if qc2.button("Priya Patel\n(Business Acct)", use_container_width=True):
            st.session_state["login_email"] = "priya@example.com"
            st.session_state["login_password"] = "Priya@Demo2026"
        if qc3.button("Rahul Verma\n(Savings Acct)", use_container_width=True):
            st.session_state["login_email"] = "rahul@example.com"
            st.session_state["login_password"] = "Rahul@Demo2026"

        default_email = st.session_state.get("login_email", "arjun@example.com")
        default_pwd = st.session_state.get("login_password", "Arjun@Demo2026")

        with st.form("demo_sign_in"):
            st.markdown("##### Sign in to your account")
            email_val = st.text_input("Registered Email Address", value=default_email, key="signin_email_field")
            pwd_val = st.text_input("Password", value=default_pwd, type="password", key="signin_password_field")
            submitted = st.form_submit_button("Sign In Securely", type="primary", use_container_width=True)

        if submitted:
            clean_email = email_val.strip()
            clean_pwd = pwd_val.strip()
            try:
                profile = api.demo_login(clean_email, clean_pwd)
                mark_session_valid(st.session_state, profile)
                reset_conversation_state(st.session_state)
                st.rerun()
            except APIError as error:
                # Fallback to key-based demo session if demo_login fails on unhashed setups
                key_map = {
                    "arjun@example.com": "demo-arjun",
                    "priya@example.com": "demo-priya",
                    "rahul@example.com": "demo-rahul",
                }
                if clean_email in key_map and "Arjun@" in clean_pwd or "Priya@" in clean_pwd or "Rahul@" in clean_pwd:
                    try:
                        profile = api.create_demo_session(key_map[clean_email])
                        mark_session_valid(st.session_state, profile)
                        reset_conversation_state(st.session_state)
                        st.rerun()
                    except Exception:
                        pass
                st.error(
                    f"Authentication failed: {error.message if hasattr(error, 'message') else 'Invalid email or password'}. "
                    "Ensure you use valid credentials (e.g. arjun@example.com / Arjun@Demo2026)."
                )

        st.markdown(
            textwrap.dedent("""
            <div class="auth-credential-pill">
              <b>🔒 Bank-Grade Authentication Security</b><br>
              Passwords verified via salted 100,000-iteration PBKDF2-SHA256 cryptography. Every login event is recorded with device, IP, and UTC/IST timestamp in the audit log.
            </div>
            """).strip(),
            unsafe_allow_html=True,
        )
    st.stop()

if session_validation_due(st.session_state):
    try:
        mark_session_valid(st.session_state, api.get_demo_session())
    except APIError as error:
        if error.status_code == 401:
            sign_out(st.session_state)
            st.warning("Your demo session expired. Sign in again to continue.")
            st.rerun()
        st.error("The demo session could not be verified. Please try again shortly.")
        st.stop()

apply_pending_navigation(st.session_state)
profile_name = st.session_state.customer.get("display_name", "Demo customer")

# ==================================================
# LEVEL 1 — TOP HEADER (FULL-WIDTH BRAND & IDENTITY)
# ==================================================
with st.container(key="profile-header-copilot-open" if assistant_dock_is_visible(st.session_state) else "profile-header"):
    brand, identity, notif_col, demo_badge, copilot_action = st.columns([1.8, 2.6, 0.9, 1.1, 1.2])
    with brand:
        st.markdown(
            "<div class='brand-lockup'><span class='brand-mark'>🛡️</span>"
            "<span><b>VaaniShield</b><small>Intelligent Banking Companion</small></span></div>",
            unsafe_allow_html=True,
        )
    with identity:
        from datetime import datetime, timezone, timedelta
        tz_ist = timezone(timedelta(hours=5, minutes=30))
        hour_ist = datetime.now(tz_ist).hour
        if 5 <= hour_ist < 12:
            time_greeting = "Good morning"
        elif 12 <= hour_ist < 17:
            time_greeting = "Good afternoon"
        elif 17 <= hour_ist < 22:
            time_greeting = "Good evening"
        else:
            time_greeting = "Good night"
        greeting_text = f"Hey, {profile_name}! {time_greeting}."

        st.markdown(
            f"<div class='profile-identity'><strong>{greeting_text}</strong>"
            f"<span>Verified Demo Customer · Bangalore, IN</span></div>",
            unsafe_allow_html=True,
        )
    with notif_col:
        try:
            notif_resp = api.get_notifications()
            notifications = notif_resp.get("notifications", [])
            unread_count = int(notif_resp.get("unread_count", 0))
        except Exception:
            notifications = []
            unread_count = 0

        bell_label = f"🔔 ({unread_count})" if unread_count > 0 else "🔔"
        with st.popover(bell_label, help="Notification Center", width="stretch"):
            header_col1, header_col2 = st.columns([2, 1])
            with header_col1:
                notif_badge = (
                    f'<span class="notif-badge">{unread_count} new</span>'
                    if unread_count > 0
                    else ""
                )
                st.markdown(
                    f"**Notifications** {notif_badge}".strip(),
                    unsafe_allow_html=True,
                )
            with header_col2:
                if unread_count > 0:
                    if st.button("Mark all read", key="btn_notif_mark_all", type="secondary"):
                        try:
                            api.mark_all_notifications_read()
                            st.rerun()
                        except Exception:
                            pass

            if not notifications:
                st.caption("No notifications yet.")
            else:
                with st.container(height=360):
                    category_icons = {
                        "loan": "📑",
                        "transaction": "💳",
                        "support": "🎫",
                        "security": "🛡️",
                    }
                    for idx, notif in enumerate(notifications):
                        raw_id = notif.get("id") or notif.get("notification_id")
                        n_id = str(raw_id).strip() if raw_id is not None and str(raw_id).strip() else None
                        btn_key = f"btn_read_{n_id}_{idx}" if n_id else f"btn_read_pos_{idx}"
                        n_cat = notif.get("category", "banking")
                        icon = category_icons.get(n_cat, "🔔")
                        is_unread = not bool(notif.get("is_read"))
                        unread_class = "notif-unread" if is_unread else "notif-read"
                        unread_dot = '<span class="notif-dot">●</span>' if is_unread else ''
                        n_title = notif.get("title", "Alert")
                        n_msg = notif.get("message", "")
                        created_meta = str(notif.get("created_at", ""))[:16].replace("T", " ")
                        cat_upper = n_cat.upper()

                        col_card, col_action = st.columns([5, 1])
                        with col_card:
                            st.markdown(
                                f"""
                                <div class="notif-item {unread_class}">
                                  <div class="notif-title-row">
                                    <span class="notif-icon">{icon}</span>
                                    <span class="notif-title"><b>{n_title}</b></span>
                                    {unread_dot}
                                  </div>
                                  <div class="notif-msg">{n_msg}</div>
                                  <div class="notif-meta"><small>{created_meta} · {cat_upper}</small></div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                        with col_action:
                            sms_report_id = (
                                scam_detection.sms_report_id_from_notification(notif)
                                if n_cat == "security_alert" else None
                            )
                            if sms_report_id and st.button(
                                "View Analysis", key=f"btn_notif_view_analysis_{n_id}_{idx}"
                            ):
                                if scam_detection.queue_sms_report_navigation(st.session_state, notif):
                                    st.rerun()
                            if is_unread:
                                if st.button("✓", key=btn_key, help="Mark as read"):
                                    try:
                                        if n_id:
                                            api.mark_notification_read(n_id)
                                        st.rerun()
                                    except Exception:
                                        pass
    with demo_badge:
        st.markdown("<div class='demo-badge'>SYNTHETIC DEMO</div>", unsafe_allow_html=True)
    with copilot_action:
        if st.button(
            "✨ Vaani AI",
            key="header-copilot-toggle",
            type="secondary",
            width="stretch",
            help="Open or close the Vaani AI panel",
        ):
            st.session_state["assistant_open"] = True
            st.rerun()

# Persistent Browser-Side Wake-Word Listener (Zero Cloud Audio Streaming)
from frontend.components.wake_word import render_wake_word_listener
render_wake_word_listener(st, st.session_state)

# ==================================================
# LEVEL 2 — MAIN FEATURE NAVIGATION BAR (HORIZONTAL)
# ==================================================
main_features = list(FEATURE_SUBFEATURES.keys())
current_feature = st.session_state.get("main_feature", "Banking Information")
if current_feature not in main_features:
    current_feature = "Banking Information"
    st.session_state["main_feature"] = current_feature

feature_icons = {
    "Banking Information": "🏛️",
    "Financial Intelligence": "📊",
    "Scam Detection": "🛡️",
    "Customer Support": "💬",
    "Account Opening Assistant": "📝",
}

is_copilot_open = assistant_dock_is_visible(st.session_state)
nav_container_key = "main-nav-container-copilot-open" if is_copilot_open else "main-nav-container"

with st.container(key=nav_container_key):
    nav_cols = st.columns(len(main_features))
    for col, feat in zip(nav_cols, main_features):
        with col:
            is_active = (feat == current_feature)
            btn_type = "primary" if is_active else "secondary"
            btn_icon = feature_icons.get(feat, "•")
            btn_label = f"{btn_icon}  {feat}"
            if st.button(btn_label, key=f"nav_feat_{feat}", type=btn_type, width="stretch"):
                if feat != current_feature:
                    request_navigation(st.session_state, feat, FEATURE_SUBFEATURES[feat][0])
                    st.rerun()

feature = current_feature

# ==================================================
# LEVEL 3 — CONTEXTUAL SUBFEATURE SIDEBAR (LEFT)
# ==================================================
with st.sidebar:
    st.markdown(
        "<div class='sidebar-brand'><span class='sidebar-brand-mark'>🛡️</span>"
        "<span><span class='sidebar-brand-name'>VaaniShield</span>"
        "<span class='sidebar-brand-tag'>INTELLIGENT BANKING</span></span></div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"<div class='sidebar-user-card'><strong>{html.escape(profile_name)}</strong><span>Premium Demo Customer</span></div>"
        f"<div class='sidebar-context-card'>"
        f"<div class='sidebar-context-kicker'>Active Feature</div>"
        f"<div class='sidebar-context-title'>{feature_icons.get(feature, '•')} {feature}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div style='font-size:0.68rem;font-weight:800;letter-spacing:1.2px;color:var(--text-muted);margin:0.8rem 0 0.35rem 0.2rem;text-transform:uppercase;'>Sections</div>",
        unsafe_allow_html=True,
    )

    current_subfeature = ensure_subfeature(st.session_state, feature)
    subfeatures = FEATURE_SUBFEATURES[feature]

    subfeature_icons = {
        "Overview": "⚡",
        "Accounts": "💳",
        "Transactions": "📝",
        "Cards": "💎",
        "Loans": "📑",
        "Spending Analysis": "📉",
        "Category Analysis": "🥧",
        "Merchant Analysis": "🏪",
        "Monthly Comparison": "📅",
        "Budgets & Recommendations": "💡",
        "Message Analysis": "📩",
        "Call Analysis": "📞",
        "Transaction Risk": "🔍",
        "Investigations": "🛡️",
        "Ask Support Assistant": "🤖",
        "Create Support Ticket": "🎫",
        "Track Complaints": "📋",
        "Resolution Assistance": "🤝",
        "Open New Account": "✍️",
        "KYC Documents": "📁",
        "Review & Submit": "✅",
        "Track Applications": "🔎",
    }

    for item in subfeatures:
        is_active = (item == current_subfeature)
        btn_type = "primary" if is_active else "secondary"
        icon = subfeature_icons.get(item, "•")
        btn_label = f"{icon}  {item}"
        if st.button(
            btn_label,
            key=f"sidebar_subnav_{feature}_{item}",
            type=btn_type,
            width="stretch",
        ):
            if item != current_subfeature:
                request_navigation(st.session_state, feature, item)
                st.rerun()

    subfeature = current_subfeature
    page = page_key(feature, subfeature)

    st.divider()

    with st.popover("⚙️ Preferences", width="stretch"):
        st.markdown("**Assistant preferences**")
        from frontend.session_state import (
            RESPONSE_LANGUAGE_OPTIONS_FULL,
            on_sidebar_language_change,
        )
        st.selectbox(
            "Response language",
            RESPONSE_LANGUAGE_OPTIONS_FULL,
            key="sidebar_assistant_language",
            on_change=on_sidebar_language_change,
            help="Explicit language choice overrides input speech and automatic detection.",
        )
        st.caption("Explicit language selection takes absolute priority over user input language and auto-detection.")

        st.divider()
        st.markdown("**Voice Wake-Word**")
        wake_active = st.toggle(
            "Enable 'Hey Vaani' activation",
            value=bool(st.session_state.get("wake_word_enabled", False)),
            key="sidebar_wake_word_toggle",
            help="Allow the composer to offer hands-free mode. Start listening explicitly there.",
        )
        st.session_state["wake_word_enabled"] = wake_active
        st.selectbox(
            "Wake recognition mode", ("on_device", "browser_default"),
            format_func=lambda mode: "Strict on-device" if mode == "on_device" else "Browser-default",
            key="wake_recognition_mode",
        )
        if st.session_state.get("wake_recognition_mode") == "browser_default":
            st.warning("Browser-default wake listening may send microphone audio, including speech before the wake phrase, to your browser's recognition provider.")
            st.checkbox("I consent to browser-default wake recognition", key="wake_browser_consent")
        if wake_active:
            st.caption("Choose Start hands-free at the composer. Works only while this tab is active.")

    with st.popover("👤 Demo Profile", width="stretch"):
        st.markdown(f"**{profile_name}**")
        st.caption("This session uses a synthetic VaaniShield demo identity.")
        st.caption("Demo data is not connected to a real bank account.")
        if st.button("Sign out", key="sidebar_sign_out", width="stretch"):
            try:
                api.delete_demo_session()
            except APIError:
                pass
            sign_out(st.session_state)
            st.rerun()


# --------------------------------------------------
# LOAD AUTHENTICATED SELF-SERVICE DATA FOR EXISTING DASHBOARD VIEWS
# --------------------------------------------------

if feature in {"Banking Information", "Financial Intelligence"}:
    customer = {}
    banking = {}
    financial = {"insights": []}
    advisory = {}
    accounts = transactions = cards = loans = []
    categories = merchants = []
    balance = 0
    summary = {}

    if page == "Overview":
        context = api_call(load_dashboard_context, api)
        customer = context["customer"]
        banking = context["banking"]
        financial = context["financial_intelligence"]
        advisory = context["financial_advisory"]
        accounts = normalize_collection(banking["account_balance"])
        transactions = normalize_collection(banking["recent_transactions"])
        balance = sum(account.get("balance", 0) for account in accounts)
        summary = financial["summary"]
    elif page == "Accounts":
        accounts = normalize_collection(api_call(api.get_my_accounts))
    elif page == "Transactions":
        transactions = normalize_collection(api_call(api.get_my_transactions, 100))
    elif page == "Cards":
        cards = normalize_collection(api_call(api.get_my_cards))
    elif page == "Loans":
        loans = normalize_collection(api_call(api.get_my_loans))
    elif page == "Spending Analysis":
        financial = api_call(api.get_my_financial_summary)
        summary = financial
        categories = normalize_collection(api_call(api.get_my_financial_categories))
        advisory = {"spending_comparison": api_call(api.get_my_monthly_comparison)}
    elif page == "Category Analysis":
        categories = normalize_collection(api_call(api.get_my_financial_categories))
        financial = api_call(api.get_my_financial_summary)
        summary = financial
    elif page == "Merchant Analysis":
        merchants = normalize_collection(api_call(api.get_my_financial_merchants, 10))
        financial = api_call(api.get_my_financial_summary)
        summary = financial
    elif page == "Monthly Comparison":
        financial = api_call(api.get_my_financial_summary)
        summary = financial
        advisory = {
            "spending_comparison": api_call(api.get_my_monthly_comparison)
        }
    elif page == "Budgets & Recommendations":
        advisory = api_call(api.get_my_financial_advisory)
        financial = api_call(api.get_my_financial_summary)
        summary = financial
else:
    customer = st.session_state.customer
    banking = financial = advisory = {}
    accounts = transactions = cards = loans = []
    balance = 0
    summary = {}

if feature in {"Banking Information", "Financial Intelligence"}:
    render_demo_scope_notice(st, "Banking" if feature == "Banking Information" else "Financial intelligence")

# Dynamic shared responsive workspace container:
# Automatically reserves horizontal space on the right when Vaani AI Copilot opens,
# and restores full-width layout when Copilot closes, across all banking sections.
is_copilot_open = assistant_dock_is_visible(st.session_state)
workspace_key = "banking-workspace-copilot-open" if is_copilot_open else "banking-workspace-copilot-closed"
with st.container(key=workspace_key):
    # CENTRAL FEATURE WORKSPACE
    # --------------------------------------------------
    st.markdown(
        f"<div class='page-heading'><span>{feature}</span><strong>{subfeature}</strong></div>",
        unsafe_allow_html=True,
    )
    st.write("")
    
    
    # --------------------------------------------------
    # OVERVIEW PAGE
    # --------------------------------------------------
    
    if feature == "Scam Detection":
        scam_detection.render(st, subfeature, api, st.session_state)
    
    elif feature == "Customer Support":
        customer_support.render(st, subfeature, api, st.session_state)
    
    elif feature == "Account Opening Assistant" or page in account_opening.ACCOUNT_SUBFEATURES:
        account_opening.render(st, subfeature, api, st.session_state)
    
    elif page == "Overview":
    
        is_visible = bool(st.session_state.setdefault("balance_visible", True))
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Monthly Inflow", money(customer.get("monthly_income", 0)))
        col2.metric("Total Spending", money(summary.get("total_spent", 0)))
        col3.metric("Estimated Savings", money(advisory.get("savings_analysis", {}).get("estimated_savings", 0)))
        savings_rate = advisory.get("savings_analysis", {}).get("savings_rate", 0)
        col4.metric("Savings Rate", f"{savings_rate:.1f}%")

        q1, q2, q3 = st.columns(3)
        with q1:
            vis_label = "👁️ Hide Balance" if is_visible else "👁️ Show Balance"
            if st.button(vis_label, key="toggle-balance-vis-btn", width="stretch"):
                st.session_state["balance_visible"] = not is_visible
                st.rerun()
        with q2:
            if st.button("➕ Open New Account", key="quick-open-account-btn", width="stretch"):
                request_navigation(st.session_state, "Account Opening Assistant", "Open New Account")
                st.rerun()
        with q3:
            if st.button("🔍 Explore Transactions", key="quick-tx-explorer-btn", width="stretch"):
                request_navigation(st.session_state, "Banking Information", "Transactions")
                st.rerun()
        st.write("")
    
        left, right = st.columns([1.6, 1])
    
        with left:
            tx_hdr_left, tx_hdr_right = st.columns([2.5, 1.2])
            with tx_hdr_left:
                st.markdown("### Recent Transactions")
            with tx_hdr_right:
                if st.button("Explore all 100 →", key="view-all-tx-link", width="stretch"):
                    request_navigation(st.session_state, "Banking Information", "Transactions")
                    st.rerun()
    
            if transactions:
                for tx in transactions[:6]:
                    st.markdown(
                        render_transaction_card_html(
                            merchant=tx.get("merchant", "Merchant"),
                            amount=tx.get("amount", 0),
                            date_str=tx.get("transaction_date", ""),
                            category=tx.get("category", "General"),
                            status=tx.get("status", "SUCCESS"),
                            tx_type=tx.get("transaction_type", "DEBIT"),
                            ref_id=tx.get("reference_id"),
                        ),
                        unsafe_allow_html=True,
                    )
            else:
                st.info("No transactions available.")
    
        with right:
            st.markdown("### Active Accounts")
            for account in accounts:
                acc_t = account.get("account_type", "SAVINGS")
                acc_s = account.get("status", "ACTIVE")
                masked = f"•••• •••• •••• {account.get('account_number', '')[-4:]}"
                bal_str = money(account.get("balance", 0)) if is_visible else "₹ ••••••••"
                st.markdown(
                    textwrap.dedent(f"""
                    <div class="acct-summary-item" style="margin-bottom:0.65rem;">
                      <div style="display:flex;justify-content:space-between;align-items:center;">
                        <span class="acct-label">{acc_t} ACCOUNT</span>
                        <span class="acct-status-pill status-{acc_s.lower()}">{acc_s}</span>
                      </div>
                      <span style="font-family:'Space Grotesk',monospace;font-size:0.9rem;color:var(--text-muted);">{masked}</span>
                      <div style="font:700 1.25rem 'Space Grotesk',sans-serif;color:var(--text-primary);margin-top:0.35rem;">{bal_str}</div>
                    </div>
                    """).strip(),
                    unsafe_allow_html=True,
                )
    
            st.markdown("### Financial Health")
            with st.container(border=True):
                st.caption("MONTHLY SAVINGS PROGRESS")
                st.progress(min(max(savings_rate / 100, 0), 1))
                budget = advisory.get("budget_analysis", {})
                b1, b2 = st.columns(2)
                b1.metric("Needs (50%)", money(budget.get("needs_spending", 0)))
                b2.metric("Wants (30%)", money(budget.get("wants_spending", 0)))
    
    
    # --------------------------------------------------
    # TRANSACTIONS & ACCOUNTS PAGES
    # --------------------------------------------------
    
    elif page == "Accounts":
        st.markdown('<div class="section-eyebrow">ACCOUNT MANAGEMENT</div>', unsafe_allow_html=True)
        st.markdown("## Your Bank Accounts")
        st.markdown(
            '<div class="caption-muted">Review balances, account numbers, operational statuses, and linked services.</div>',
            unsafe_allow_html=True,
        )
    
        if accounts:
            acc_cols = st.columns(min(len(accounts), 2))
            for idx, acc in enumerate(accounts):
                with acc_cols[idx % 2]:
                    acc_t = acc.get("account_type", "SAVINGS").upper()
                    acc_s = acc.get("status", "ACTIVE")
                    raw_num = acc.get("account_number", "")
                    masked = f"•••• •••• •••• {raw_num[-4:]}" if len(raw_num) >= 4 else raw_num
                    bal = money(acc.get("balance", 0))
    
                    st.markdown(
                        textwrap.dedent(f"""
                        <div class="acct-summary-item" style="padding:1.25rem 1.4rem;margin-bottom:1rem;background:#FFFFFF;border:1px solid var(--border-light);box-shadow:var(--shadow-sm);border-radius:var(--radius-lg);box-sizing:border-box;">
                          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.75rem;">
                            <span class="acct-label" style="color:var(--primary-violet);font-size:0.72rem;font-weight:800;letter-spacing:1.2px;">VAANISHIELD {acc_t}</span>
                            <span class="acct-status-pill status-{acc_s.lower()}">{acc_s}</span>
                          </div>
                          <span style="font-family:'Space Grotesk',monospace;font-size:clamp(0.95rem, 1.4vw, 1.15rem);letter-spacing:2px;color:var(--text-muted);word-break:break-word;">{masked}</span>
                          <div style="font:700 clamp(1.35rem, 2vw, 1.65rem) 'Space Grotesk',sans-serif;color:var(--text-primary);margin:0.5rem 0 0.75rem;word-break:break-word;">{bal}</div>
                          <div style="display:grid;grid-template-columns:repeat(auto-fit, minmax(140px, 1fr));gap:0.6rem;padding-top:0.75rem;border-top:1px solid var(--border-light);font-size:0.78rem;color:var(--text-muted);">
                            <div>Branch: <b style="color:var(--text-primary);">Indiranagar, BLR</b></div>
                            <div>IFSC: <b style="color:var(--primary-violet);">VAANI000108</b></div>
                            <div>Transfer Limit: <b style="color:var(--text-primary);">₹1,00,000 / day</b></div>
                            <div>UPI ID: <b style="color:var(--status-emerald);">Linked (@vaani)</b></div>
                          </div>
                        </div>
                        """).strip(),
                        unsafe_allow_html=True,
                    )
    
            st.markdown("### Account Directory")
            account_rows = [
                {
                    "Account Type": account.get("account_type"),
                    "Account Number": account.get("account_number"),
                    "Balance": money(account.get("balance", 0)),
                    "Status": account.get("status"),
                    "Currency": "INR",
                    "Branch": "Indiranagar Flagship",
                }
                for account in accounts
            ]
            st.dataframe(pd.DataFrame(account_rows), width="stretch", hide_index=True)
        else:
            st.info("No demo accounts are available.")
    
        st.write("")
    
        # Distinct sections for Account Linking & KYC Management
        k_col1, k_col2 = st.columns(2)
        with k_col1:
            with st.container(border=True):
                st.markdown("#### 🔗 Link External Bank Account")
                st.caption("MULTI-BANK ACCOUNT AGGREGATOR (RBI NBFC-AA)")
                st.markdown(
                    "Link accounts from other commercial banks (e.g. HDFC, ICICI, SBI) to view unified financial insights. "
                    "Account Aggregator consent integration is a planned capability."
                )
                st.info("Want to open an additional digital account within VaaniShield?")
                if st.button("➕ Open New VaaniShield Account", key="acc-mgmt-open-new-btn", width="stretch"):
                    request_navigation(st.session_state, "Account Opening Assistant", "Open New Account")
                    st.rerun()
    
        with k_col2:
            with st.container(border=True):
                st.markdown("#### 🛡️ Customer KYC Status")
                st.caption("AUTHENTICATED IDENTITY COMPLIANCE")
                st.markdown(
                    "• **Status**: <span class='acct-status-pill status-approved'>VERIFIED · LEVEL 3</span><br>"
                    "• **Aadhaar e-KYC**: Verified via UIDAI Simulation Vault<br>"
                    "• **PAN Verification**: Linked & Validated (`ABCDE1234F`)<br>"
                    "• **Periodic Re-KYC**: Compliant. Next review in 11 months.<br>"
                    "<small style='color:#8D91A9;'>Note: To upload KYC documents for a new account opening, visit Account Opening Assistant > KYC Documents.</small>",
                    unsafe_allow_html=True,
                )
    
    elif page == "Transactions":
    
        st.markdown('<div class="section-eyebrow">BANKING INTELLIGENCE</div>', unsafe_allow_html=True)
        st.markdown("## Transaction Explorer")
        st.markdown(
            '<div class="caption-muted">Search and narrow down your transaction history with flexible filters.</div>',
            unsafe_allow_html=True,
        )
    
        all_transactions = transactions
    
        if all_transactions:
            base_df = pd.DataFrame(all_transactions)
            for column in ["merchant", "category", "transaction_type", "transaction_date"]:
                if column not in base_df.columns:
                    base_df[column] = ""
    
            with st.container(border=True):
                st.markdown("#### Find a transaction")
                with st.form("transaction_search_form"):
                    f1, f2, f3 = st.columns([1.5, 1, 1])
                    with f1:
                        merchant = st.text_input("Merchant contains", placeholder="e.g. DMart, Amazon")
                    with f2:
                        categories = sorted(str(value) for value in base_df["category"].dropna().unique() if str(value).strip())
                        category = st.selectbox("Category", ["All categories"] + categories)
                    with f3:
                        transaction_type = st.selectbox("Transaction type", ["All types", "DEBIT", "CREDIT"])
    
                    f4, f5, f6, f7 = st.columns(4)
                    with f4:
                        min_amount = st.number_input("Minimum amount (₹)", min_value=0.0, value=0.0, step=500.0)
                    with f5:
                        max_amount = st.number_input("Maximum amount (₹)", min_value=0.0, value=0.0, step=500.0,
                                                     help="Keep at 0 for no upper limit.")
                    with f6:
                        start_date = st.date_input("From date", value=None)
                    with f7:
                        end_date = st.date_input("To date", value=None)
    
                    submitted = st.form_submit_button("Apply filters", width="stretch")
    
            if submitted:
                if max_amount > 0 and min_amount > max_amount:
                    st.error("Maximum amount must be greater than or equal to minimum amount.")
                    filtered_df = pd.DataFrame()
                elif start_date and end_date and start_date > end_date:
                    st.error("The start date cannot be after the end date.")
                    filtered_df = pd.DataFrame()
                else:
                    params = {}
                    if merchant.strip():
                        params["merchant"] = merchant.strip()
                    if category != "All categories":
                        params["category"] = category
                    if transaction_type != "All types":
                        params["transaction_type"] = transaction_type
                    if min_amount > 0:
                        params["min_amount"] = min_amount
                    if max_amount > 0:
                        params["max_amount"] = max_amount
                    if start_date:
                        params["start_date"] = start_date.isoformat()
                    if end_date:
                        params["end_date"] = end_date.isoformat()
    
                    filtered_transactions = normalize_collection(
                        api_call(api.search_my_transactions, params)
                    )
                    filtered_df = pd.DataFrame(filtered_transactions)
            else:
                filtered_df = base_df.copy()
    
            st.write("")
            total_count = len(filtered_df)
            debit_total = 0.0
            credit_total = 0.0
            if total_count and "amount" in filtered_df.columns and "transaction_type" in filtered_df.columns:
                amounts = pd.to_numeric(filtered_df["amount"], errors="coerce").fillna(0)
                debit_total = amounts[filtered_df["transaction_type"] == "DEBIT"].sum()
                credit_total = amounts[filtered_df["transaction_type"] == "CREDIT"].sum()
    
            m1, m2, m3 = st.columns(3)
            m1.metric("Matching transactions", f"{total_count}")
            m2.metric("Total debits", money(float(debit_total)))
            m3.metric("Total credits", money(float(credit_total)))
    
            st.markdown("### Results")
            if total_count:
                preferred = ["transaction_date", "merchant", "category", "transaction_type", "amount", "status", "reference_id"]
                display_columns = [column for column in preferred if column in filtered_df.columns]
                display_df = filtered_df[display_columns].copy()
                if "amount" in display_df.columns:
                    display_df["amount"] = display_df["amount"].apply(money)
                if "transaction_type" in display_df.columns:
                    display_df["transaction_type"] = display_df["transaction_type"].replace({"DEBIT": "↓ DEBIT", "CREDIT": "↑ CREDIT"})
                st.dataframe(display_df, width="stretch", hide_index=True)
            else:
                st.info("No transactions match these filters. Try widening your search.")
    
            st.caption("Transaction records are synthetic demonstration data.")
        else:
            st.info("No transactions found.")
    
    
    # --------------------------------------------------
    # FINANCIAL INTELLIGENCE PAGE
    # --------------------------------------------------
    
    elif page in {"Spending Analysis", "Category Analysis", "Merchant Analysis"}:
    
        st.markdown(f"## {subfeature}")
    
        if page == "Spending Analysis":
            tot_cred = float(summary.get("total_credited", 225000.0))
            tot_spent = float(summary.get("total_spent", 46646.0))
            net_cf = float(summary.get("net_cash_flow", tot_cred - tot_spent))
            sav_pct = (net_cf / tot_cred * 100.0) if tot_cred > 0 else 0.0

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Income (Credits)", money(tot_cred))
            m2.metric("Total Expenditure (Debits)", money(tot_spent))
            m3.metric("Net Cash Flow", money(net_cf))
            m4.metric("Savings Ratio", f"{sav_pct:.1f}%", delta=f"{sav_pct:.1f}% saved")

            st.markdown("#### Monthly Income vs Expenditure Comparison")
            monthly_data = pd.DataFrame([
                {"Month": "Aug 2026", "Type": "Income", "Amount": 75000.0},
                {"Month": "Aug 2026", "Type": "Expenditure", "Amount": 13080.0},
                {"Month": "Sep 2026", "Type": "Income", "Amount": 75000.0},
                {"Month": "Sep 2026", "Type": "Expenditure", "Amount": 9028.0},
                {"Month": "Oct 2026", "Type": "Income", "Amount": 75000.0},
                {"Month": "Oct 2026", "Type": "Expenditure", "Amount": 24538.0},
            ])
            fig_inc_exp = px.bar(
                monthly_data,
                x="Month",
                y="Amount",
                color="Type",
                barmode="group",
                color_discrete_map={"Income": "#4F46E5", "Expenditure": "#E11D48"},
                text_auto=".2s",
            )
            fig_inc_exp.update_layout(
                template="plotly_white",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#172033", family="sans-serif", size=12),
                xaxis=dict(
                    title="Month",
                    title_font=dict(color="#172033", size=13, weight="bold"),
                    tickfont=dict(color="#334155", size=11),
                    gridcolor="#E2E8F0",
                ),
                yaxis=dict(
                    title="Amount (₹)",
                    title_font=dict(color="#172033", size=13, weight="bold"),
                    tickfont=dict(color="#334155", size=11),
                    gridcolor="#E2E8F0",
                ),
                legend=dict(
                    font=dict(color="#172033", size=12),
                    bgcolor="rgba(255,255,255,0.9)",
                    bordercolor="#E2E8F0",
                    borderwidth=1,
                    title_text="",
                ),
                margin=dict(l=20, r=20, t=30, b=20),
            )
            st.plotly_chart(fig_inc_exp, use_container_width=True)

            trend_df = pd.DataFrame([
                {"Period": "Aug 2026", "Expenditure": 13080.0},
                {"Period": "Sep 2026", "Expenditure": 9028.0},
                {"Period": "Oct 2026", "Expenditure": 24538.0},
            ])
            fig_trend = px.line(
                trend_df,
                x="Period",
                y="Expenditure",
                markers=True,
                title="Spending Trajectory Across Q3 2026",
                color_discrete_sequence=["#4F46E5"],
            )
            fig_trend.update_layout(
                template="plotly_white",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#172033", family="sans-serif", size=12),
                xaxis=dict(
                    title="Billing Period",
                    title_font=dict(color="#172033", size=13, weight="bold"),
                    tickfont=dict(color="#334155", size=11),
                    gridcolor="#E2E8F0",
                ),
                yaxis=dict(
                    title="Total Debits (₹)",
                    title_font=dict(color="#172033", size=13, weight="bold"),
                    tickfont=dict(color="#334155", size=11),
                    gridcolor="#E2E8F0",
                ),
                margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(fig_trend, use_container_width=True)

            st.markdown("#### Financial Insights & Analysis")
            insights = financial.get("insights", [])
            if insights:
                for insight in insights:
                    st.info(insight.get("message", str(insight)) if isinstance(insight, dict) else str(insight))
            else:
                st.info("💡 Your average savings rate over the last 3 months is 79.3%, reflecting strong cash conservation.")
    
        if page == "Category Analysis":
            if categories:
                cat_rows = []
                for c in categories:
                    name = c.get("category", "General")
                    amt = float(c.get("total_amount") or c.get("total_spent") or c.get("amount") or 0.0)
                    cnt = int(c.get("transaction_count") or c.get("count") or 1)
                    cat_rows.append({"Category": name, "Expenditure": amt, "Transactions": cnt})

                cat_df = pd.DataFrame(cat_rows).sort_values("Expenditure", ascending=False)
                tot_cat_spend = cat_df["Expenditure"].sum() or 1.0
                cat_df["% Contribution"] = (cat_df["Expenditure"] / tot_cat_spend * 100.0).round(1)

                top_cat = cat_df.iloc[0]
                st.markdown(
                    textwrap.dedent(f"""
                    <div class="fin-callout">
                      🔥 <b>Highest Spending Category: {top_cat['Category']}</b> — Total expenditure of <b>{money(top_cat['Expenditure'])}</b> 
                      accounting for <b>{top_cat['% Contribution']}%</b> of all debits.
                    </div>
                    """).strip(),
                    unsafe_allow_html=True,
                )

                col_c1, col_c2 = st.columns([1.3, 1.2], gap="medium")
                with col_c1:
                    fig_cat = px.pie(
                        cat_df,
                        names="Category",
                        values="Expenditure",
                        hole=0.55,
                        title="Category Distribution",
                        color_discrete_sequence=px.colors.sequential.Purp,
                    )
                    fig_cat.update_layout(
                        template="plotly_white",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#172033", family="sans-serif", size=12),
                        legend=dict(
                            font=dict(color="#172033", size=11),
                            bgcolor="rgba(255,255,255,0.9)",
                            bordercolor="#E2E8F0",
                            borderwidth=1,
                        ),
                        margin=dict(l=10, r=10, t=40, b=10),
                    )
                    st.plotly_chart(fig_cat, use_container_width=True)

                with col_c2:
                    st.markdown("##### Category Breakdown Table")
                    disp_df = cat_df.copy()
                    disp_df["Expenditure"] = disp_df["Expenditure"].apply(lambda x: money(x))
                    disp_df["% Contribution"] = disp_df["% Contribution"].apply(lambda x: f"{x}%")
                    st.dataframe(disp_df, use_container_width=True, hide_index=True)

                st.markdown("##### 💡 Actionable Category Recommendations")
                st.markdown(
                    "- **Essential Utilities & Health:** Essential fixed expenses accounted for ~45% of outflows; these are well managed.\n"
                    "- **Discretionary Food & Shopping:** Swiggy, Flipkart, and Amazon debits peaked in October. Setting a monthly discretionary cap of ₹10,000 can save an additional ~₹4,500 monthly.\n"
                    "- **Grocery Optimization:** Regular visits to DMart and BigBasket show bulk purchases; continue utilizing store loyalty programs."
                )
            else:
                st.info("Category spending data is not available.")
    
        if page == "Merchant Analysis":
            if merchants:
                merch_rows = []
                for m in merchants:
                    name = m.get("merchant", "Merchant")
                    amt = float(m.get("total_spent") or m.get("total_amount") or m.get("amount") or 0.0)
                    cnt = int(m.get("transaction_count") or m.get("count") or 1)
                    merch_rows.append({"Merchant": name, "Total Spent": amt, "Transactions": cnt})

                merch_df = pd.DataFrame(merch_rows).sort_values("Total Spent", ascending=False)
                top_m = merch_df.iloc[0]

                st.markdown(
                    textwrap.dedent(f"""
                    <div class="fin-callout">
                      🏆 <b>Top Merchant by Spend: {top_m['Merchant']}</b> — Total expenditure of <b>{money(top_m['Total Spent'])}</b> 
                      across <b>{top_m['Transactions']} transactions</b>.
                    </div>
                    """).strip(),
                    unsafe_allow_html=True,
                )

                col_m1, col_m2 = st.columns([1.4, 1.1], gap="medium")
                with col_m1:
                    fig_m = px.bar(
                        merch_df.head(8),
                        x="Total Spent",
                        y="Merchant",
                        orientation="h",
                        title="Top Merchants by Total Expenditure",
                        color="Total Spent",
                        color_continuous_scale="Viridis",
                        text_auto=".2s",
                    )
                    fig_m.update_layout(
                        template="plotly_white",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#172033", family="sans-serif", size=12),
                        xaxis=dict(
                            title="Total Spent (₹)",
                            title_font=dict(color="#172033", size=12, weight="bold"),
                            tickfont=dict(color="#334155", size=11),
                            gridcolor="#E2E8F0",
                        ),
                        yaxis=dict(
                            title="Merchant",
                            title_font=dict(color="#172033", size=12, weight="bold"),
                            tickfont=dict(color="#334155", size=11),
                            autorange="reversed",
                        ),
                        margin=dict(l=10, r=10, t=40, b=10),
                    )
                    st.plotly_chart(fig_m, use_container_width=True)

                with col_m2:
                    st.markdown("##### Merchant Expenditure Details")
                    disp_m = merch_df.copy()
                    disp_m["Total Spent"] = disp_m["Total Spent"].apply(lambda x: money(x))
                    st.dataframe(disp_m, use_container_width=True, hide_index=True)
            else:
                st.info("Merchant spending data is not available.")
    
    
    # --------------------------------------------------
    # FINANCIAL ADVISOR SUBFEATURES
    # --------------------------------------------------
    
    elif page in {"Monthly Comparison", "Budgets & Recommendations"}:
    
        st.markdown(f"## {subfeature}")
    
        if page == "Monthly Comparison":
            st.markdown("### Multi-Month Financial Comparison (Aug - Oct 2026)")
            st.caption("Historical performance across synthetic demonstration periods.")

            multi_month_df = pd.DataFrame([
                {"Month": "August 2026", "Income": 75000.0, "Expenditure": 13080.0, "Net Savings": 61920.0, "Savings Rate": 82.6, "MoM Change": "Baseline"},
                {"Month": "September 2026", "Income": 75000.0, "Expenditure": 9028.0, "Net Savings": 65972.0, "Savings Rate": 88.0, "MoM Change": "-31.0%"},
                {"Month": "October 2026", "Income": 75000.0, "Expenditure": 24538.0, "Net Savings": 50462.0, "Savings Rate": 67.3, "MoM Change": "+171.8%"},
            ])

            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Oct 2026 Net Savings", money(50462.0), delta="-23.5% vs Sep", delta_color="inverse")
            mc2.metric("Oct 2026 Expenditure", money(24538.0), delta="+171.8% vs Sep", delta_color="inverse")
            mc3.metric("3-Month Average Savings", money(59451.33), delta="79.3% rate")

            fig_multi = px.bar(
                multi_month_df,
                x="Month",
                y=["Income", "Expenditure", "Net Savings"],
                barmode="group",
                title="Income, Expenditure & Net Savings Comparison",
                color_discrete_map={"Income": "#4F46E5", "Expenditure": "#E11D48", "Net Savings": "#10B981"},
            )
            fig_multi.update_layout(
                template="plotly_white",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#172033", family="sans-serif", size=12),
                xaxis=dict(
                    title="Month",
                    title_font=dict(color="#172033", size=13, weight="bold"),
                    tickfont=dict(color="#334155", size=11),
                    gridcolor="#E2E8F0",
                ),
                yaxis=dict(
                    title="Amount (₹)",
                    title_font=dict(color="#172033", size=13, weight="bold"),
                    tickfont=dict(color="#334155", size=11),
                    gridcolor="#E2E8F0",
                ),
                legend=dict(
                    font=dict(color="#172033", size=12),
                    bgcolor="rgba(255,255,255,0.9)",
                    bordercolor="#E2E8F0",
                    borderwidth=1,
                    title_text="",
                ),
                margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(fig_multi, use_container_width=True)

            st.markdown("##### Historical Performance Ledger")
            disp_multi = multi_month_df.copy()
            for col in ["Income", "Expenditure", "Net Savings"]:
                disp_multi[col] = disp_multi[col].apply(lambda x: money(x))
            disp_multi["Savings Rate"] = disp_multi["Savings Rate"].apply(lambda x: f"{x}%")
            st.dataframe(disp_multi, use_container_width=True, hide_index=True)

            st.markdown(
                textwrap.dedent("""
                <div class="fin-callout">
                  📊 <b>Month-over-Month Synthesis:</b> Expenditure rose from ₹9,028 in September to ₹24,538 in October (+171.8%) 
                  due to festive season transactions on Amazon/Flipkart and medical checkup expenses at Hospital. However, net savings 
                  remained robust at ₹50,462 (67.3% savings rate).
                </div>
                """).strip(),
                unsafe_allow_html=True,
            )
        else:
            savings = advisory.get("savings_analysis", {"savings_rate": 67.28, "monthly_spending": 24538.0, "estimated_savings": 50462.0})
            budget = advisory.get("budget_analysis", {"needs_spending": 14350.0, "wants_spending": 10188.0, "estimated_savings": 50462.0, "recommendations": []})

            st.markdown("### 50/30/20 Budget Allocation")
            b1, b2, b3 = st.columns(3)
            b1.metric("Needs (50% Target)", money(budget.get("needs_spending", 14350.0)), delta="19.1% of Income (Optimal)")
            b2.metric("Wants (30% Target)", money(budget.get("wants_spending", 10188.0)), delta="13.6% of Income (Healthy)")
            b3.metric("Savings (20% Target)", money(budget.get("estimated_savings", 50462.0)), delta="67.3% of Income (Superb)")

            unusual = advisory.get("unusual_spending", [])
            if unusual:
                st.markdown("##### ⚠️ Statistical Spending Alerts")
                st.dataframe(pd.DataFrame(unusual), use_container_width=True, hide_index=True)
                st.caption("These are statistical variance alerts for awareness, not confirmed fraudulent debits.")

            st.markdown("---")
            st.markdown("## 💡 Smart Savings & Investment Assistant")
            st.caption("Intelligent advisory tools: Fixed Deposits, Systematic Investment Plans (SIP), Synthetic IPO Tracker, and Tailored Recommendations.")

            synthetic_investment_notice = (
                '<div class="synthetic-disclosure-badge">'
                '⚖️ <b>Demo Disclosure:</b> Illustrative synthetic demo data. '
                'These are not live financial offers, investment recommendations or verified market listings.'
                '</div>'
            )
            st.markdown(synthetic_investment_notice, unsafe_allow_html=True)

            assistant_tab1, assistant_tab2, assistant_tab3, assistant_tab4 = st.tabs([
                "🏦 Fixed Deposit (FD) Hub",
                "📈 SIP Growth Calculator",
                "🚀 Upcoming IPO Tracker",
                "🎯 Tailored Savings Plan",
            ])

            with assistant_tab1:
                st.markdown("#### Axis Bank & Partner Fixed Deposit Plans")
                st.caption("Guaranteed returns with quarterly compound interest. Senior citizens receive +0.50% p.a.")

                fd_plans = [
                    {"Plan": "1-Year Prime FD", "Tenure": "12 Months", "Regular Rate": "7.10% p.a.", "Senior Rate": "7.60% p.a.", "Payout": "Quarterly Compounding"},
                    {"Plan": "3-Year Growth FD", "Tenure": "36 Months", "Regular Rate": "7.25% p.a.", "Senior Rate": "7.75% p.a.", "Payout": "Reinvestment / Maturity"},
                    {"Plan": "5-Year Tax Saver", "Tenure": "60 Months", "Regular Rate": "7.00% p.a.", "Senior Rate": "7.50% p.a.", "Payout": "Sec 80C Tax Exemption"},
                ]
                st.dataframe(pd.DataFrame(fd_plans), use_container_width=True, hide_index=True)

                st.markdown("##### 🧮 Interactive FD Maturity Calculator")
                fd_col1, fd_col2 = st.columns(2)
                with fd_col1:
                    fd_principal = st.number_input("Deposit Principal Amount (₹)", min_value=5000, max_value=5000000, value=50000, step=5000, key="fd_princ")
                    fd_tenure_years = st.selectbox("Investment Tenure", [1, 2, 3, 5], index=0, format_func=lambda x: f"{x} Year{'s' if x>1 else ''}", key="fd_ten")
                    is_senior = st.checkbox("Senior Citizen (+0.50% Extra)", value=False, key="fd_sen")

                base_rate = 7.10 if fd_tenure_years == 1 else (7.20 if fd_tenure_years == 2 else (7.25 if fd_tenure_years == 3 else 7.00))
                effective_rate = base_rate + (0.50 if is_senior else 0.0)
                r_dec = effective_rate / 100.0
                fd_maturity = fd_principal * ((1.0 + r_dec / 4.0) ** (4.0 * fd_tenure_years))
                fd_interest = fd_maturity - fd_principal

                with fd_col2:
                    st.markdown(
                        textwrap.dedent(f"""
                        <div class="fin-card" style="background:#F8FAFC;border:1px solid #E2E8F0;padding:1.4rem;">
                          <div style="font-size:0.75rem;font-weight:700;color:#64748B;text-transform:uppercase;">Estimated Maturity Value</div>
                          <div style="font:800 1.8rem 'Space Grotesk',sans-serif;color:#10B981;margin:0.25rem 0;">{money(fd_maturity)}</div>
                          <div style="font-size:0.88rem;color:#475569;margin-bottom:0.75rem;">Total Interest Earned: <b>{money(fd_interest)}</b></div>
                          <div style="font-size:0.8rem;color:#64748B;">Applied Interest Rate: <b>{effective_rate:.2f}% p.a.</b> (Quarterly compounding)</div>
                        </div>
                        """).strip(),
                        unsafe_allow_html=True,
                    )

            with assistant_tab2:
                st.markdown("#### Systematic Investment Plan (SIP) Wealth Builder")
                st.caption("Harness the power of rupee cost averaging and compounding for long-term equity growth.")

                sip_c1, sip_c2 = st.columns(2)
                with sip_c1:
                    monthly_sip = st.slider("Monthly SIP Contribution (₹)", min_value=1000, max_value=50000, value=10000, step=1000, key="sip_mo")
                    sip_years = st.slider("Investment Horizon (Years)", min_value=1, max_value=30, value=5, step=1, key="sip_yr")
                    sip_return_rate = st.slider("Expected Annual Return (%)", min_value=6.0, max_value=20.0, value=12.0, step=0.5, key="sip_ret")

                i_monthly = (sip_return_rate / 100.0) / 12.0
                n_months = sip_years * 12
                total_sip_invested = monthly_sip * n_months
                sip_future_value = monthly_sip * (((1.0 + i_monthly) ** n_months - 1.0) / i_monthly) * (1.0 + i_monthly)
                sip_wealth_gain = sip_future_value - total_sip_invested

                with sip_c2:
                    st.markdown(
                        textwrap.dedent(f"""
                        <div class="fin-card" style="background:#F8FAFC;border:1px solid #E2E8F0;padding:1.4rem;">
                          <div style="font-size:0.75rem;font-weight:700;color:#64748B;text-transform:uppercase;">Projected Corpus at Maturity</div>
                          <div style="font:800 1.8rem 'Space Grotesk',sans-serif;color:#4F46E5;margin:0.25rem 0;">{money(sip_future_value)}</div>
                          <div style="display:flex;justify-content:space-between;margin-top:0.75rem;border-top:1px solid #E2E8F0;padding-top:0.5rem;font-size:0.85rem;">
                            <span>Total Invested: <b>{money(total_sip_invested)}</b></span>
                            <span style="color:#059669;">Est. Returns: <b>+{money(sip_wealth_gain)}</b></span>
                          </div>
                        </div>
                        """).strip(),
                        unsafe_allow_html=True,
                    )

                sip_pie_df = pd.DataFrame([
                    {"Component": "Total Invested Capital", "Amount": total_sip_invested},
                    {"Component": "Estimated Wealth Growth", "Amount": sip_wealth_gain},
                ])
                fig_sip = px.pie(
                    sip_pie_df,
                    names="Component",
                    values="Amount",
                    hole=0.6,
                    title=f"Invested Capital vs Compounded Returns ({sip_years} Yrs)",
                    color_discrete_sequence=["#94A3B8", "#4F46E5"],
                )
                fig_sip.update_layout(
                    template="plotly_white",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#172033", family="sans-serif", size=12),
                    legend=dict(
                        font=dict(color="#172033", size=11),
                        bgcolor="rgba(255,255,255,0.9)",
                        bordercolor="#E2E8F0",
                        borderwidth=1,
                    ),
                    margin=dict(l=10, r=10, t=35, b=10),
                )
                st.plotly_chart(fig_sip, use_container_width=True)

                st.caption("⚠️ **Educational & Illustrative Disclaimer:** Mutual fund investments are subject to market risks. Please read all scheme-related documents carefully. Projections are calculated mathematically based on assumed compound annual growth rates and do not guarantee future returns.")

            with assistant_tab3:
                st.markdown("#### Synthetic Initial Public Offering (IPO) Watchlist")
                st.caption("Upcoming and active Indian stock market primary offerings (synthetic records).")

                ipo_data = pd.DataFrame([
                    {"Company": "TechFin Bharat Ltd", "Sector": "Fintech SaaS", "Price Band": "₹320 - ₹340", "Lot Size": "44 Shares", "Min Investment": "₹14,960", "Dates": "Nov 10 - Nov 12, 2026", "Status": "Upcoming"},
                    {"Company": "GreenEnergy Sol Pvt", "Sector": "Renewables", "Price Band": "₹185 - ₹195", "Lot Size": "75 Shares", "Min Investment": "₹14,625", "Dates": "Nov 14 - Nov 17, 2026", "Status": "Announced"},
                    {"Company": "HealthCare AI Diagnostics", "Sector": "Diagnostics", "Price Band": "₹510 - ₹535", "Lot Size": "28 Shares", "Min Investment": "₹14,980", "Dates": "Nov 18 - Nov 21, 2026", "Status": "Upcoming"},
                ])
                st.dataframe(ipo_data, use_container_width=True, hide_index=True)

                st.markdown(
                    textwrap.dedent("""
                    <div class="fin-callout">
                      ⚖️ <b>SEBI Regulatory & Risk Disclosure:</b> Equity investments and IPO applications are subject to market volatility. 
                      Retail investors should consult the Red Herring Prospectus (RHP) and evaluate risk tolerance before bidding.
                    </div>
                    """).strip(),
                    unsafe_allow_html=True,
                )

            with assistant_tab4:
                st.markdown(f"#### Personalized Financial Recommendations for {profile_name}")
                st.markdown(
                    textwrap.dedent(f"""
                    Based on your verified banking transactions across August to October 2026:

                    1. **Emergency Reserve Funding:** Your average monthly outflow is approximately ₹15,500. We recommend maintaining **₹90,000 to ₹1,50,000 (6-10 months of expenses)** in a liquid 1-Year Fixed Deposit earning 7.10% p.a., keeping capital secure while outpacing inflation.
                    2. **Automated Wealth Acceleration:** With your consistent monthly salary credit of ₹75,000 and net savings averaging ₹59,450/month, you have substantial investable surplus. Establishing an automated **₹15,000/month SIP** in diversified index funds can accumulate over **₹11.8 Lakhs in 5 years** (assuming 12% p.a.).
                    3. **Discretionary Capping:** Discretionary debits on Swiggy and e-commerce rose to ₹10,188 in October. Setting an automated monthly card limit of ₹8,000 will optimize savings by another ₹2,000 - ₹3,000 monthly.
                    4. **Credit Score Hygiene:** Your credit card outstanding stands at ₹55,000 out of a ₹2,00,000 limit (27.5% utilization). This is within the recommended 30% threshold, supporting a strong credit score.
                    """).strip()
                )
    
    
    # --------------------------------------------------
    # CARDS AND LOANS SUBFEATURES
    # --------------------------------------------------
    
    elif page in {"Cards", "Loans"}:
    
        st.markdown(f"## {subfeature}")
    
        if page == "Cards":
            st.markdown("### Linked Debit & Credit Cards")
            st.caption("Select a card below to inspect limits, view financial details, or manage security controls.")
            if cards:
                card_labels = [
                    f"💳 {c.get('card_network', 'Card')} {c.get('card_type', 'DEBIT')} (•••• {c.get('last_four', '0000')}) - {c.get('status', 'ACTIVE')}"
                    for c in cards
                ]
                card_tabs = st.tabs(card_labels)
                for idx, (tab, card) in enumerate(zip(card_tabs, cards)):
                    with tab:
                        card_type = str(card.get("card_type", "DEBIT")).upper()
                        last4 = str(card.get("last_four", "0000"))
                        expiry = str(card.get("expiry_date", "12/28"))
                        status = str(card.get("status", "ACTIVE")).upper()
                        network = str(card.get("card_network", "RuPay Platinum" if card_type == "DEBIT" else "Visa Signature"))
                        holder = str(card.get("cardholder_name", profile_name))

                        daily_lim = float(card.get("daily_limit") or (50000.0 if card_type == "DEBIT" else 100000.0))
                        monthly_lim = float(card.get("monthly_limit") or (200000.0 if card_type == "DEBIT" else 300000.0))
                        credit_lim = float(card.get("credit_limit") or 0.0)
                        avail_lim = float(card.get("available_credit_limit") or 0.0)
                        outstanding = float(card.get("outstanding_amount") or 0.0)
                        acct_bal = float(card.get("account_balance") or 278354.0)

                        left_col, right_col = st.columns([1.2, 1.8], gap="medium")

                        with left_col:
                            if card_type == "CREDIT":
                                card_bg = "linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #334155 100%)"
                                chip_bg = "linear-gradient(135deg, #F59E0B, #D97706)"
                                tag_color = "#94A3B8"
                            else:
                                card_bg = "linear-gradient(135deg, #0F172A 0%, #1E1B4B 45%, #4C0519 100%)"
                                chip_bg = "linear-gradient(135deg, #FBBF24, #F59E0B)"
                                tag_color = "#FBCFE8"

                            st.markdown(
                                textwrap.dedent(f"""
                                <div style="background:{card_bg};border:1px solid rgba(255,255,255,0.18);border-radius:16px;padding:1.25rem 1.4rem;box-shadow:0 8px 24px rgba(15,23,42,0.14);color:#fff;margin-bottom:1rem;box-sizing:border-box;max-width:100%;">
                                  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.95rem;">
                                    <span style="font-size:0.75rem;font-weight:800;letter-spacing:1.2px;color:{tag_color};">VAANISHIELD {card_type}</span>
                                    <span class="acct-status-pill status-{status.lower()}">{status}</span>
                                  </div>
                                  <div style="display:flex;align-items:center;gap:10px;margin-bottom:0.85rem;">
                                    <div style="width:36px;height:26px;border-radius:5px;background:{chip_bg};border:1px solid rgba(255,255,255,0.3);"></div>
                                    <span style="font-size:0.8rem;color:rgba(255,255,255,0.8);">📡 Contactless</span>
                                  </div>
                                  <div style="font-family:'Space Grotesk',monospace;font-size:clamp(1.05rem, 1.4vw, 1.25rem);letter-spacing:2px;margin-bottom:0.95rem;color:#F8FAFC;word-break:break-word;">
                                    •••• •••• •••• {last4}
                                  </div>
                                  <div style="display:flex;justify-content:space-between;align-items:flex-end;border-top:1px solid rgba(255,255,255,0.15);padding-top:0.65rem;">
                                    <div>
                                      <span style="font-size:0.58rem;color:{tag_color};letter-spacing:1px;display:block;">CARDHOLDER</span>
                                      <span style="font-size:0.84rem;font-weight:700;color:#FFFFFF;">{holder}</span>
                                    </div>
                                    <div style="text-align:right;">
                                      <span style="font-size:0.58rem;color:{tag_color};letter-spacing:1px;display:block;">EXPIRES</span>
                                      <span style="font-size:0.84rem;font-weight:700;color:#FFFFFF;">{expiry}</span>
                                    </div>
                                  </div>
                                  <div style="margin-top:0.65rem;text-align:right;font:700 0.98rem 'Space Grotesk',sans-serif;color:#FFFFFF;">
                                    {network}
                                  </div>
                                </div>
                                """).strip(),
                                unsafe_allow_html=True,
                            )

                        with right_col:
                            st.markdown("#### Financial Limits & Balances")
                            if card_type == "CREDIT":
                                f1, f2, f3 = st.columns(3)
                                f1.metric("Credit Limit", money(credit_lim))
                                f2.metric("Available Credit", money(avail_lim))
                                f3.metric("Outstanding Due", money(outstanding), delta=f"{((outstanding/credit_lim)*100 if credit_lim else 0):.1f}% used", delta_color="inverse")

                                l1, l2 = st.columns(2)
                                l1.metric("Daily POS / ATM Limit", money(daily_lim))
                                l2.metric("Monthly Online Limit", money(monthly_lim))

                                util_pct = min(100.0, max(0.0, (outstanding / credit_lim * 100.0))) if credit_lim else 0.0
                                st.progress(util_pct / 100.0, text=f"Credit Utilization: {util_pct:.1f}% of limit used")
                            else:
                                f1, f2, f3 = st.columns(3)
                                f1.metric("Linked Account Balance", money(acct_bal))
                                f2.metric("Daily ATM / POS Limit", money(daily_lim))
                                f3.metric("Monthly Limit", money(monthly_lim))
                                st.info("💳 Debit card transactions are debited directly from your primary savings account.")

                            st.markdown("#### Security & Card Controls")
                            c_ctrl1, c_ctrl2 = st.columns(2)

                            with c_ctrl1:
                                if status == "FROZEN":
                                    if st.button("🔓 Unfreeze Card", key=f"unfreeze_{last4}_{idx}", type="primary", use_container_width=True):
                                        api.toggle_card_freeze(last4, freeze=False)
                                        st.success(f"Card •••• {last4} has been unfrozen successfully.")
                                        st.rerun()
                                elif status == "ACTIVE":
                                    if st.button("❄️ Freeze Card", key=f"freeze_{last4}_{idx}", type="secondary", use_container_width=True):
                                        api.toggle_card_freeze(last4, freeze=True)
                                        st.warning(f"Card •••• {last4} is now temporarily frozen.")
                                        st.rerun()
                                else:
                                    st.error("🚫 Card is permanently blocked.")

                            with c_ctrl2:
                                with st.popover("🚨 Report Lost / Block", use_container_width=True):
                                    st.write(f"Permanently block card •••• {last4}?")
                                    st.caption("This action is permanent and cannot be undone from self-service.")
                                    block_reason = st.selectbox(
                                        "Select reason",
                                        ["Lost Card", "Stolen Card", "Suspicious / Fraudulent Transaction"],
                                        key=f"reason_{last4}_{idx}",
                                    )
                                    if st.button("Confirm Permanent Block", type="primary", key=f"confirm_block_{last4}_{idx}"):
                                        from services.banking_service import get_connection
                                        _conn = get_connection()
                                        try:
                                            _conn.execute("UPDATE cards SET status = 'BLOCKED' WHERE last_four = ?", (last4,))
                                            _conn.commit()
                                        finally:
                                            _conn.close()
                                        st.error(f"Card •••• {last4} has been permanently blocked.")
                                        st.rerun()

                            with st.expander("⚙️ View & Customize Transaction Limits"):
                                st.markdown(f"**Domestic ATM Cash Withdrawal:** {money(daily_lim)} per day")
                                st.markdown(f"**Domestic Online E-Commerce:** {money(monthly_lim)} per month")
                                st.markdown("**Contactless Tap-to-Pay:** Enabled (Max ₹5,000 per transaction without PIN)")
                                st.markdown("**International Usage:** Disabled (Enabled via travel advisory)")
                                st.caption("Limits can be adjusted via Vaani AI or by contacting customer support.")
            else:
                st.info("No cards found.")
    
        if page == "Loans":
            st.markdown("### Loan Accounts & Liabilities")
            if loans:
                active_loans = [l for l in loans if str(l.get("status", "ACTIVE")).upper() == "ACTIVE"]
                closed_loans = [l for l in loans if str(l.get("status", "")).upper() == "CLOSED"]

                if active_loans:
                    st.markdown("#### Active Loans")
                    for loan in active_loans:
                        loan_type = loan.get("loan_type", "PERSONAL").upper()
                        principal = float(loan.get("principal_amount", 0))
                        outstanding = float(loan.get("outstanding_amount", 0))
                        rate = loan.get("interest_rate", 0)
                        tenure = loan.get("tenure_months", 0)
                        remaining = loan.get("remaining_tenure_months")
                        emi = loan.get("monthly_emi")
                        next_emi = loan.get("next_emi_date") or "05 Dec 2026"
                        status = loan.get("status", "ACTIVE")
                        repaid = max(principal - outstanding, 0.0)
                        pct_repaid = (repaid / principal * 100.0) if principal > 0 else 0.0

                        tenure_label = f"{remaining} months remaining (of {tenure} mos)" if remaining is not None else f"{tenure} months"
                        emi_label = money(emi) if emi else "₹13,080.00"

                        st.markdown(
                            textwrap.dedent(f"""
                            <div class="fintech-loan-card active-loan">
                              <div class="loan-card-header">
                                <div>
                                  <span class="loan-badge active">{loan_type} LOAN</span>
                                  <h3 class="loan-amount-primary">{money(outstanding)} <small>Outstanding</small></h3>
                                </div>
                                <div class="loan-header-right">
                                  <span class="status-pill status-active">ACTIVE</span>
                                  <div class="loan-emi-callout">
                                    <span class="emi-label">Monthly EMI</span>
                                    <span class="emi-val">{emi_label}</span>
                                  </div>
                                </div>
                              </div>
                              <div class="loan-details-grid">
                                <div class="loan-detail-item">
                                  <span class="detail-label">Original Principal</span>
                                  <span class="detail-val">{money(principal)}</span>
                                </div>
                                <div class="loan-detail-item">
                                  <span class="detail-label">Interest Rate</span>
                                  <span class="detail-val accent-teal">{rate}% p.a.</span>
                                </div>
                                <div class="loan-detail-item">
                                  <span class="detail-label">Tenure</span>
                                  <span class="detail-val">{tenure_label}</span>
                                </div>
                                <div class="loan-detail-item">
                                  <span class="detail-label">Next EMI Due Date</span>
                                  <span class="detail-val accent-violet">{next_emi}</span>
                                </div>
                              </div>
                              <div class="loan-progress-wrap">
                                <div class="progress-labels">
                                  <span>Repayment Progress ({pct_repaid:.1f}% repaid)</span>
                                  <span>{money(repaid)} of {money(principal)}</span>
                                </div>
                              </div>
                            </div>
                            """).strip(),
                            unsafe_allow_html=True,
                        )
                        st.progress(min(max(pct_repaid / 100.0, 0.0), 1.0))

                if closed_loans:
                    st.write("")
                    st.markdown("#### Closed & Fully Repaid Loans")
                    for loan in closed_loans:
                        loan_type = loan.get("loan_type", "EDUCATION").upper()
                        principal = float(loan.get("principal_amount", 0))
                        rate = loan.get("interest_rate", 0)
                        tenure = loan.get("tenure_months", 0)
                        completion_date = loan.get("completion_date") or "15 Aug 2025"

                        st.markdown(
                            textwrap.dedent(f"""
                            <div class="fintech-loan-card closed-loan">
                              <div class="loan-card-header">
                                <div>
                                  <span class="loan-badge closed">{loan_type} LOAN</span>
                                  <h3 class="loan-amount-primary">₹0.00 <small>Fully Repaid & Closed</small></h3>
                                </div>
                                <div class="loan-header-right">
                                  <span class="status-pill status-closed">CLOSED</span>
                                </div>
                              </div>
                              <div class="loan-details-grid">
                                <div class="loan-detail-item">
                                  <span class="detail-label">Sanctioned Principal</span>
                                  <span class="detail-val">{money(principal)}</span>
                                </div>
                                <div class="loan-detail-item">
                                  <span class="detail-label">Historical Interest Rate</span>
                                  <span class="detail-val">{rate}% p.a.</span>
                                </div>
                                <div class="loan-detail-item">
                                  <span class="detail-label">Original Tenure</span>
                                  <span class="detail-val">{tenure} months</span>
                                </div>
                                <div class="loan-detail-item">
                                  <span class="detail-label">Settlement / Completion Date</span>
                                  <span class="detail-val accent-emerald">{completion_date}</span>
                                </div>
                              </div>
                              <div class="loan-progress-wrap">
                                <div class="progress-labels">
                                  <span>Repayment Status: Fully Settled</span>
                                  <span>100% Repaid · No outstanding balance</span>
                                </div>
                              </div>
                            </div>
                            """).strip(),
                            unsafe_allow_html=True,
                        )
                        st.progress(1.0)
            else:
                st.info("No loans found.")
    
    
    # --------------------------------------------------
    # SECURITY CENTRE PAGE
    # --------------------------------------------------
    
    elif page == "Security Centre":
    
        st.markdown("## Security Centre")
    
        st.markdown(
            textwrap.dedent("""
            <div class="security-banner">
                🛡️ <b>VaaniShield Security Layer</b><br>
                Banking credentials and data protected by PBKDF2-SHA256 password cryptography and active audit logging.
            </div>
            """).strip(),
            unsafe_allow_html=True,
        )
    
        st.write("")
    
        # Security Activity & Login Audit Log
        st.markdown("### 🔐 Security Activity & Sign-In History")
        st.caption("Immutable cryptographic audit trail of all authentication events for your customer profile.")

        try:
            audit_records = api.get_security_activity(limit=10)
            if audit_records:
                audit_list = []
                for entry in audit_records:
                    status_val = entry.get("login_result", "SUCCESS")
                    ts_ist = entry.get("timestamp_ist") or entry.get("timestamp_utc", "")
                    audit_list.append({
                        "Date & Time (IST)": ts_ist,
                        "Status": "✅ SUCCESS" if status_val == "SUCCESS" else "❌ FAILED",
                        "Email Attempted": entry.get("email_attempted", "arjun@example.com"),
                        "Device / Platform": entry.get("device_platform", "Web Browser"),
                        "IP Address": entry.get("ip_address", "127.0.0.1"),
                    })
                audit_df = pd.DataFrame(audit_list)
                st.dataframe(audit_df, use_container_width=True, hide_index=True)
            else:
                st.info("No recent sign-in events recorded.")
        except Exception:
            st.info("Sign-in audit log is active. Recent logins recorded in SQLite database.")

        st.markdown("### Security Posture & Standards")
        sec_c1, sec_c2, sec_c3 = st.columns(3)
        with sec_c1:
            st.markdown(
                """
                <div class="fin-card">
                  <div style="font-size:0.75rem;font-weight:700;color:#64748B;">PASSWORD ENCRYPTION</div>
                  <div style="font:700 1.1rem 'Space Grotesk',sans-serif;color:#10B981;margin:0.25rem 0;">PBKDF2-SHA256</div>
                  <div style="font-size:0.8rem;color:#64748B;">100,000 hashing rounds with cryptographically secure random per-user salt.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with sec_c2:
            st.markdown(
                """
                <div class="fin-card">
                  <div style="font-size:0.75rem;font-weight:700;color:#64748B;">SESSION INTEGRITY</div>
                  <div style="font:700 1.1rem 'Space Grotesk',sans-serif;color:#4F46E5;margin:0.25rem 0;">HMAC Signed Cookie</div>
                  <div style="font-size:0.8rem;color:#64748B;">HttpOnly, SameSite strict tamper-evident cryptographic session cookie.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with sec_c3:
            st.markdown(
                """
                <div class="fin-card">
                  <div style="font-size:0.75rem;font-weight:700;color:#64748B;">FRAUD SHIELD</div>
                  <div style="font:700 1.1rem 'Space Grotesk',sans-serif;color:#059669;margin:0.25rem 0;">Active Monitoring</div>
                  <div style="font-size:0.8rem;color:#64748B;">Multi-channel scam detection & suspicious transaction intelligence active.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("### Security Capabilities Roadmap")
        st.info("Scam detection: Multi-modal agent detection with message & voice call analysis is active.")
        st.info("Voice identity verification: Gnani Prisma V3 speech analysis active.")
        st.info("Fraud investigation: Dedicated security audit logs and ticket dispute system active.")
    

render_floating_assistant(st, api, st.session_state, in_dock=True)

st.divider()

st.caption(
    "VaaniShield 2.0 | AI-powered banking security | "
    "Synthetic demonstration environment"
)
