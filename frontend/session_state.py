"""Streamlit session defaults for authenticated VaaniShield UI state."""

from datetime import datetime, timezone
import streamlit as st
from streamlit.errors import StreamlitWidgetAlreadyInstantiatedError

from frontend.api_client import VaaniShieldAPI

RESPONSE_LANGUAGE_OPTIONS_FULL = ["Automatic", "English", "Kannada", "Hindi"]
RESPONSE_LANGUAGE_OPTIONS_SHORT = ["Auto", "English", "Kannada", "Hindi"]
RESPONSE_MODE_LABELS = {
    "text": "Text",
    "speech": "Speech",
    "speech_and_text": "Speech + Text",
}
RESPONSE_MODE_CODES = {label: code for code, label in RESPONSE_MODE_LABELS.items()}

CODE_TO_FULL_LABEL = {
    None: "Automatic",
    "en-IN": "English",
    "kn-IN": "Kannada",
    "hi-IN": "Hindi",
}

FULL_LABEL_TO_CODE = {
    "Automatic": None,
    "English": "en-IN",
    "Kannada": "kn-IN",
    "Hindi": "hi-IN",
}

CODE_TO_SHORT_LABEL = {
    None: "Auto",
    "en-IN": "English",
    "kn-IN": "Kannada",
    "hi-IN": "Hindi",
}

SHORT_LABEL_TO_CODE = {
    "Auto": None,
    "English": "en-IN",
    "Kannada": "kn-IN",
    "Hindi": "hi-IN",
}


DEFAULTS = {
    "authenticated": False,
    "customer": None,
    "main_feature": "Banking Information",
    "assistant_open": False,
    "assistant_conversations": [],
    "subfeature_banking_information": "Overview",
    "subfeature_financial_intelligence": "Spending Analysis",
    "subfeature_scam_detection": "Message Analysis",
    "subfeature_customer_support": "Ask Support Assistant",
    "subfeature_account_opening_assistant": "Open New Account",
    "nav_history": [],
    "conversation_id": None,
    "conversation_messages": [],
    "loaded_conversation_id": None,
    "conversation_cursor": None,
    "conversation_has_more": False,
    "conversation_list": [],
    "conversation_list_cursor": None,
    "conversation_list_has_more": False,
    "last_audio_hash": None,
    "audio_widget_epoch": 0,
    "audio_cache": {},
    "audio_playback_visible": {},
    "response_mode": "text",
    "response_mode_explicit": False,
    "dock_response_mode_selector": "Text",
    "assistant_response_mode": "Text",
    "composer_menu_epoch": 0,
    "language_hint": "auto",
    "language_display": "Automatic",
    "response_language": None,
    "sidebar_assistant_language": "Automatic",
    "dock_copilot_language_selector": "Auto",
    "response_language_display": "Automatic",
    "session_validated_at": None,
    "last_api_error": None,
    "assistant_retry_payload": None,
    "assistant_failed_turn": None,
    "pending_scam_assessment_id": None,
    "pending_scam_conversation_id": None,
    "latest_scam_assessment": None,
    "scam_audio_epoch": 0,
    "last_scam_audio_hash": None,
    "support_action_outcomes": {},
    "support_selected_ticket_ref": None,
    "support_ticket_filters": {"status": "All statuses", "category": "All categories"},
    "support_ticket_items": None,
    "support_ticket_cursor": None,
    "support_ticket_has_more": False,
    "support_ticket_filter_key": None,
    "support_ticket_cache_customer_key": None,
    "transaction_risk_customer_key": None,
    "transaction_risk_transactions": None,
    "transaction_risk_tx_page": 0,
    "transaction_risk_selected_ref": None,
    "transaction_risk_latest_result": None,
    "transaction_risk_transactions_error": None,
    "transaction_risk_history_items": None,
    "transaction_risk_history_cursor": None,
    "transaction_risk_history_has_more": False,
    "transaction_risk_history_error": None,
    "transaction_risk_detail": None,
    "transaction_risk_detail_id": None,
    "transaction_risk_detail_error": None,
    "transaction_risk_follow_up_assessment_id": None,
    "transaction_risk_follow_up_conversation_id": None,
    "investigation_customer_key": None,
    "investigation_items": None,
    "investigation_cursor": None,
    "investigation_has_more": False,
    "investigation_error": None,
    "investigation_detail": None,
    "investigation_detail_ref": None,
    "investigation_action_outcomes": {},
    "investigation_filter_key": None,
    "investigation_status_filter": "All statuses",
    "investigation_category_filter": "All categories",
    "investigation_search": "",
    "investigation_evidence_items": None,
    "investigation_transaction_evidence_items": None,
    "account_application_step": None,
    "account_application_cache": None,
    "account_application_selected_ref": None,
    "account_application_last_submitted": None,
    "pending_navigation": None,
    "wake_word_enabled": False,
    "wake_recognition_mode": "on_device",
    "wake_browser_consent": False,
    "voice_component_response": None,
    "voice_component_event_id": None,
    "wake_activated": False,
}


def initialize_session_state(state, api_factory=VaaniShieldAPI):
    if "api_client" not in state:
        state["api_client"] = api_factory()
    for key, value in DEFAULTS.items():
        if key not in state:
            state[key] = value.copy() if isinstance(value, (dict, list)) else value

    # Ensure widget selector keys align with canonical response_language
    if state.get("response_mode") not in RESPONSE_MODE_LABELS:
        state["response_mode"] = "text"
    expected_mode = RESPONSE_MODE_LABELS[state["response_mode"]]
    if state.get("dock_response_mode_selector") not in RESPONSE_MODE_CODES:
        state["dock_response_mode_selector"] = expected_mode
    if state.get("assistant_response_mode") not in RESPONSE_MODE_CODES:
        state["assistant_response_mode"] = expected_mode

    canon = state.get("response_language")
    if canon in CODE_TO_FULL_LABEL:
        expected_full = CODE_TO_FULL_LABEL[canon]
        expected_short = CODE_TO_SHORT_LABEL[canon]
        if "sidebar_assistant_language" not in state or state.get("sidebar_assistant_language") not in RESPONSE_LANGUAGE_OPTIONS_FULL:
            state["sidebar_assistant_language"] = expected_full
        if "dock_copilot_language_selector" not in state or state.get("dock_copilot_language_selector") not in RESPONSE_LANGUAGE_OPTIONS_SHORT:
            state["dock_copilot_language_selector"] = expected_short
        if "response_language_display" not in state or state.get("response_language_display") not in RESPONSE_LANGUAGE_OPTIONS_FULL:
            state["response_language_display"] = expected_full


def clear_pending_scam_assessment(state):
    state["pending_scam_assessment_id"] = None
    state["pending_scam_conversation_id"] = None


def clear_support_ticket_cache(state):
    state["support_selected_ticket_ref"] = None
    state["support_ticket_items"] = None
    state["support_ticket_cursor"] = None
    state["support_ticket_has_more"] = False
    state["support_ticket_filter_key"] = None
    state["support_ticket_cache_customer_key"] = None


def clear_transaction_risk_cache(state):
    """Drop customer-scoped transaction data and pending assistant handoff state."""
    for key, value in DEFAULTS.items():
        if key.startswith("transaction_risk_"):
            state[key] = value.copy() if isinstance(value, (dict, list)) else value


def clear_investigation_cache(state):
    for key, value in DEFAULTS.items():
        if key.startswith("investigation_"):
            state[key] = value.copy() if isinstance(value, (dict, list)) else value


def clear_account_application_cache(state):
    for key, value in DEFAULTS.items():
        if key.startswith("account_application_"):
            state[key] = value.copy() if isinstance(value, (dict, list)) else value
    state["pending_navigation"] = None


def bind_pending_scam_assessment(state, assessment_id, conversation_id):
    state["pending_scam_assessment_id"] = str(assessment_id) if assessment_id else None
    state["pending_scam_conversation_id"] = str(conversation_id) if assessment_id and conversation_id else None


def reset_conversation_state(state):
    for key in list(state):
        if key.startswith(("kyc_chat_", "chat_application_", "account_chat_", "account_review_for_", "voice_component_", "active_voice_component_")):
            if key in {"voice_component_processed_ids", "voice_component_event_id"}:
                continue
            state.pop(key, None)
    state["wake_word_enabled"] = False
    state.pop("voice_retry_audio", None)
    state["conversation_id"] = None
    state["conversation_messages"] = []
    state["loaded_conversation_id"] = None
    state["conversation_cursor"] = None
    state["conversation_has_more"] = False
    state["audio_cache"] = {}
    state["audio_playback_visible"] = {}
    state["assistant_retry_payload"] = None
    state["assistant_failed_turn"] = None
    state["support_action_outcomes"] = {}
    clear_pending_scam_assessment(state)
    state["investigation_action_outcomes"] = {}


def sign_out(state, api_factory=VaaniShieldAPI):
    old_client = state.get("api_client")
    if old_client:
        old_client.close()
    state["api_client"] = api_factory()
    state["authenticated"] = False
    state["customer"] = None
    state["session_validated_at"] = None
    state["latest_scam_assessment"] = None
    state["last_scam_audio_hash"] = None
    state["conversation_list"] = []
    state["conversation_list_cursor"] = None
    state["conversation_list_has_more"] = False
    reset_conversation_state(state)
    state.pop("voice_component_processed_ids", None)
    state.pop("voice_component_event_id", None)
    state.pop("voice_diagnostics", None)
    clear_support_ticket_cache(state)
    clear_transaction_risk_cache(state)
    clear_investigation_cache(state)
    clear_account_application_cache(state)


def mark_session_valid(state, customer):
    previous = state.get("customer") or {}
    if previous.get("customer_key") != customer.get("customer_key"):
        clear_support_ticket_cache(state)
        clear_transaction_risk_cache(state)
        clear_investigation_cache(state)
        clear_account_application_cache(state)
        reset_conversation_state(state)
    state["authenticated"] = True
    state["customer"] = customer
    state["session_validated_at"] = datetime.now(timezone.utc)


def session_validation_due(state, max_age_seconds=120):
    validated = state.get("session_validated_at")
    if not validated:
        return True
    age = (datetime.now(timezone.utc) - validated).total_seconds()
    return age >= max_age_seconds


def on_sidebar_language_change():
    """Callback triggered by Streamlit when the sidebar response language changes."""
    selected = st.session_state.get("sidebar_assistant_language", "Automatic")
    code = FULL_LABEL_TO_CODE.get(selected)
    st.session_state["response_language"] = code
    dock_label = CODE_TO_SHORT_LABEL.get(code, "Auto")
    try:
        st.session_state["dock_copilot_language_selector"] = dock_label
    except (StreamlitWidgetAlreadyInstantiatedError, Exception):
        pass
    try:
        st.session_state["response_language_display"] = selected
    except (StreamlitWidgetAlreadyInstantiatedError, Exception):
        pass


def on_dock_language_change():
    """Callback triggered by Streamlit when the dock copilot response language changes."""
    selected = st.session_state.get("dock_copilot_language_selector", "Auto")
    code = SHORT_LABEL_TO_CODE.get(selected)
    st.session_state["response_language"] = code
    full_label = CODE_TO_FULL_LABEL.get(code, "Automatic")
    try:
        st.session_state["sidebar_assistant_language"] = full_label
    except (StreamlitWidgetAlreadyInstantiatedError, Exception):
        pass
    try:
        st.session_state["response_language_display"] = full_label
    except (StreamlitWidgetAlreadyInstantiatedError, Exception):
        pass


def on_composer_language_change():
    """Callback triggered by Streamlit when the voice composer response language changes."""
    selected = st.session_state.get("response_language_display", "Automatic")
    code = FULL_LABEL_TO_CODE.get(selected)
    st.session_state["response_language"] = code
    full_label = CODE_TO_FULL_LABEL.get(code, "Automatic")
    dock_label = CODE_TO_SHORT_LABEL.get(code, "Auto")
    try:
        st.session_state["sidebar_assistant_language"] = full_label
    except (StreamlitWidgetAlreadyInstantiatedError, Exception):
        pass
    try:
        st.session_state["dock_copilot_language_selector"] = dock_label
    except (StreamlitWidgetAlreadyInstantiatedError, Exception):
        pass


def sync_response_language(state, language_code: str | None):
    """Keep response language and its UI widget selectboxes synchronized across feature views.

    Updates the canonical `response_language` state and safely propagates
    corresponding labels to widget-backed keys. Catches StreamlitWidgetAlreadyInstantiatedError
    to ensure calling this function never crashes the Streamlit execution lifecycle.
    """
    state["response_language"] = language_code
    dock_label = CODE_TO_SHORT_LABEL.get(language_code, "Auto")
    full_label = CODE_TO_FULL_LABEL.get(language_code, "Automatic")
    for key, val in (
        ("sidebar_assistant_language", full_label),
        ("dock_copilot_language_selector", dock_label),
        ("response_language_display", full_label),
    ):
        try:
            state[key] = val
        except (StreamlitWidgetAlreadyInstantiatedError, Exception) as exc:
            # StreamlitWidgetAlreadyInstantiatedError cannot be reassigned once instantiated.
            # Catching here protects against mid-run programmatic mutations.
            if isinstance(exc, StreamlitWidgetAlreadyInstantiatedError) or "cannot be modified after the widget with that key is instantiated" in str(exc):
                pass
            else:
                raise


def _sync_response_mode(selected, source_key):
    code = RESPONSE_MODE_CODES.get(selected, "text")
    st.session_state["response_mode"] = code
    st.session_state["response_mode_explicit"] = True
    for key in ("dock_response_mode_selector", "assistant_response_mode"):
        if key == source_key:
            continue
        try:
            st.session_state[key] = RESPONSE_MODE_LABELS[code]
        except StreamlitWidgetAlreadyInstantiatedError:
            pass


def on_dock_response_mode_change():
    _sync_response_mode(
        st.session_state.get("dock_response_mode_selector", "Text"),
        "dock_response_mode_selector",
    )


def on_assistant_response_mode_change():
    _sync_response_mode(
        st.session_state.get("assistant_response_mode", "Text"),
        "assistant_response_mode",
    )

