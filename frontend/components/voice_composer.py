"""Voice and text input controls for the assistant page."""

from __future__ import annotations

import base64
import io
from pathlib import Path
import streamlit.components.v1 as components

_capture = components.declare_component("vaani_voice_capture", path=str(Path(__file__).with_name("voice_capture")))


def render_compact_voice(state):
    return _capture(
        hands_free_allowed=bool(state.get("wake_word_enabled", False)),
        recognition_mode=state.get("wake_recognition_mode", "on_device"),
        browser_consent=bool(state.get("wake_browser_consent", False)),
        scope=str(state.get("conversation_id") or ""),
        response=state.get("voice_component_response"),
        key="vaani-compact-microphone", default=None,
    )


def decode_voice_event(event):
    """Bound component input before handing it to the authenticated upload path."""
    if not isinstance(event, dict) or event.get("kind") != "audio":
        raise ValueError("Invalid voice recording.")
    encoded = event.get("audio")
    if not isinstance(encoded, str) or len(encoded) > 1_300_000:
        raise ValueError("Recording exceeds the 30 second limit.")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, base64.binascii.Error):
        raise ValueError("Invalid voice recording.") from None
    if not raw.startswith(b"RIFF") or raw[8:12] != b"WAVE":
        raise ValueError("A WAV recording is required.")
    audio = io.BytesIO(raw)
    audio.name, audio.type = "question.wav", "audio/wav"
    return audio


def render_voice_input(st, epoch: int, *, key_prefix="assistant"):
    """Return a newly captured WAV payload, or None until the user records."""
    return st.audio_input(
        "Record voice message",
        sample_rate=16000,
        key=f"{key_prefix}-audio-{epoch}",
        help="Record a short question in Kannada, English, Hindi, or a mix.",
        label_visibility="collapsed",
    )


def render_language_controls(st, state):
    col_lang1, col_lang2 = st.columns(2)
    with col_lang1:
        st.selectbox(
            "Spoken language",
            ["Automatic", "Kannada", "English", "Hindi"],
            index={"auto": 0, "kn-IN": 1, "en-IN": 2, "hi-IN": 3}.get(state.get("language_hint"), 0),
            key="language_display",
            help="Automatic detection supports code-mixed speech where the provider can identify it.",
        )
        state["language_hint"] = {
            "Automatic": "auto", "Kannada": "kn-IN", "English": "en-IN", "Hindi": "hi-IN"
        }[state["language_display"]]

    with col_lang2:
        from frontend.session_state import (
            RESPONSE_LANGUAGE_OPTIONS_FULL,
            on_composer_language_change,
        )
        st.selectbox(
            "Response language",
            RESPONSE_LANGUAGE_OPTIONS_FULL,
            key="response_language_display",
            on_change=on_composer_language_change,
            help="Language in which Vaani AI will compose answers and generate voice output.",
        )

