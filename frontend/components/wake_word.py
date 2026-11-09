"""Hands-free consent guidance; the compact composer owns the microphone lifecycle."""


def render_wake_word_listener(st, state):
    if state.get("wake_word_enabled", False):
        st.caption(
            'Use Start hands-free beside the composer to allow "Hey Vaani" listening. '
            'Choose a recognition mode in Voice Wake-Word settings. '
            'Listening stops when this tab is hidden. Stop / mute releases the microphone.'
        )
