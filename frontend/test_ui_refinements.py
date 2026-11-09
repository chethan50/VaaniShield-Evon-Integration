"""Unit tests for VaaniShield UI and Assistant Refinements.

Covers:
1. CSS contrast and visibility rules for user and assistant chat messages.
2. Copilot toggle button permanent visibility and contrast rules.
3. Streamlit base light theme configuration.
4. Metric cards ellipsis-clipping prevention and responsive sizing.
5. Wake-word script error handling and exact welcome phrase.
"""

import unittest
from pathlib import Path
import tomllib


class TestUIRefinements(unittest.TestCase):
    """Test suite verifying UI and styling invariants."""

    @classmethod
    def setUpClass(cls):
        styles_path = Path("frontend/assets/styles.css")
        cls.css_content = styles_path.read_text(encoding="utf-8")
        config_path = Path(".streamlit/config.toml")
        cls.config_content = config_path.read_text(encoding="utf-8")
        cls.config_toml = tomllib.loads(cls.config_content)
        wake_path = Path("frontend/components/wake_word.py")
        cls.wake_content = wake_path.read_text(encoding="utf-8")

    def test_streamlit_theme_base_light(self):
        """Verify .streamlit/config.toml sets base light theme to prevent OS dark mode text color injection."""
        self.assertIn("theme", self.config_toml)
        theme = self.config_toml["theme"]
        self.assertEqual(theme.get("base"), "light")
        self.assertEqual(theme.get("textColor"), "#172033")
        self.assertEqual(theme.get("backgroundColor"), "#F8FAFD")

    def test_chat_message_contrast_rules(self):
        """Verify chat messages have explicit dark text colors and high contrast rules."""
        css = self.css_content

        # Universal message bubble background and text
        self.assertIn('[data-testid="stChatMessage"]', css)
        self.assertIn('[data-testid="stChatMessageContent"] p', css)
        self.assertIn('color: #172033 !important', css)

        # Headings, bold, and code blocks
        self.assertIn('[data-testid="stChatMessage"] strong', css)
        self.assertIn('[data-testid="stChatMessage"] code', css)
        self.assertIn('[data-testid="stChatMessage"] table', css)
        self.assertIn('[data-testid="stChatMessage"] th', css)
        self.assertIn('[data-testid="stChatMessage"] td', css)

        # User message specific bubble
        self.assertIn('[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"])', css)
        self.assertIn('background: #EEF2FF !important', css)
        self.assertIn('color: #1E1B4B !important', css)

        # Metadata & captions
        self.assertIn('[data-testid="stChatMessage"] [data-testid="stCaptionContainer"]', css)

    def test_copilot_toggle_button_permanent_visibility(self):
        """Verify the Copilot button in top header has permanent contrasting colors."""
        css = self.css_content

        self.assertIn('.st-key-header-copilot-toggle', css)
        self.assertIn('background: #EEF2FF !important', css)
        self.assertIn('color: #4F46E5 !important', css)
        self.assertIn('border: 1.5px solid #C7D2FE !important', css)
        self.assertIn('.st-key-header-copilot-toggle > button:hover', css)

    def test_metric_card_ellipsis_prevention(self):
        """Verify metric cards have responsive clamp font size and visible overflow."""
        css = self.css_content

        self.assertIn('div[data-testid="stMetricValue"]', css)
        self.assertIn('clamp(1.15rem, 2vw, 1.45rem) !important', css)
        self.assertIn('overflow: visible !important', css)
        self.assertIn('div[data-testid="stMetricLabel"]', css)

    def test_wake_word_script_and_phrase(self):
        """Verify wake-word component contains exact phrase and permission protection."""
        content = self.wake_content

        self.assertIn('"Hey Vaani"', content)
        self.assertIn('tab is hidden', content)
        script = Path("frontend/components/voice_capture/voice.js").read_text(encoding="utf-8")
        self.assertIn("not-allowed", script)
        self.assertIn("this.stop();", script)
        self.assertIn("processLocally = true", script)



if __name__ == "__main__":
    unittest.main()
