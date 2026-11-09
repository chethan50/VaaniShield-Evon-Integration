"""Unit tests for Axis Bank-inspired banking UI refinements and continuous Hey Vaani voice interaction."""

import unittest
from pathlib import Path

from frontend.components.fintech_ui import (
    format_inr,
    render_balance_hero_html,
    render_transaction_card_html,
)
from frontend.components.wake_word import render_wake_word_listener


class TestApprovedPurpleBankingUI(unittest.TestCase):
    def setUp(self):
        self.css_path = Path(__file__).parent / "assets" / "styles.css"
        self.assertTrue(self.css_path.exists(), "styles.css must exist")
        self.css = self.css_path.read_text(encoding="utf-8")

    def test_approved_purple_and_white_tokens(self):
        """Verify the approved VaaniShield lavender, purple, and white design tokens."""
        self.assertIn("--primary-violet: #5B3FC4", self.css)
        self.assertIn("--primary-violet-dark: #472E9E", self.css)
        self.assertIn("--light-lavender: #EEE9FF", self.css)
        self.assertIn("--bg-card: #FFFFFF", self.css)
        self.assertIn("--border-light: #E7E0FF", self.css)
        # Risk and transaction status colors remain semantic.
        self.assertIn("--status-emerald: #059669", self.css)
        self.assertIn("--status-coral: #DC2626", self.css)

    def test_balance_hero_responsive_and_clean_design(self):
        """Verify balance hero uses responsive typography and executive styling."""
        html = render_balance_hero_html(
            balance=124500.50,
            account_number="VS1000001234",
            customer_name="Arjun Sharma",
            is_visible=True,
            monthly_inflow=85000.0,
        )
        self.assertIn("VAANISHIELD SECURE BANKING", html)
        self.assertIn("₹1,24,500.50", html)
        self.assertIn("Arjun Sharma", html)
        self.assertIn("RuPay", html)

        # CSS checks for hero balance text wrapping and font scaling
        self.assertIn(".hero-balance-amount", self.css)
        self.assertIn("clamp(1.75rem", self.css)
        self.assertIn("word-break: break-word", self.css)

    def test_responsive_workspace_overflow_and_metrics_protection(self):
        """Verify responsive workspace prevents clipping and squeezes on open copilot."""
        self.assertIn(".st-key-banking-workspace-copilot-open", self.css)
        self.assertIn("calc(100% - var(--copilot-dock-reserve))", self.css)

        # Metric value & label wrapping
        self.assertIn(":is(.st-key-banking-workspace-copilot-open,.st-key-banking-workspace-copilot-closed) div[data-testid=\"stMetricValue\"]", self.css)
        self.assertIn("overflow-wrap: anywhere", self.css)
        self.assertIn(":is(.st-key-banking-workspace-copilot-open,.st-key-banking-workspace-copilot-closed) div[data-testid=\"stMetricLabel\"]", self.css)

        # Cards and columns reflow
        self.assertIn(":is(.st-key-banking-workspace-copilot-open,.st-key-banking-workspace-copilot-closed) div[data-testid=\"stHorizontalBlock\"]", self.css)
        self.assertIn("flex-wrap: wrap !important", self.css)


class TestContinuousVoiceInteraction(unittest.TestCase):
    def test_executable_voice_lifecycle(self):
        import shutil
        import subprocess
        if not shutil.which("node"):
            self.skipTest("Node.js is not installed")
        result = subprocess.run(["node", "--test", "frontend/test_voice_lifecycle.js"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


class TestTargetedFrontendRefinements(unittest.TestCase):
    def setUp(self):
        self.css_path = Path(__file__).parent / "assets" / "styles.css"
        self.assertTrue(self.css_path.exists())
        self.css = self.css_path.read_text(encoding="utf-8")

    def test_issue_1_content_overflow_and_metric_wrapping_all_tabs(self):
        """Verify metric cards, card layouts, and values wrap cleanly without clipping across all tabs."""
        # Cards page virtual card & limits reflow
        self.assertIn(":is(.st-key-banking-workspace-copilot-open,.st-key-banking-workspace-copilot-closed) div[data-testid=\"stHorizontalBlock\"]:has(.acct-status-pill)", self.css)
        self.assertIn("flex: 1 1 100% !important", self.css)

        # Overview page stacking
        self.assertIn(":is(.st-key-banking-workspace-copilot-open,.st-key-banking-workspace-copilot-closed) div[data-testid=\"stHorizontalBlock\"]:has(.acct-summary-item)", self.css)

        # Metric value & label wrapping
        self.assertIn(":is(.st-key-banking-workspace-copilot-open,.st-key-banking-workspace-copilot-closed) div[data-testid=\"stMetricValue\"]", self.css)
        self.assertIn("white-space: normal !important", self.css)
        self.assertIn("word-break: break-word !important", self.css)
        self.assertIn("overflow-wrap: anywhere !important", self.css)

        # Global metric value override also permits wrapping (no white-space: nowrap)
        self.assertIn("div[data-testid=\"stMetricValue\"]", self.css)

        # Loans detail grid and header reflow
        self.assertIn(":is(.st-key-banking-workspace-copilot-open,.st-key-banking-workspace-copilot-closed) .loan-card-header", self.css)
        self.assertIn(":is(.st-key-banking-workspace-copilot-open,.st-key-banking-workspace-copilot-closed) .loan-details-grid", self.css)
        self.assertIn("repeat(2, minmax(130px, 1fr))", self.css)

        # Transactions filters reflow
        self.assertIn(":is(.st-key-banking-workspace-copilot-open,.st-key-banking-workspace-copilot-closed) form div[data-testid=\"column\"]", self.css)

    def test_issue_2_typography_hierarchy_main_features_and_subfeatures(self):
        """Verify Level 2 main feature buttons are bold 700/800 and Level 3 sidebar items are semibold 600/700."""
        # Level 2 Horizontal Nav typography
        self.assertIn(":is(.st-key-main-nav-container-copilot-open,.st-key-main-nav-container) div[data-testid=\"column\"] button", self.css)
        self.assertIn("font-weight: 700 !important", self.css)
        self.assertIn(":is(.st-key-main-nav-container-copilot-open,.st-key-main-nav-container) div[data-testid=\"column\"] button[kind=\"primary\"]", self.css)
        self.assertIn("font-weight: 800 !important", self.css)

        # Level 3 Sidebar Subfeatures typography
        self.assertIn("[data-testid=\"stSidebar\"] div.stButton > button", self.css)
        self.assertIn("font-weight: 600 !important", self.css)
        self.assertIn("[data-testid=\"stSidebar\"] div.stButton > button[kind=\"primary\"]", self.css)
        self.assertIn("color: #97144D !important", self.css)

    def test_issue_3_popover_dropdown_layering_and_z_index(self):
        """Verify sidebar popovers render above dashboard content without clipping."""
        # Sidebar elevated z-index and overflow visible
        self.assertIn("[data-testid=\"stSidebar\"]", self.css)
        self.assertIn("z-index: 1005 !important", self.css)
        self.assertIn("overflow: visible !important", self.css)

        # BaseWeb popover layers and bodies have highest z-index
        self.assertIn("div[data-baseweb=\"popover\"]", self.css)
        self.assertIn("z-index: 999999 !important", self.css)
        self.assertIn("[data-testid=\"stSidebar\"] div[data-testid=\"stPopoverBody\"]", self.css)

    def test_sidebar_single_button_surface_no_nested_borders_or_backgrounds(self):
        """Verify sidebar navigation buttons have exactly ONE surface and reset nested descendants."""
        # Button container has padding & surface
        self.assertIn("[data-testid=\"stSidebar\"] div.stButton > button", self.css)
        self.assertIn("padding: 0.55rem 0.85rem !important", self.css)

        # Inner structural elements explicitly reset to transparent with no borders
        self.assertIn("[data-testid=\"stSidebar\"] div.stButton > button *", self.css)
        self.assertIn("background: transparent !important", self.css)
        self.assertIn("border: none !important", self.css)
        self.assertIn("box-shadow: none !important", self.css)

        # Popover trigger buttons also reset inner elements
        self.assertIn("[data-testid=\"stSidebar\"] div[data-testid=\"stPopover\"] > button *", self.css)

    def test_css_syntax_balance_and_no_orphaned_rules(self):
        """Verify CSS has perfectly balanced braces, comments, and zero orphaned rules."""
        open_b = self.css.count("{")
        close_b = self.css.count("}")
        self.assertEqual(open_b, close_b, f"CSS braces must be balanced: {open_b} open vs {close_b} close")

        open_c = self.css.count("/*")
        close_c = self.css.count("*/")
        self.assertEqual(open_c, close_c, f"CSS comments must be balanced: {open_c} open vs {close_c} close")

        # Verify orphaned declarations were repaired
        self.assertNotIn("  background: var(--primary-violet-gradient) !important;\n  border: 1px solid var(--primary-violet);\n", self.css)
        self.assertNotIn("  background: #E0E7FF !important;\n  color: var(--primary-violet-dark) !important;\n}", self.css)

    def test_voice_component_uses_bounded_wav_and_no_browser_tts(self):
        code = (Path(__file__).parent / "components/voice_capture/voice.js").read_text(encoding="utf-8")
        self.assertIn("processLocally = true", code)
        self.assertIn("30000", code)
        self.assertIn("generation !== this.generation", code)
        self.assertNotIn("speechSynthesis", code)

    def test_copilot_independent_conversation_scrolling_layout(self):
        """Verify Copilot conversation history scrolls independently with pinned header and sticky composer."""
        # Parent panel fixed and non-scrolling
        self.assertIn(".st-key-floating-assistant-panel", self.css)
        self.assertIn("overflow: hidden !important", self.css)
        self.assertIn("min-height: 0 !important", self.css)

        # Streamlit stLayoutWrapper around scroll area constrained with flex: 1 1 0% and min-height: 0
        self.assertIn(".st-key-floating-assistant-panel > div[data-testid=\"stLayoutWrapper\"]:has(> .st-key-copilot-chat-scroll-area)", self.css)
        self.assertIn("flex: 1 1 0% !important", self.css)

        # Scroll area has independent vertical scrollbar and overscroll containment
        self.assertIn(".st-key-copilot-chat-scroll-area", self.css)
        self.assertIn("overflow-y: auto !important", self.css)
        self.assertIn("overscroll-behavior: contain !important", self.css)

        # Child messages inside scroll area do not flex-shrink
        self.assertIn(".st-key-copilot-chat-scroll-area > div[data-testid=\"stLayoutWrapper\"]", self.css)
        self.assertIn("flex-shrink: 0 !important", self.css)

        # Smart scroll preservation across reruns in JS
        wake_path = Path(__file__).parent / "components/voice_capture/voice.js"
        wake_code = wake_path.read_text(encoding="utf-8")
        self.assertIn("setupCopilotScrollAndComposer", wake_code)
        self.assertIn("userScrolledUp", wake_code)


if __name__ == "__main__":
    unittest.main()



