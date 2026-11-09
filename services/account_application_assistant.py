"""Deterministic Account Opening Assistant turns for Vaani AI.

Account-opening intent is recognised by server-side rules, every action runs
through the approved backend dispatcher (``execute_tool``), and the reply is
rendered from the structured tool results. No language model is called for
these turns, so no application detail, identity document or personal field
ever leaves the backend, and the model cannot decide that an application is
complete or submitted. Text and speech share this path: speech arrives here
already transcribed.
"""

import logging
import re
from uuid import uuid4

from backend.api_models import ToolExecutionResult
from services.account_application_service import (
    ApplicationError,
    ApplicationNotFound,
    ApplicationValidationError,
    has_active_application,
)
from services import account_application_service as applications

logger = logging.getLogger(__name__)

SUPPORTED_LANGUAGES = ("en-IN", "hi-IN", "kn-IN")

# --------------------------------------------------------------------------
# Intent classification
# --------------------------------------------------------------------------

_GLOBAL_BLOCK = re.compile(
    r"\b(?:tickets?|complaints?|investigat\w*|scam\w*|fraud\w*|otp|phishing|suspicious|"
    r"loans?|emi|clos(?:e|ing)|block\w*|delete)\b"
)
_GLOBAL_BLOCK_NATIVE = (
    "शिकायत", "टिकट", "धोखा", "ठगी", "स्कैम", "लोन", "ऋण", "बंद",
    "ದೂರು", "ಟಿಕೆಟ್", "ವಂಚನೆ", "ಮೋಸ", "ಸಾಲ", "ಮುಚ್ಚ",
)
_OPEN_BLOCK = re.compile(r"\b(?:statement|balance|passbook|transactions?|history|number)\b")
_OPEN_BLOCK_NATIVE = ("शेष", "बैलेंस", "ಬ್ಯಾಲೆನ್ಸ್")

_FILLER = (
    r"(?:a|an|the|my|new|another|additional|more|one|bank|savings?|salary|current|"
    r"joint|digital|me|for|with|to|help|please|also)"
)
_OPEN_EN = (
    re.compile(
        r"\b(?:open|opening|create|creating|apply|applying|setup|set\s+up|setting\s+up|start|starting)\b"
        r"(?:\s+" + _FILLER + r")*\s+accounts?\b"
    ),
    re.compile(r"\b(?:new|another|additional)\s+(?:(?:bank|savings?|salary|current|joint|digital)\s+)*accounts?\b"),
    re.compile(r"\baccount\s+(?:opening|application)\b"),
    re.compile(r"\bapply\s+for\s+(?:a|an)\s+(?:savings?|salary|current)\b"),
)
_STATUS_EN = re.compile(
    r"\b(?:status|track|tracking|progress|update|updates|where)\b[^.?!]{0,50}\bapplication\b|"
    r"\bapplication\b[^.?!]{0,40}\b(?:status|progress|approved|submitted|stands?)\b"
)
_ACCOUNT_WORD = re.compile(r"\b(?:account|accounts|savings?|salary|current|kyc|bank)\b")
_OPTIONS_EN = re.compile(
    r"\b(?:what|which)\b[^.?!]{0,30}\b(?:types?|kinds?|options?)\b[^.?!]{0,30}\baccounts?\b"
    r"[^.?!]{0,30}\b(?:open|offer|available|choose)\b|"
    r"\b(?:types?|kinds?)\s+of\s+accounts?\s+(?:can\s+i\s+open|do\s+you\s+offer|are\s+available|available)\b|"
    r"\baccount\s+options\b"
)
_REMAINING_EN = re.compile(
    r"\bwhat(?:'s|\s+is|\s+else)?\s*(?:remaining|left|missing|pending|next)\b|"
    r"\b(?:remaining|anything\s+missing|anything\s+else\s+(?:needed|required))\b|"
    r"\bwhat\s+(?:do|should)\s+i\s+(?:do|need)(?:\s+next)?\b|\bnext\s+step\b|\bwhat\s+next\b|"
    r"\bhow\s+much\s+(?:is\s+)?left\b"
)
_REMAINING_BLOCK = re.compile(r"\b(?:balance|budget|tenure|spend\w*|expenses?)\b")
_UPLOAD_EN = re.compile(
    r"\b(?:upload(?:ed)?|attach(?:ed)?)\b[^.?!]{0,50}\b(?:aadhaar|aadhar|pan|kyc|documents?|address\s+proof|proof)\b|"
    r"\b(?:aadhaar|aadhar|pan(?:\s+card)?|kyc|documents?|address\s+proof)\b[^.?!]{0,50}\b(?:upload(?:ed)?|attach(?:ed)?|where)\b|"
    r"\bwhere\b[^.?!]{0,40}\b(?:upload|submit\s+my\s+(?:document|aadhaar|pan))|"
    r"\b(?:check|status|verified|verify|missing|received|complete|ready)\b[^.?!]{0,50}\b(?:documents?|kyc|aadhaar|pan|address\s+proof)\b|"
    r"\b(?:documents?|kyc|aadhaar|pan|address\s+proof)\b[^.?!]{0,50}\b(?:status|verified|verify|missing|received|complete|ready)\b"
)
_REVIEW_EN = re.compile(
    r"\bsubmit\b|\breview\b[^.?!]{0,30}\b(?:application|details|summary)\b|"
    r"\b(?:finish|complete|finali[sz]e)\b[^.?!]{0,25}\b(?:application|account|process)\b|"
    r"\bready\s+to\s+(?:submit|apply)\b"
)
_LATER_EN = re.compile(
    r"\b(?:finish|continue|complete|resume|come\s+back|do\s+(?:this|it))\b[^.?!]{0,25}\blater\b|"
    r"\bsave\s+(?:my\s+)?(?:progress|application|draft)\b|\b(?:pause|stop)\s+(?:for\s+now|here)\b"
)
_EMPLOYMENT_EN = (
    ("GOVERNMENT_EMPLOYEE", re.compile(r"\bgovernment\s+(?:employee|job|servant)\b")),
    ("SELF_EMPLOYED", re.compile(r"\bself[\s-]?employed\b|\bfreelanc\w+")),
    ("BUSINESS_OWNER", re.compile(r"\bbusiness\s+owner\b|\bi\s+(?:own|run)\s+a\s+business\b")),
    ("SALARIED", re.compile(r"\bsalaried\b")),
    ("STUDENT", re.compile(r"\bstudent\b")),
    ("RETIRED", re.compile(r"\bretired\b")),
)
_TYPE_PATTERNS = (
    ("SAVINGS", re.compile(r"\bsavings?\b|बचत|ಉಳಿತಾಯ")),
    ("SALARY", re.compile(r"\bsalary\b|वेतन|सैलरी|ವೇತನ")),
    ("CURRENT", re.compile(r"\bcurrent\s+account\b|\bcurrent\b(?=\s*$)|चालू|ಚಾಲ್ತಿ")),
)
_SELECTION_WORDS = frozenset({
    "i", "want", "would", "like", "to", "choose", "select", "pick", "prefer", "a", "an",
    "the", "account", "please", "lets", "let's", "go", "with", "make", "it", "do", "ok",
    "okay", "yes", "then", "take", "need", "new", "खाता", "खाते", "चाहिए", "चुनें",
    "चुनना", "चाहता", "चाहती", "हूँ", "हूं", "मुझे", "ಖಾತೆ", "ಬೇಕು", "ಆರಿಸಿ", "ನನಗೆ",
    "ಬಯಸುತ್ತೇನೆ",
})
_EMPLOYMENT_WORDS = frozenset({
    "i", "am", "i'm", "im", "a", "an", "the", "my", "employment", "category", "is", "it's",
    "its", "work", "as", "are", "currently", "please", "job", "status", "self", "employed",
    "self-employed", "government", "employee", "servant", "business", "owner", "student",
    "retired", "freelancer", "freelance", "salaried",
})
_TOKEN_SPLIT = re.compile(r"[\s,.;:!?\"()\[\]।]+")


def _mentioned_type(normalized):
    found = [key for key, pattern in _TYPE_PATTERNS if pattern.search(normalized)]
    return found[0] if len(found) == 1 else None


def classify_account_intent(text):
    """Return ``{"action", "needs", "args"}`` or None for non-account turns.

    ``needs`` is None, ``"active"`` (an unfinished application must exist) or
    ``"bare"`` (a status question without an explicit account word).
    """
    if not isinstance(text, str) or not text.strip() or len(text) > 600:
        return None
    normalized = text.casefold()
    if _GLOBAL_BLOCK.search(normalized) or any(term in normalized for term in _GLOBAL_BLOCK_NATIVE):
        return None
    account_type = _mentioned_type(normalized)
    type_args = {"account_type": account_type} if account_type else {}

    native_account = any(word in normalized for word in ("खाता", "खाते", "ಖಾತೆ"))
    if (
        _STATUS_EN.search(normalized)
        or (("आवेदन" in normalized) and any(w in normalized for w in ("स्थिति", "कहां", "कहाँ")))
        or (("ಅರ್ಜಿ" in normalized) and any(w in normalized for w in ("ಸ್ಥಿತಿ", "ಎಲ್ಲಿ")))
    ):
        has_account_word = bool(_ACCOUNT_WORD.search(normalized)) or native_account
        return {
            "action": "STATUS",
            "needs": None if has_account_word else "bare",
            "args": type_args,
        }

    open_block = _OPEN_BLOCK.search(normalized) or any(term in normalized for term in _OPEN_BLOCK_NATIVE)
    if not open_block:
        english_open = any(pattern.search(normalized) for pattern in _OPEN_EN)
        hindi_open = native_account and any(
            word in normalized for word in ("खोल", "खुलव", "बनाना", "बनाएं", "बनवा", "आवेदन")
        )
        kannada_open = "ಖಾತೆ" in normalized and any(word in normalized for word in ("ತೆರೆ", "ಅರ್ಜಿ"))
        if english_open or hindi_open or kannada_open:
            return {"action": "OPEN", "needs": None, "args": type_args}

    if _OPTIONS_EN.search(normalized):
        return {"action": "OPTIONS", "needs": None, "args": {}}

    # The remaining intents only make sense while a draft exists.
    if _LATER_EN.search(normalized) or (
        any(word in normalized for word in ("बाद में", "बाद मे", "ನಂತರ")) and
        any(word in normalized for word in ("पूरा", "जारी", "सेव", "ಪೂರ್ಣ", "ಮುಂದುವರಿ", "ಉಳಿಸ"))
    ):
        return {"action": "SAVE_LATER", "needs": "active", "args": {}}
    if _UPLOAD_EN.search(normalized) or (
        any(word in normalized for word in ("अपलोड", "ಅಪ್ಲೋಡ್", "ಅಪ್\u200cಲೋಡ್"))
        and any(word in normalized for word in ("आधार", "पैन", "दस्तावेज", "ಆಧಾರ್", "ಪ್ಯಾನ್", "ದಾಖಲೆ"))
    ):
        return {"action": "UPLOAD_HELP", "needs": None, "args": {}}
    if re.search(r"\b(?:i\s+(?:have|got|sent|provided)|my|all\s+(?:my|the))\b[^.?!]{0,35}\b(?:documents?|kyc|aadhaar|pan|address\s+proof)\b", normalized):
        return {"action": "UPLOAD_HELP", "needs": "active", "args": {}}
    if _REVIEW_EN.search(normalized) or any(
        word in normalized for word in ("सबमिट", "जमा करें", "जमा कर", "समीक्षा", "ಸಲ್ಲಿಸ", "ಸಬ್ಮಿಟ್", "ಪರಿಶೀಲಿಸ")
    ):
        return {"action": "REVIEW", "needs": "active", "args": {}}
    if (_REMAINING_EN.search(normalized) and not _REMAINING_BLOCK.search(normalized)) or any(
        phrase in normalized for phrase in (
            "क्या बाकी", "आगे क्या", "अगला चरण", "ಏನು ಬಾಕಿ", "ಮುಂದೆ ಏನು", "ಮುಂದಿನ ಹಂತ",
        )
    ):
        return {"action": "REMAINING", "needs": "active", "args": {}}

    tokens = [token for token in _TOKEN_SPLIT.split(normalized) if token]
    # Short, selection-only replies while a draft exists. Requiring every word
    # to be a selection word keeps "show my savings" and similar questions
    # about existing finances from being read as a choice.
    if account_type and tokens and len(tokens) <= 8 and all(
        token in _SELECTION_WORDS or _mentioned_type(token) == account_type for token in tokens
    ):
        return {"action": "SELECT_TYPE", "needs": "active", "args": {"account_type": account_type}}
    if tokens and len(tokens) <= 10 and all(token in _EMPLOYMENT_WORDS for token in tokens):
        for category, pattern in _EMPLOYMENT_EN:
            if pattern.search(normalized):
                return {"action": "SET_EMPLOYMENT", "needs": "active", "args": {"employment_category": category}}
    return None


# --------------------------------------------------------------------------
# Localised reply text (English, Hindi, Kannada)
# --------------------------------------------------------------------------

_TYPE_LABEL = {
    "SAVINGS": {"en-IN": "savings", "hi-IN": "बचत", "kn-IN": "ಉಳಿತಾಯ"},
    "SALARY": {"en-IN": "salary", "hi-IN": "वेतन", "kn-IN": "ವೇತನ"},
    "CURRENT": {"en-IN": "current", "hi-IN": "चालू", "kn-IN": "ಚಾಲ್ತಿ"},
}
_FIELD_LABEL = {
    "account_type": {"en-IN": "account type", "hi-IN": "खाते का प्रकार", "kn-IN": "ಖಾತೆಯ ವಿಧ"},
    "date_of_birth": {"en-IN": "date of birth", "hi-IN": "जन्म तिथि", "kn-IN": "ಜನ್ಮ ದಿನಾಂಕ"},
    "phone": {"en-IN": "phone number", "hi-IN": "फ़ोन नंबर", "kn-IN": "ಫೋನ್ ಸಂಖ್ಯೆ"},
    "address_line": {"en-IN": "address", "hi-IN": "पता", "kn-IN": "ವಿಳಾಸ"},
    "city": {"en-IN": "city", "hi-IN": "शहर", "kn-IN": "ನಗರ"},
    "state": {"en-IN": "state", "hi-IN": "राज्य", "kn-IN": "ರಾಜ್ಯ"},
    "postal_code": {"en-IN": "PIN code", "hi-IN": "पिन कोड", "kn-IN": "ಪಿನ್ ಕೋಡ್"},
    "employment_category": {"en-IN": "employment category", "hi-IN": "रोज़गार श्रेणी", "kn-IN": "ಉದ್ಯೋಗ ವರ್ಗ"},
}
_DOCUMENT_LABEL = {
    "AADHAAR_DEMO": {
        "en-IN": "Aadhaar demonstration document",
        "hi-IN": "आधार डेमो दस्तावेज़",
        "kn-IN": "ಆಧಾರ್ ಡೆಮೊ ದಾಖಲೆ",
    },
    "PAN_DEMO": {
        "en-IN": "PAN demonstration document",
        "hi-IN": "पैन डेमो दस्तावेज़",
        "kn-IN": "ಪ್ಯಾನ್ ಡೆಮೊ ದಾಖಲೆ",
    },
    "ADDRESS_PROOF_DEMO": {
        "en-IN": "address proof demonstration document",
        "hi-IN": "पते के प्रमाण का डेमो दस्तावेज़",
        "kn-IN": "ವಿಳಾಸ ಪುರಾವೆ ಡೆಮೊ ದಾಖಲೆ",
    },
}
_EMPLOYMENT_LABEL = {
    "SALARIED": {"en-IN": "salaried", "hi-IN": "वेतनभोगी", "kn-IN": "ವೇತನದಾರ"},
    "GOVERNMENT_EMPLOYEE": {"en-IN": "government employee", "hi-IN": "सरकारी कर्मचारी", "kn-IN": "ಸರ್ಕಾರಿ ನೌಕರ"},
    "SELF_EMPLOYED": {"en-IN": "self-employed", "hi-IN": "स्वरोजगार", "kn-IN": "ಸ್ವಯಂ ಉದ್ಯೋಗಿ"},
    "BUSINESS_OWNER": {"en-IN": "business owner", "hi-IN": "व्यवसाय मालिक", "kn-IN": "ವ್ಯಾಪಾರ ಮಾಲೀಕ"},
    "STUDENT": {"en-IN": "student", "hi-IN": "छात्र", "kn-IN": "ವಿದ್ಯಾರ್ಥಿ"},
    "RETIRED": {"en-IN": "retired", "hi-IN": "सेवानिवृत्त", "kn-IN": "ನಿವೃತ್ತ"},
    "OTHER": {"en-IN": "other", "hi-IN": "अन्य", "kn-IN": "ಇತರೆ"},
}
_STATUS_LABEL = {
    "DRAFT": {"en-IN": "a saved draft", "hi-IN": "सहेजा हुआ ड्राफ़्ट", "kn-IN": "ಉಳಿಸಿದ ಕರಡು"},
    "PENDING_CONFIRMATION": {
        "en-IN": "waiting for your confirmation",
        "hi-IN": "आपकी पुष्टि की प्रतीक्षा में",
        "kn-IN": "ನಿಮ್ಮ ದೃಢೀಕರಣಕ್ಕಾಗಿ ಕಾಯುತ್ತಿದೆ",
    },
    "SUBMITTED": {"en-IN": "submitted", "hi-IN": "जमा किया गया", "kn-IN": "ಸಲ್ಲಿಸಲಾಗಿದೆ"},
}
_STATUS_NOTE = {
    "DRAFT": {
        "en-IN": "You can resume and finish it any time.",
        "hi-IN": "आप इसे कभी भी फिर से शुरू करके पूरा कर सकते हैं।",
        "kn-IN": "ನೀವು ಇದನ್ನು ಯಾವಾಗ ಬೇಕಾದರೂ ಮುಂದುವರಿಸಿ ಪೂರ್ಣಗೊಳಿಸಬಹುದು.",
    },
    "PENDING_CONFIRMATION": {
        "en-IN": "Nothing is submitted until you press the Confirm button yourself.",
        "hi-IN": "जब तक आप स्वयं पुष्टि बटन नहीं दबाते, कुछ भी जमा नहीं होता।",
        "kn-IN": "ನೀವೇ ದೃಢೀಕರಣ ಬಟನ್ ಒತ್ತುವವರೆಗೆ ಏನೂ ಸಲ್ಲಿಕೆಯಾಗುವುದಿಲ್ಲ.",
    },
    "SUBMITTED": {
        "en-IN": "It was submitted to this synthetic demo workflow. No real bank review or approval takes place.",
        "hi-IN": "इसे इस डेमो वर्कफ़्लो में जमा किया गया है। कोई वास्तविक बैंक समीक्षा या स्वीकृति नहीं होती।",
        "kn-IN": "ಇದನ್ನು ಈ ಡೆಮೊ ವರ್ಕ್‌ಫ್ಲೋಗೆ ಸಲ್ಲಿಸಲಾಗಿದೆ. ನಿಜವಾದ ಬ್ಯಾಂಕ್ ಪರಿಶೀಲನೆ ಅಥವಾ ಅನುಮೋದನೆ ನಡೆಯುವುದಿಲ್ಲ.",
    },
}

_TEXT = {
    "options": {
        "en-IN": (
            "I can help you open an account in this synthetic demo. The available types are a savings account, "
            "a salary account and a current account. A savings or salary account needs the Aadhaar and PAN "
            "demonstration documents, and a current account also needs an address proof demonstration document. "
            "Applicants must be at least 18 under the demo rules. These are demonstration rules only, not real bank terms. "
            "Which account would you like?"
        ),
        "hi-IN": (
            "मैं इस डेमो में खाता खुलवाने में आपकी मदद कर सकता हूँ। उपलब्ध प्रकार बचत खाता, वेतन खाता और चालू खाता हैं। "
            "बचत या वेतन खाते के लिए आधार और पैन डेमो दस्तावेज़ चाहिए, और चालू खाते के लिए पते के प्रमाण का डेमो दस्तावेज़ भी चाहिए। "
            "डेमो नियमों में आवेदक की आयु कम से कम 18 वर्ष होनी चाहिए। ये केवल डेमो नियम हैं, असली बैंक की शर्तें नहीं। "
            "आप कौन सा खाता चाहेंगे?"
        ),
        "kn-IN": (
            "ಈ ಡೆಮೊದಲ್ಲಿ ಖಾತೆ ತೆರೆಯಲು ನಾನು ನಿಮಗೆ ಸಹಾಯ ಮಾಡುತ್ತೇನೆ. ಲಭ್ಯವಿರುವ ವಿಧಗಳು ಉಳಿತಾಯ ಖಾತೆ, ವೇತನ ಖಾತೆ ಮತ್ತು ಚಾಲ್ತಿ ಖಾತೆ. "
            "ಉಳಿತಾಯ ಅಥವಾ ವೇತನ ಖಾತೆಗೆ ಆಧಾರ್ ಮತ್ತು ಪ್ಯಾನ್ ಡೆಮೊ ದಾಖಲೆಗಳು ಬೇಕು, ಚಾಲ್ತಿ ಖಾತೆಗೆ ವಿಳಾಸ ಪುರಾವೆ ಡೆಮೊ ದಾಖಲೆಯೂ ಬೇಕು. "
            "ಡೆಮೊ ನಿಯಮಗಳ ಪ್ರಕಾರ ಅರ್ಜಿದಾರರಿಗೆ ಕನಿಷ್ಠ 18 ವರ್ಷ ಆಗಿರಬೇಕು. ಇವು ಕೇವಲ ಡೆಮೊ ನಿಯಮಗಳು, ನಿಜವಾದ ಬ್ಯಾಂಕ್ ನಿಯಮಗಳಲ್ಲ. "
            "ನಿಮಗೆ ಯಾವ ಖಾತೆ ಬೇಕು?"
        ),
    },
    "started": {
        "en-IN": "I have started your {type} account application. This is a synthetic demo and your progress is saved as a draft.",
        "hi-IN": "मैंने आपका {type} खाता आवेदन शुरू कर दिया है। यह एक डेमो है और आपकी प्रगति ड्राफ़्ट के रूप में सहेजी गई है।",
        "kn-IN": "ನಾನು ನಿಮ್ಮ {type} ಖಾತೆ ಅರ್ಜಿಯನ್ನು ಪ್ರಾರಂಭಿಸಿದ್ದೇನೆ. ಇದು ಡೆಮೊ ಆಗಿದ್ದು, ನಿಮ್ಮ ಪ್ರಗತಿಯನ್ನು ಕರಡಾಗಿ ಉಳಿಸಲಾಗಿದೆ.",
    },
    "started_untyped": {
        "en-IN": "I have started a draft account application. This is a synthetic demo and your progress is saved.",
        "hi-IN": "मैंने खाता आवेदन का ड्राफ़्ट शुरू कर दिया है। यह एक डेमो है और आपकी प्रगति सहेजी गई है।",
        "kn-IN": "ನಾನು ಖಾತೆ ಅರ್ಜಿಯ ಕರಡನ್ನು ಪ್ರಾರಂಭಿಸಿದ್ದೇನೆ. ಇದು ಡೆಮೊ ಆಗಿದ್ದು, ನಿಮ್ಮ ಪ್ರಗತಿಯನ್ನು ಉಳಿಸಲಾಗಿದೆ.",
    },
    "resumed": {
        "en-IN": "I found your unfinished account application and resumed it. Your progress is saved.",
        "hi-IN": "मुझे आपका अधूरा खाता आवेदन मिला और मैंने उसे फिर शुरू कर दिया। आपकी प्रगति सहेजी गई है।",
        "kn-IN": "ನಿಮ್ಮ ಅಪೂರ್ಣ ಖಾತೆ ಅರ್ಜಿ ಸಿಕ್ಕಿತು, ಅದನ್ನು ಮುಂದುವರಿಸಿದ್ದೇನೆ. ನಿಮ್ಮ ಪ್ರಗತಿಯನ್ನು ಉಳಿಸಲಾಗಿದೆ.",
    },
    "choose_type": {
        "en-IN": "Please choose an account type first: savings, salary or current.",
        "hi-IN": "कृपया पहले खाते का प्रकार चुनें: बचत, वेतन या चालू।",
        "kn-IN": "ದಯವಿಟ್ಟು ಮೊದಲು ಖಾತೆಯ ವಿಧವನ್ನು ಆರಿಸಿ: ಉಳಿತಾಯ, ವೇತನ ಅಥವಾ ಚಾಲ್ತಿ.",
    },
    "next_details": {
        "en-IN": (
            "Next step: open Account Services, then Open New Account, and fill in these details: {fields}. "
            "Your name is taken from your demo profile."
        ),
        "hi-IN": (
            "अगला चरण: अकाउंट सर्विसेज में नया खाता खोलें सेक्शन खोलें और ये जानकारी भरें: {fields}। "
            "आपका नाम आपकी डेमो प्रोफ़ाइल से लिया गया है।"
        ),
        "kn-IN": (
            "ಮುಂದಿನ ಹಂತ: ಅಕೌಂಟ್ ಸರ್ವಿಸಸ್‌ನಲ್ಲಿ ಹೊಸ ಖಾತೆ ತೆರೆಯಿರಿ ವಿಭಾಗವನ್ನು ತೆರೆದು ಈ ವಿವರಗಳನ್ನು ಭರ್ತಿ ಮಾಡಿ: {fields}. "
            "ನಿಮ್ಮ ಹೆಸರನ್ನು ನಿಮ್ಮ ಡೆಮೊ ಪ್ರೊಫೈಲ್‌ನಿಂದ ತೆಗೆದುಕೊಳ್ಳಲಾಗಿದೆ."
        ),
    },
    "next_details_fix": {
        "en-IN": "Some of your details need correcting. Open Account Services, then Open New Account, and review the highlighted fields.",
        "hi-IN": "आपकी कुछ जानकारी सुधारनी होगी। अकाउंट सर्विसेज में नया खाता खोलें सेक्शन खोलकर चिह्नित फ़ील्ड देखें।",
        "kn-IN": "ನಿಮ್ಮ ಕೆಲವು ವಿವರಗಳನ್ನು ಸರಿಪಡಿಸಬೇಕು. ಅಕೌಂಟ್ ಸರ್ವಿಸಸ್‌ನ ಹೊಸ ಖಾತೆ ತೆರೆಯಿರಿ ವಿಭಾಗದಲ್ಲಿ ಗುರುತಿಸಿದ ಕ್ಷೇತ್ರಗಳನ್ನು ಪರಿಶೀಲಿಸಿ.",
    },
    "next_documents": {
        "en-IN": (
            "Your details are complete. Next, open the KYC Documents section and upload your demo documents: {docs}. "
            "I will check that each file meets the demo type and size rules. These are synthetic demo documents, "
            "and nothing is checked with any government system."
        ),
        "hi-IN": (
            "आपकी जानकारी पूरी है। अब केवाईसी डॉक्यूमेंट्स सेक्शन खोलें और अपने डेमो दस्तावेज़ अपलोड करें: {docs}। "
            "मैं जाँचूँगा कि हर फ़ाइल डेमो के प्रकार और आकार के नियमों को पूरा करती है। ये सिंथेटिक डेमो दस्तावेज़ हैं, "
            "और किसी सरकारी प्रणाली से कुछ भी सत्यापित नहीं किया जाता।"
        ),
        "kn-IN": (
            "ನಿಮ್ಮ ವಿವರಗಳು ಪೂರ್ಣವಾಗಿವೆ. ಈಗ ಕೆವೈಸಿ ಡಾಕ್ಯುಮೆಂಟ್ಸ್ ವಿಭಾಗವನ್ನು ತೆರೆದು ನಿಮ್ಮ ಡೆಮೊ ದಾಖಲೆಗಳನ್ನು ಅಪ್‌ಲೋಡ್ ಮಾಡಿ: {docs}. "
            "ಪ್ರತಿ ಫೈಲ್ ಡೆಮೊ ಪ್ರಕಾರ ಮತ್ತು ಗಾತ್ರದ ನಿಯಮಗಳನ್ನು ಪೂರೈಸುತ್ತದೆಯೇ ಎಂದು ನಾನು ಪರಿಶೀಲಿಸುತ್ತೇನೆ. ಇವು ಕೃತಕ ಡೆಮೊ ದಾಖಲೆಗಳು, "
            "ಯಾವುದೇ ಸರ್ಕಾರಿ ವ್ಯವಸ್ಥೆಯಿಂದ ಏನನ್ನೂ ಪರಿಶೀಲಿಸಲಾಗುವುದಿಲ್ಲ."
        ),
    },
    "next_review": {
        "en-IN": (
            "Everything required is complete. Open Review and Submit to see your full summary. Nothing is submitted until "
            "you press the Confirm button yourself, and I cannot submit it for you."
        ),
        "hi-IN": (
            "ज़रूरी सब कुछ पूरा है। पूरा सारांश देखने के लिए समीक्षा और जमा करें सेक्शन खोलें। जब तक आप स्वयं पुष्टि बटन नहीं दबाते, "
            "कुछ भी जमा नहीं होता, और मैं आपकी ओर से इसे जमा नहीं कर सकता।"
        ),
        "kn-IN": (
            "ಅಗತ್ಯವಿರುವ ಎಲ್ಲವೂ ಪೂರ್ಣವಾಗಿದೆ. ಪೂರ್ಣ ಸಾರಾಂಶ ನೋಡಲು ಪರಿಶೀಲನೆ ಮತ್ತು ಸಲ್ಲಿಕೆ ವಿಭಾಗವನ್ನು ತೆರೆಯಿರಿ. ನೀವೇ ದೃಢೀಕರಣ ಬಟನ್ ಒತ್ತುವವರೆಗೆ "
            "ಏನೂ ಸಲ್ಲಿಕೆಯಾಗುವುದಿಲ್ಲ, ಮತ್ತು ನಾನು ನಿಮ್ಮ ಪರವಾಗಿ ಸಲ್ಲಿಸಲು ಸಾಧ್ಯವಿಲ್ಲ."
        ),
    },
    "next_submitted": {
        "en-IN": "This application is already submitted. Open Track Applications to see it.",
        "hi-IN": "यह आवेदन पहले ही जमा हो चुका है। देखने के लिए आवेदन ट्रैक करें सेक्शन खोलें।",
        "kn-IN": "ಈ ಅರ್ಜಿಯನ್ನು ಈಗಾಗಲೇ ಸಲ್ಲಿಸಲಾಗಿದೆ. ನೋಡಲು ಅರ್ಜಿ ಟ್ರ್ಯಾಕ್ ವಿಭಾಗವನ್ನು ತೆರೆಯಿರಿ.",
    },
    "type_set": {
        "en-IN": "Done. I have set your account type to {type}.",
        "hi-IN": "हो गया। मैंने आपके खाते का प्रकार {type} कर दिया है।",
        "kn-IN": "ಆಯಿತು. ನಿಮ್ಮ ಖಾತೆಯ ವಿಧವನ್ನು {type} ಎಂದು ಹೊಂದಿಸಿದ್ದೇನೆ.",
    },
    "employment_set": {
        "en-IN": "I have saved your employment category as {category}.",
        "hi-IN": "मैंने आपकी रोज़गार श्रेणी {category} के रूप में सहेज ली है।",
        "kn-IN": "ನಿಮ್ಮ ಉದ್ಯೋಗ ವರ್ಗವನ್ನು {category} ಎಂದು ಉಳಿಸಿದ್ದೇನೆ.",
    },
    "employment_ineligible": {
        "en-IN": "That choice does not fit this account type under the synthetic demo rules. Please choose a different category or account type. Nothing was changed.",
        "hi-IN": "डेमो नियमों के अनुसार यह विकल्प इस खाते के प्रकार के लिए उपयुक्त नहीं है। कृपया दूसरी श्रेणी या खाते का प्रकार चुनें। कुछ भी नहीं बदला गया।",
        "kn-IN": "ಡೆಮೊ ನಿಯಮಗಳ ಪ್ರಕಾರ ಈ ಆಯ್ಕೆ ಈ ಖಾತೆಯ ವಿಧಕ್ಕೆ ಹೊಂದುವುದಿಲ್ಲ. ದಯವಿಟ್ಟು ಬೇರೆ ವರ್ಗ ಅಥವಾ ಖಾತೆಯ ವಿಧವನ್ನು ಆರಿಸಿ. ಏನೂ ಬದಲಾಗಿಲ್ಲ.",
    },
    "missing_fields": {
        "en-IN": "Missing details: {fields}.",
        "hi-IN": "बाकी जानकारी: {fields}।",
        "kn-IN": "ಬಾಕಿ ವಿವರಗಳು: {fields}.",
    },
    "missing_documents": {
        "en-IN": "Missing documents: {docs}.",
        "hi-IN": "बाकी दस्तावेज़: {docs}।",
        "kn-IN": "ಬಾಕಿ ದಾಖಲೆಗಳು: {docs}.",
    },
    "nothing_missing": {
        "en-IN": "Nothing is missing.",
        "hi-IN": "कुछ भी बाकी नहीं है।",
        "kn-IN": "ಏನೂ ಬಾಕಿಯಿಲ್ಲ.",
    },
    "not_ready": {
        "en-IN": "Your application is not ready to review yet.",
        "hi-IN": "आपका आवेदन अभी समीक्षा के लिए तैयार नहीं है।",
        "kn-IN": "ನಿಮ್ಮ ಅರ್ಜಿ ಇನ್ನೂ ಪರಿಶೀಲನೆಗೆ ಸಿದ್ಧವಾಗಿಲ್ಲ.",
    },
    "upload_help": {
        "en-IN": (
            "Open the KYC Documents section. You will see an upload area for each required document, including the "
            "Aadhaar demonstration document. Select your demo file there and upload it, and I will check that it meets "
            "the demo file type and size rules. {status}"
        ),
        "hi-IN": (
            "केवाईसी डॉक्यूमेंट्स सेक्शन खोलें। वहाँ हर ज़रूरी दस्तावेज़ के लिए अपलोड का क्षेत्र दिखेगा, जिसमें आधार डेमो दस्तावेज़ भी है। "
            "वहाँ अपनी डेमो फ़ाइल चुनकर अपलोड करें, और मैं जाँचूँगा कि वह डेमो फ़ाइल प्रकार और आकार के नियमों को पूरा करती है। {status}"
        ),
        "kn-IN": (
            "ಕೆವೈಸಿ ಡಾಕ್ಯುಮೆಂಟ್ಸ್ ವಿಭಾಗವನ್ನು ತೆರೆಯಿರಿ. ಅಲ್ಲಿ ಪ್ರತಿ ಅಗತ್ಯ ದಾಖಲೆಗೆ ಅಪ್‌ಲೋಡ್ ಪ್ರದೇಶ ಕಾಣುತ್ತದೆ, ಆಧಾರ್ ಡೆಮೊ ದಾಖಲೆಯೂ ಅದರಲ್ಲಿದೆ. "
            "ಅಲ್ಲಿ ನಿಮ್ಮ ಡೆಮೊ ಫೈಲ್ ಆಯ್ಕೆ ಮಾಡಿ ಅಪ್‌ಲೋಡ್ ಮಾಡಿ, ಅದು ಡೆಮೊ ಫೈಲ್ ಪ್ರಕಾರ ಮತ್ತು ಗಾತ್ರದ ನಿಯಮಗಳನ್ನು ಪೂರೈಸುತ್ತದೆಯೇ ಎಂದು ನಾನು ಪರಿಶೀಲಿಸುತ್ತೇನೆ. {status}"
        ),
    },
    "upload_still": {
        "en-IN": "Still needed: {docs}.",
        "hi-IN": "अभी भी चाहिए: {docs}।",
        "kn-IN": "ಇನ್ನೂ ಬೇಕಾಗಿದೆ: {docs}.",
    },
    "upload_done": {
        "en-IN": "All required documents are uploaded.",
        "hi-IN": "सभी ज़रूरी दस्तावेज़ अपलोड हो चुके हैं।",
        "kn-IN": "ಎಲ್ಲಾ ಅಗತ್ಯ ದಾಖಲೆಗಳು ಅಪ್‌ಲೋಡ್ ಆಗಿವೆ.",
    },
    "upload_needs_type": {
        "en-IN": "Please choose an account type first, because the required documents depend on it.",
        "hi-IN": "कृपया पहले खाते का प्रकार चुनें, क्योंकि ज़रूरी दस्तावेज़ उसी पर निर्भर करते हैं।",
        "kn-IN": "ದಯವಿಟ್ಟು ಮೊದಲು ಖಾತೆಯ ವಿಧವನ್ನು ಆರಿಸಿ, ಏಕೆಂದರೆ ಅಗತ್ಯ ದಾಖಲೆಗಳು ಅದರ ಮೇಲೆ ಅವಲಂಬಿತವಾಗಿವೆ.",
    },
    "saved_later": {
        "en-IN": (
            "Yes. Your application progress is saved as a draft. You can come back later and resume it from Account Services, "
            "Open New Account, or simply ask me to continue."
        ),
        "hi-IN": (
            "हाँ। आपके आवेदन की प्रगति ड्राफ़्ट के रूप में सहेजी गई है। आप बाद में अकाउंट सर्विसेज के नया खाता खोलें सेक्शन से "
            "इसे फिर शुरू कर सकते हैं, या मुझसे जारी रखने को कह सकते हैं।"
        ),
        "kn-IN": (
            "ಹೌದು. ನಿಮ್ಮ ಅರ್ಜಿಯ ಪ್ರಗತಿಯನ್ನು ಕರಡಾಗಿ ಉಳಿಸಲಾಗಿದೆ. ನೀವು ನಂತರ ಅಕೌಂಟ್ ಸರ್ವಿಸಸ್‌ನ ಹೊಸ ಖಾತೆ ತೆರೆಯಿರಿ ವಿಭಾಗದಿಂದ ಮುಂದುವರಿಸಬಹುದು, "
            "ಅಥವಾ ಮುಂದುವರಿಸಲು ನನಗೆ ಹೇಳಬಹುದು."
        ),
    },
    "status_found": {
        "en-IN": "Your {type}account application {ref} is {status}. {note}",
        "hi-IN": "आपका {type}खाता आवेदन {ref} {status} है। {note}",
        "kn-IN": "ನಿಮ್ಮ {type}ಖಾತೆ ಅರ್ಜಿ {ref} {status}. {note}",
    },
    "status_none": {
        "en-IN": "I could not find an account application for you yet. You can tell me you want to open a new account to start one.",
        "hi-IN": "मुझे आपका कोई खाता आवेदन अभी नहीं मिला। शुरू करने के लिए आप कह सकते हैं कि आप नया खाता खोलना चाहते हैं।",
        "kn-IN": "ನಿಮ್ಮ ಯಾವುದೇ ಖಾತೆ ಅರ್ಜಿ ಇನ್ನೂ ಸಿಕ್ಕಿಲ್ಲ. ಪ್ರಾರಂಭಿಸಲು ಹೊಸ ಖಾತೆ ತೆರೆಯಲು ಬಯಸುವುದಾಗಿ ಹೇಳಿ.",
    },
    "error": {
        "en-IN": "I could not complete that step, and your application was not changed. Please try again.",
        "hi-IN": "मैं यह चरण पूरा नहीं कर सका, और आपका आवेदन नहीं बदला गया। कृपया फिर कोशिश करें।",
        "kn-IN": "ನಾನು ಈ ಹಂತವನ್ನು ಪೂರ್ಣಗೊಳಿಸಲು ಸಾಧ್ಯವಾಗಲಿಲ್ಲ, ನಿಮ್ಮ ಅರ್ಜಿ ಬದಲಾಗಿಲ್ಲ. ದಯವಿಟ್ಟು ಮತ್ತೆ ಪ್ರಯತ್ನಿಸಿ.",
    },
}

_STEP_NAVIGATION = {
    "account_type": "Open New Account",
    "details": "Open New Account",
    "documents": "KYC Documents",
    "review": "Review & Submit",
    "submitted": "Track Applications",
}
_TOOL_SUMMARY = {
    "start_account_application": "Started or resumed the synthetic account application draft.",
    "get_account_application": "Retrieved the account application progress.",
    "update_account_application": "Saved the selected choice on the draft application.",
    "validate_account_application": "Validated the draft application.",
    "get_required_application_documents": "Listed the synthetic demo account options.",
    "get_application_document_status": "Checked which demo documents are uploaded.",
    "prepare_account_application_submission": "Prepared the application review; nothing was submitted.",
    "get_account_application_status": "Retrieved the account application status.",
}


def _language(language):
    return language if language in SUPPORTED_LANGUAGES else "en-IN"


def _t(key, language, **values):
    return _TEXT[key][_language(language)].format(**values)


def _labels(table, keys, language):
    return ", ".join(table[key][_language(language)] for key in keys if key in table)


def _next_step_text(view, language):
    step = view["progress"]["current_step"]
    validation = view["validation"]
    if step == "account_type":
        return _t("choose_type", language)
    if step == "details":
        fields = [name for name in validation["missing_fields"] if name != "account_type"]
        if fields:
            if language == "en-IN":
                field = fields[0]
                hint = " (YYYY-MM-DD)" if field == "date_of_birth" else ""
                return f"Please tell me your {applications.FIELD_LABELS[field].lower()}{hint}. You can type or speak it."
            return _t("next_details", language, fields=_labels(_FIELD_LABEL, fields, language))
        return _t("next_details_fix", language)
    if step == "documents":
        if language == "en-IN":
            return ("Please upload " + _labels(_DOCUMENT_LABEL, validation["missing_documents"], language)
                    + " using + > KYC document in this conversation. Uploads are not identity verification.")
        return _t("next_documents", language, docs=_labels(_DOCUMENT_LABEL, validation["missing_documents"], language))
    if step == "review":
        if not _ready_for_review(view):
            return "I cannot confirm that the application is ready. Please check the required documents."
        return _t("next_review", language)
    return _t("next_submitted", language)


def _missing_text(view, language):
    validation = view["validation"]
    parts = []
    fields = [name for name in validation["missing_fields"] if name != "account_type"]
    if view.get("account_type") is None:
        parts.append(_t("choose_type", language))
    if fields:
        parts.append(_t("missing_fields", language, fields=_labels(_FIELD_LABEL, fields, language)))
    if validation["missing_documents"]:
        parts.append(_t("missing_documents", language, docs=_labels(_DOCUMENT_LABEL, validation["missing_documents"], language)))
    return " ".join(parts)


def _navigation(view):
    step = (view.get("progress") or {}).get("current_step")
    subfeature = _STEP_NAVIGATION.get(step)
    if subfeature is None:
        return None
    nav = {"target": "account_services", "subfeature": subfeature, "step": step}
    if step == "review":
        if not _ready_for_review(view):
            return None
        nav["application_ref"] = view["application_ref"]
    return nav


def _ready_for_review(view):
    validation = view.get("validation") or {}
    if not validation.get("is_valid") or validation.get("missing_documents"):
        return False
    required = applications.required_documents(view.get("account_type"))
    documents = view.get("documents")
    if not required or not isinstance(documents, list):
        return False
    by_type = {doc.get("document_type"): doc for doc in documents if isinstance(doc, dict)}
    return all(
        kind in by_type and by_type[kind].get("uploaded")
        and str(by_type[kind].get("verification_status") or "").upper() != "REJECTED"
        for kind in required
    )


# --------------------------------------------------------------------------
# Turn execution
# --------------------------------------------------------------------------

_FIELD_INPUTS = {
    "date_of_birth": r"(?:date of birth|birth date|dob)",
    "phone": r"(?:phone(?: number)?|mobile(?: number)?)",
    "address_line": r"(?:address|street address)",
    "city": r"city", "state": r"state", "postal_code": r"(?:pin(?: code)?|postal code|postcode)",
    "nominee_name": r"nominee name", "nominee_relationship": r"nominee relationship",
}


def redact_application_history(turns):
    """Keep personal slot values in owner-only history, out of later model context."""
    result = []
    previous_reply = ""
    for turn in turns:
        item = dict(turn)
        if item.get("role") == "assistant":
            previous_reply = item.get("assistant_response_text") or ""
        else:
            text = item.get("user_text_or_transcript") or ""
            labeled = any(re.search(r"(?:^|[;\n])\s*(?:my\s+)?" + pattern + r"\s*(?::|is\b)", text, re.I) for pattern in _FIELD_INPUTS.values())
            slot_reply = "Please tell me your " in previous_reply and not re.search(
                r"\b(?:show|check|balance|transactions?|loans?|cards?|scam|support|ticket|submit|cancel|stop|open)\b|\?", text, re.I,
            )
            if labeled or slot_reply:
                item["user_text_or_transcript"] = "[Customer-provided application fields omitted from model context]"
        result.append(item)
    return result


def _save_conversational_fields(principal, text, language, history):
    """Parse user-authored fields only; never infer identity from attachments/LLM output."""
    changes = {}
    implicit = False
    for piece in re.split(r"[;\n]", text.strip()):
        for field, pattern in _FIELD_INPUTS.items():
            match = re.fullmatch(r"(?:my\s+)?" + pattern + r"\s*(?::|is\b)\s*(.+)", piece.strip(), re.I)
            if match:
                changes[field] = match.group(1).strip()
                break
        else:
            changes = {}
            break
    if not changes and history:
        previous = next((turn for turn in reversed(history) if turn.get("role") == "assistant"), {})
        prompt = previous.get("assistant_response_text") or ""
        for field in _FIELD_INPUTS:
            label = applications.FIELD_LABELS[field].lower()
            if f"Please tell me your {label}" in prompt and not classify_account_intent(text):
                if not re.search(r"\b(?:yes|no|show|check|balance|transactions?|loans?|cards?|scam|support|ticket|submit|cancel|stop|open)\b|\?", text, re.I):
                    changes[field] = text.strip()
                    implicit = True
                break
    if not changes:
        return None
    try:
        view = applications.get_current_application(principal)
    except ApplicationNotFound:
        return None
    if implicit and not set(changes).issubset(view["validation"]["missing_fields"]):
        return "That field is already saved. " + _next_step_text(view, language), []
    if "date_of_birth" in changes:
        from datetime import datetime
        value = re.sub(r"(\d)(st|nd|rd|th)\b", r"\1", changes["date_of_birth"], flags=re.I)
        for fmt in ("%d %B %Y", "%B %d %Y", "%d %b %Y"):
            try:
                changes["date_of_birth"] = datetime.strptime(value.replace(",", ""), fmt).date().isoformat()
                break
            except ValueError:
                pass
    if "nominee_relationship" in changes:
        changes["nominee_relationship"] = changes["nominee_relationship"].upper().replace(" ", "_")
    try:
        updated = applications.update_application(principal, view["application_ref"], changes, include_personal=False)
    except ApplicationValidationError as exc:
        return " ".join(item["message"] for item in exc.errors), []
    result = ToolExecutionResult(
        tool_call_id=uuid4(), tool_name="update_account_application", status="succeeded",
        summary="Saved customer-provided application details.", data={"result": updated},
    )
    nav = _navigation(updated)
    if nav:
        result.data["ui_navigation"] = nav
    return "Your application details are saved. " + _next_step_text(updated, language), [result]


def run_account_application_turn(principal, text, language, execute, *, support_create_active=False, history=(), confirmation=None, input_mode="text"):
    """Run one account-opening turn; return ``(text, tool_results)`` or None.

    ``execute`` is the approved dispatcher (``execute_tool``). Returning None
    means the turn is not an account-opening turn and normal routing applies.
    """
    if support_create_active:
        return None
    if re.fullmatch(r"i confirm[,:]? submit my application[.!]?", text.strip(), re.I):
        return "Please review and submit your application personally on the Review & Submit page. No submission was made here.", []
    previous = next((item.get("assistant_response_text") or "" for item in reversed(history) if item.get("role") == "assistant"), "")
    pending = re.search(r"Unfinished application (VSA-[A-F0-9]{16}) ", previous)
    if pending and re.fullmatch(r"(?:yes|yes please|yes,? resume(?: it)?|resume|resume it|continue|continue it|हाँ|हां|ಹೌದು|no(?:[ ,].*)?|decline|नहीं|ಇಲ್ಲ)[.!]?", text.strip(), re.I):
        if re.match(r"(?:no\b|decline\b|नहीं|ಇಲ್ಲ)", text.strip(), re.I):
            return "Your existing draft remains unchanged. Only one unfinished application is currently permitted. You can resume it later.", []
        try:
            view = applications.get_current_application(principal, include_personal=False)
            if view["application_ref"] != pending.group(1):
                return "The application has changed. Please ask to open an account again.", []
            return "Resuming your existing application. " + _next_step_text(view, language), []
        except (ApplicationError, ApplicationNotFound, PermissionError):
            return "The application is unavailable. Please ask to open an account again.", []
    if pending and (classify_account_intent(text) or any(
        re.search(r"(?:^|[;\n])\s*(?:my\s+)?" + pattern + r"\s*(?::|is\b)", text, re.I)
        for pattern in _FIELD_INPUTS.values()
    )):
        return previous + " Please say yes to resume it or no to leave it unchanged.", []
    field_turn = _save_conversational_fields(principal, text, language, history)
    if field_turn is not None:
        return field_turn
    intent = classify_account_intent(text)
    if intent is None and re.search(r"document|kyc|aadhaar|pan|proof", previous, re.I) and (
        re.search(r"\b(?:uploaded|attached)\s+(?:them|those|these|it)\b", text, re.I)
        or re.fullmatch(r"(?:okay[, ]*)?(?:they(?:'re| are) (?:done|ready|uploaded|complete)|(?:are|were) (?:they|those|these) (?:done|ready|received|verified|complete)|all done|done)[.!?]?", text.strip(), re.I)
    ):
        intent = {"action": "UPLOAD_HELP", "needs": None, "args": {}}
    if intent is None and re.fullmatch(r"(?:resume|continue)(?: my)?(?: application)?[.!]?", text.strip(), re.I):
        intent = {"action": "OPEN", "needs": "active", "args": {}}
    if intent is None:
        return None
    action, needs, args = intent["action"], intent["needs"], dict(intent["args"])
    if needs == "active":
        try:
            if not has_active_application(principal):
                return None
        except PermissionError:
            return None
    lang = _language(language)
    results = []
    failure = {"error": None}

    def call(name, arguments):
        result_id = uuid4()
        try:
            data = execute(name, principal, arguments)
        except ApplicationValidationError as exc:
            failure["error"] = exc
            results.append(ToolExecutionResult(
                tool_call_id=result_id, tool_name=name, status="failed",
                summary="The request did not meet the demo validation rules.",
                error_code="VALIDATION_FAILED",
            ))
            return None
        except (ApplicationError, ApplicationNotFound, ValueError, PermissionError) as exc:
            failure["error"] = exc
            results.append(ToolExecutionResult(
                tool_call_id=result_id, tool_name=name, status="failed",
                summary="The request could not be completed.", error_code="REQUEST_REJECTED",
            ))
            return None
        except Exception as exc:  # noqa: BLE001 - never expose provider/storage detail
            logger.warning("Account application tool failed (%s)", type(exc).__name__)
            failure["error"] = exc
            results.append(ToolExecutionResult(
                tool_call_id=result_id, tool_name=name, status="failed",
                summary="The account application service is unavailable.",
                error_code="SERVICE_UNAVAILABLE",
            ))
            return None
        results.append(ToolExecutionResult(
            tool_call_id=result_id, tool_name=name, status="succeeded",
            summary=_TOOL_SUMMARY.get(name, "Completed."), data={"result": data},
        ))
        return data

    def attach_navigation(view):
        navigation = _navigation(view) if isinstance(view, dict) and "progress" in view else None
        if navigation and results and results[-1].status == "succeeded":
            results[-1].data = {**(results[-1].data or {}), "ui_navigation": navigation}

    def error_text():
        error = failure["error"]
        if isinstance(error, ApplicationValidationError) and any(
            item.get("code") == "EMPLOYMENT_NOT_ELIGIBLE" for item in error.errors
        ):
            return _t("employment_ineligible", lang)
        return _t("error", lang)

    if action == "OPTIONS":
        data = call("get_required_application_documents", {})
        return (_t("options", lang) if data is not None else error_text()), results

    if action == "OPEN":
        try:
            existing = applications.get_current_application(principal, include_personal=False)
        except ApplicationNotFound:
            existing = None
        except (ApplicationError, PermissionError):
            return _t("error", lang), results
        if existing:
            label = existing.get("account_type_label") or "account type not selected"
            status = "awaiting final review" if existing.get("status") == "PENDING_CONFIRMATION" else "draft"
            return (f"Unfinished application {existing['application_ref']} ({label}; {status}) exists. "
                    "Would you like to resume it? Only one unfinished application is permitted; "
                    "a separate application cannot be started until this one is completed."), results
        view = call("start_account_application", args)
        if view is None:
            return error_text(), results
        attach_navigation(view)
        if view.get("resumed"):
            label = view.get("account_type_label") or "account type not selected"
            return (f"Unfinished application {view['application_ref']} ({label}) exists. Would you like to resume it? "
                    "Only one unfinished application is permitted."), results
        elif view.get("account_type"):
            lead = _t("started", lang, type=_TYPE_LABEL[view["account_type"]][lang])
        else:
            lead = _t("started_untyped", lang)
        follow = _t("options", lang) if view["progress"]["current_step"] == "account_type" else _next_step_text(view, lang)
        return f"{lead} {follow}", results

    if action in {"SELECT_TYPE", "SET_EMPLOYMENT"}:
        if action == "SELECT_TYPE":
            try:
                existing = applications.get_current_application(principal, include_personal=False)
            except (ApplicationError, ApplicationNotFound, PermissionError):
                return error_text(), results
            if existing.get("account_type") and existing["account_type"] != args["account_type"]:
                return (f"Your existing {existing.get('account_type_label') or 'account'} application remains unchanged. "
                        "Only one unfinished application is permitted. Review its details on Open New Account before making any changes."), results
        view = call("update_account_application", args)
        if view is None:
            return error_text(), results
        attach_navigation(view)
        if action == "SELECT_TYPE":
            lead = _t("type_set", lang, type=_TYPE_LABEL[view["account_type"]][lang])
        else:
            lead = _t("employment_set", lang, category=_EMPLOYMENT_LABEL[args["employment_category"]][lang])
        return f"{lead} {_next_step_text(view, lang)}", results

    if action == "REMAINING":
        view = call("validate_account_application", {})
        if view is None:
            return error_text(), results
        attach_navigation(view)
        missing = _missing_text(view, lang)
        lead = missing if missing else _t("nothing_missing", lang)
        return f"{lead} {_next_step_text(view, lang)}", results

    if action == "UPLOAD_HELP":
        data = call("get_application_document_status", {})
        if data is None:
            return "I could not check the document checklist. Please try again.", results
        if data.get("account_type") is None:
            return _t("upload_needs_type", lang), results
        documents = data.get("documents")
        if not isinstance(documents, list) or not isinstance(data.get("missing_documents"), list):
            return "I could not check the document checklist. Please try again.", results
        descriptions = []
        for doc in documents:
            if not isinstance(doc, dict) or not doc.get("required"):
                continue
            label = doc.get("label") or "Required document"
            verification = str(doc.get("verification_status") or "UNKNOWN").upper()
            if verification == "REJECTED":
                status = "rejected; upload a replacement"
            elif not doc.get("uploaded") or doc.get("document_type") in data["missing_documents"]:
                status = "missing"
            elif verification == "VERIFIED":
                status = "uploaded and verified"
            elif verification in {"PENDING", "PENDING_VERIFICATION"}:
                status = "uploaded; verification pending"
            else:
                status = "uploaded; not verified"
            descriptions.append(f"{label}: {status}")
        if not descriptions:
            return "I could not check the required document checklist. Please try again.", results
        results[-1].data = {**(results[-1].data or {}), "ui_navigation": {"target": "account_services", "subfeature": "KYC Documents", "step": "documents"}}
        review = call("validate_account_application", {})
        suffix = " Uploading does not verify identity or submit an application."
        if review is None:
            suffix += " I could not check whether the application is ready for review."
        elif _ready_for_review(review) and not data["missing_documents"] and not any(
            d.get("required") and str(d.get("verification_status") or "").upper() == "REJECTED"
            for d in documents if isinstance(d, dict)
        ):
            results[-1].data = {**(results[-1].data or {}), "ui_navigation": {
                "target": "account_services", "subfeature": "Review & Submit", "step": "review",
                "application_ref": review["application_ref"],
            }}
            suffix += " The application is ready for your review on Review & Submit."
        return "Document checklist: " + "; ".join(descriptions) + "." + suffix, results

    if action == "REVIEW":
        view = call("validate_account_application", {})
        if view is None:
            return "I could not check the application or document checklist. Please try again.", results
        if _ready_for_review(view):
            results[-1].data = {**(results[-1].data or {}), "ui_navigation": {
                "target": "account_services", "subfeature": "Review & Submit", "step": "review",
                "application_ref": view["application_ref"],
            }}
            return (f"Application {view['application_ref']} is ready for your review. "
                    "Uploads are not identity verification. Please review and submit personally on Review & Submit."), results
        return f"{_t('not_ready', lang)} {_missing_text(view, lang)} {_next_step_text(view, lang)}".replace("  ", " "), results

    if action == "SAVE_LATER":
        view = call("get_account_application", {})
        if view is None:
            return error_text(), results
        return _t("saved_later", lang), results

    # STATUS
    data = call("get_account_application_status", args)
    if data is None:
        return error_text(), results
    if not data.get("found"):
        if needs == "bare":
            return None
        return _t("status_none", lang), results
    account_type = data.get("account_type")
    type_prefix = f"{_TYPE_LABEL[account_type][lang]} " if account_type in _TYPE_LABEL else ""
    results[-1].data = {
        **(results[-1].data or {}),
        "ui_navigation": {"target": "account_services", "subfeature": "Track Applications", "step": "status"},
    }
    return _t(
        "status_found", lang, type=type_prefix, ref=data["application_ref"],
        status=_STATUS_LABEL[data["status"]][lang], note=_STATUS_NOTE[data["status"]][lang],
    ), results
