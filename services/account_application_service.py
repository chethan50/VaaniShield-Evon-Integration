"""Owner-scoped synthetic account-opening applications.

Everything here is a SYNTHETIC DEMO workflow. No real bank, government or KYC
system is contacted, and no document is verified. Validation, ownership,
status transitions, document handling and submission are all deterministic
Python; an LLM never decides that an application is complete or submitted.

State machine:  DRAFT -> PENDING_CONFIRMATION -> SUBMITTED
Any edit to a PENDING_CONFIRMATION application returns it to DRAFT, so a
confirmation always refers to exactly the content that was reviewed.
"""

import hashlib
import hmac
import json
import re
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime, timezone
from pathlib import Path
from uuid import uuid4

from services import database
from services.banking_service import get_customer_profile_by_key
from services.principal import CustomerPrincipal


SYNTHETIC_NOTICE = (
    "Synthetic VaaniShield demo workflow. Account types, rules and documents are "
    "demonstration data only; nothing is verified by a bank or government system."
)

ACTIVE_STATUSES = ("DRAFT", "PENDING_CONFIRMATION")
STATUS_NOTES = {
    "DRAFT": "Draft saved. You can resume it any time.",
    "PENDING_CONFIRMATION": "Review prepared. Nothing has been submitted until you press the Confirm button.",
    "SUBMITTED": (
        "Submitted to the synthetic demo workflow. No real bank review, approval "
        "or account creation takes place in this demo."
    ),
}

EMPLOYMENT_CATEGORIES = (
    "SALARIED", "GOVERNMENT_EMPLOYEE", "SELF_EMPLOYED", "BUSINESS_OWNER",
    "STUDENT", "RETIRED", "OTHER",
)
NOMINEE_RELATIONSHIPS = ("SPOUSE", "PARENT", "CHILD", "SIBLING", "OTHER")

DOCUMENT_TYPES = {
    "AADHAAR_DEMO": "Aadhaar demonstration document",
    "PAN_DEMO": "PAN demonstration document",
    "ADDRESS_PROOF_DEMO": "Address proof demonstration document",
}

# Clearly synthetic business rules. They are NOT real eligibility conditions,
# interest rates, minimum balances or fees, and none are stated here.
ACCOUNT_TYPES = {
    "SAVINGS": {
        "label": "Savings account",
        "description": "Synthetic demo savings account for everyday deposits.",
        "required_documents": ("AADHAAR_DEMO", "PAN_DEMO"),
        "required_employment": None,
        "employment_rule": "Employment category is optional.",
    },
    "SALARY": {
        "label": "Salary account",
        "description": "Synthetic demo salary account for salaried customers.",
        "required_documents": ("AADHAAR_DEMO", "PAN_DEMO"),
        "required_employment": ("SALARIED", "GOVERNMENT_EMPLOYEE"),
        "employment_rule": "Employment category must be Salaried or Government employee.",
    },
    "CURRENT": {
        "label": "Current account",
        "description": "Synthetic demo current account for self-employed customers and businesses.",
        "required_documents": ("AADHAAR_DEMO", "PAN_DEMO", "ADDRESS_PROOF_DEMO"),
        "required_employment": ("SELF_EMPLOYED", "BUSINESS_OWNER"),
        "employment_rule": "Employment category must be Self-employed or Business owner.",
    },
}

FIELD_LABELS = {
    "account_type": "Account type",
    "date_of_birth": "Date of birth",
    "phone": "Phone number",
    "address_line": "Address",
    "city": "City",
    "state": "State",
    "postal_code": "PIN code",
    "employment_category": "Employment category",
    "nominee_name": "Nominee name",
    "nominee_relationship": "Nominee relationship",
}
EDITABLE_FIELDS = tuple(FIELD_LABELS)
_PERSONAL_FIELDS = (
    "full_name", "date_of_birth", "phone", "email", "address_line", "city",
    "state", "postal_code", "employment_category", "nominee_name",
    "nominee_relationship",
)

MINIMUM_AGE_YEARS = 18
MAXIMUM_AGE_YEARS = 120
MAX_DOCUMENT_BYTES = 2 * 1024 * 1024
_DOCUMENT_FORMATS = {
    "application/pdf": (".pdf", lambda data: data.startswith(b"%PDF-")),
    "image/png": (".png", lambda data: data.startswith(b"\x89PNG\r\n\x1a\n")),
    "image/jpeg": (".jpg", lambda data: data.startswith(b"\xff\xd8\xff")),
}
# Resolved lazily so importing this module never touches the filesystem.
DOCUMENT_ROOT = database.DATA_DIR / "kyc_documents"

_REF_PATTERN = re.compile(r"^VSA-[A-F0-9]{16}$")
_STORAGE_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")


class ApplicationError(ValueError):
    """Safe application error; messages are suitable for API responses."""


class ApplicationNotFound(LookupError):
    """The application or document does not exist for this customer."""


class ApplicationConflict(ApplicationError):
    """The requested transition is not allowed in the current state."""


class ApplicationValidationError(ApplicationError):
    """One or more submitted values failed deterministic validation."""

    def __init__(self, errors):
        self.errors = list(errors)
        super().__init__("; ".join(item["message"] for item in self.errors) or "Invalid application data.")


class DocumentRejected(ApplicationError):
    """The uploaded file failed type, size or content validation."""


# ---------------------------------------------------------------------------
# Infrastructure
# ---------------------------------------------------------------------------

@contextmanager
def _transaction():
    conn = database.get_connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _now():
    return datetime.now(timezone.utc)


def _ts(value=None):
    return (value or _now()).astimezone(timezone.utc).isoformat(timespec="microseconds")


def _owner(principal):
    """Return (customer_key, customer_id) for a verified demo principal."""
    if (
        not isinstance(principal, CustomerPrincipal)
        or principal.auth_method != "demo_cookie"
        or principal.customer_key not in frozenset(database.DEMO_CUSTOMER_KEYS.values())
        or principal.expires_at <= _now()
    ):
        raise PermissionError("An authenticated demo customer is required.")
    profile = get_customer_profile_by_key(principal.customer_key)
    if profile is None:
        raise PermissionError("The authenticated demo customer is unavailable.")
    return principal.customer_key, profile


def _audit(conn, profile, action, application_ref):
    # Audit details carry only the public reference: never personal data.
    conn.execute(
        "INSERT INTO audit_logs (customer_id, action, details) VALUES (?, ?, ?)",
        (profile["customer_id"], action, f"application_ref={application_ref}"),
    )


def _event(conn, row, owner, event_type, summary, profile=None):
    conn.execute(
        "INSERT INTO account_application_events "
        "(application_id, customer_key, event_type, summary, occurred_at) VALUES (?, ?, ?, ?, ?)",
        (row["application_id"], owner, event_type, summary, _ts()),
    )
    if profile is not None:
        _audit(conn, profile, "ACCOUNT_APPLICATION_" + event_type, row["application_ref"])


def _find(conn, owner, application_ref=None, *, for_status=None):
    """Owner-scoped lookup. A foreign or unknown reference is simply not found."""
    if application_ref is None:
        row = conn.execute(
            "SELECT * FROM account_applications WHERE customer_key = ? AND status IN ('DRAFT', 'PENDING_CONFIRMATION') "
            "ORDER BY application_id DESC LIMIT 1",
            (owner,),
        ).fetchone()
        if row is None and for_status != "active":
            row = conn.execute(
                "SELECT * FROM account_applications WHERE customer_key = ? "
                "ORDER BY created_at DESC, application_id DESC LIMIT 1",
                (owner,),
            ).fetchone()
    else:
        if not isinstance(application_ref, str) or not _REF_PATTERN.fullmatch(application_ref.upper()):
            raise ApplicationNotFound("Application not found.")
        row = conn.execute(
            "SELECT * FROM account_applications WHERE application_ref = ? AND customer_key = ?",
            (application_ref.upper(), owner),
        ).fetchone()
    if row is None:
        raise ApplicationNotFound("Application not found.")
    return row


# ---------------------------------------------------------------------------
# Field normalization and validation (pure Python)
# ---------------------------------------------------------------------------

def _error(code, field, message):
    return {"code": code, "field": field, "message": message}


def _clean(value, *, minimum, maximum, field, pattern=None, label=None):
    if not isinstance(value, str):
        raise ApplicationValidationError([_error("INVALID_TYPE", field, f"{label or field} must be text.")])
    text = re.sub(r"\s+", " ", value).strip()
    if any(ord(character) < 32 or ord(character) == 127 for character in text):
        raise ApplicationValidationError([_error("INVALID_CHARACTERS", field, f"{label or field} contains invalid characters.")])
    if not minimum <= len(text) <= maximum:
        raise ApplicationValidationError([_error(
            "INVALID_LENGTH", field, f"{label or field} must be {minimum} to {maximum} characters.",
        )])
    if pattern is not None and not pattern(text):
        raise ApplicationValidationError([_error("INVALID_FORMAT", field, f"{label or field} has an invalid format.")])
    return text


def _is_name(text):
    return all(character.isalpha() or character in " .'-" or "\u0900" <= character <= "\u0dff" for character in text)


def _is_address(text):
    return all(
        character.isalnum() or character in " ,.-/#'" or "\u0900" <= character <= "\u0dff"
        for character in text
    )


def parse_date_of_birth(value, today=None):
    today = today or date.today()
    try:
        parsed = date.fromisoformat(value) if isinstance(value, str) and len(value) == 10 else None
    except ValueError:
        parsed = None
    if parsed is None:
        raise ApplicationValidationError([_error("INVALID_DATE", "date_of_birth", "Date of birth must be a valid date (YYYY-MM-DD).")])
    if parsed >= today:
        raise ApplicationValidationError([_error("INVALID_DATE", "date_of_birth", "Date of birth cannot be in the future.")])
    age = today.year - parsed.year - ((today.month, today.day) < (parsed.month, parsed.day))
    if age < MINIMUM_AGE_YEARS:
        raise ApplicationValidationError([_error(
            "UNDER_MINIMUM_AGE", "date_of_birth",
            f"Applicants must be at least {MINIMUM_AGE_YEARS} years old (synthetic demo rule).",
        )])
    if age > MAXIMUM_AGE_YEARS:
        raise ApplicationValidationError([_error("INVALID_DATE", "date_of_birth", "Date of birth is not plausible.")])
    return parsed.isoformat()


def normalize_phone(value):
    if not isinstance(value, str):
        return None
    digits = re.sub(r"[\s-]", "", value)
    digits = re.sub(r"^(?:\+91|0091|91|0)(?=\d{10}$)", "", digits)
    return digits if re.fullmatch(r"[6-9]\d{9}", digits) else None


def normalize_field(field, value):
    """Validate and normalize one editable value. None clears the field."""
    if value is None:
        return None
    if field == "account_type":
        if value not in ACCOUNT_TYPES:
            raise ApplicationValidationError([_error("INVALID_CHOICE", field, "Choose Savings, Salary or Current.")])
        return value
    if field == "date_of_birth":
        return parse_date_of_birth(value)
    if field == "phone":
        phone = normalize_phone(value)
        if phone is None:
            raise ApplicationValidationError([_error("INVALID_FORMAT", field, "Enter a valid 10-digit Indian mobile number.")])
        return phone
    if field == "address_line":
        return _clean(value, minimum=5, maximum=120, field=field, pattern=_is_address, label="Address")
    if field in {"city", "state"}:
        return _clean(value, minimum=2, maximum=60, field=field, pattern=_is_name, label=FIELD_LABELS[field])
    if field == "postal_code":
        return _clean(value, minimum=6, maximum=6, field=field, pattern=lambda text: bool(re.fullmatch(r"[1-9]\d{5}", text)), label="PIN code")
    if field == "employment_category":
        if value not in EMPLOYMENT_CATEGORIES:
            raise ApplicationValidationError([_error("INVALID_CHOICE", field, "Choose a valid employment category.")])
        return value
    if field == "nominee_name":
        return _clean(value, minimum=2, maximum=80, field=field, pattern=_is_name, label="Nominee name")
    if field == "nominee_relationship":
        if value not in NOMINEE_RELATIONSHIPS:
            raise ApplicationValidationError([_error("INVALID_CHOICE", field, "Choose a valid nominee relationship.")])
        return value
    raise ApplicationValidationError([_error("UNSUPPORTED_FIELD", str(field)[:40], "That field cannot be changed.")])


def required_fields(account_type):
    fields = ["date_of_birth", "phone", "address_line", "city", "state", "postal_code"]
    if account_type and ACCOUNT_TYPES[account_type]["required_employment"]:
        fields.append("employment_category")
    return fields


def required_documents(account_type):
    return list(ACCOUNT_TYPES[account_type]["required_documents"]) if account_type in ACCOUNT_TYPES else []


def _document_path(storage_id, media_type):
    extension = _DOCUMENT_FORMATS.get(media_type, (None,))[0]
    if extension is None or not _STORAGE_ID_PATTERN.fullmatch(storage_id or ""):
        return None
    root = Path(DOCUMENT_ROOT).resolve()
    path = (root / f"{storage_id}{extension}").resolve()
    return path if path.parent == root else None


def _document_is_intact(doc_row):
    path = _document_path(doc_row["storage_id"], doc_row["media_type"])
    try:
        return path is not None and path.is_file() and path.stat().st_size == doc_row["size_bytes"]
    except OSError:
        return False


def _validate(row, documents):
    """Pure deterministic validation of one application row and its documents."""
    errors = []
    missing_fields = []
    account_type = row["account_type"]
    if account_type not in ACCOUNT_TYPES:
        errors.append(_error("ACCOUNT_TYPE_REQUIRED", "account_type", "Choose an account type."))
        missing_fields.append("account_type")
    for field in required_fields(account_type if account_type in ACCOUNT_TYPES else None):
        if not row[field]:
            missing_fields.append(field)
            errors.append(_error("FIELD_REQUIRED", field, f"{FIELD_LABELS[field]} is required."))
    if not row["full_name"]:
        errors.append(_error("FIELD_REQUIRED", "full_name", "Full name is missing from the demo profile."))
    # Re-run format validation on stored values as defence in depth.
    for field in EDITABLE_FIELDS:
        if field == "account_type" or not row[field]:
            continue
        try:
            normalize_field(field, row[field])
        except ApplicationValidationError as exc:
            errors.extend(exc.errors)
    if account_type in ACCOUNT_TYPES:
        allowed = ACCOUNT_TYPES[account_type]["required_employment"]
        category = row["employment_category"]
        if allowed and category and category not in allowed:
            errors.append(_error("EMPLOYMENT_NOT_ELIGIBLE", "employment_category", ACCOUNT_TYPES[account_type]["employment_rule"] + " (synthetic demo rule)"))
    if bool(row["nominee_name"]) != bool(row["nominee_relationship"]):
        errors.append(_error("NOMINEE_INCOMPLETE", "nominee_name", "Provide both nominee name and relationship, or leave both empty."))

    uploaded = {}
    for doc in documents:
        uploaded[doc["document_type"]] = doc
    missing_documents = []
    for document_type in required_documents(account_type):
        doc = uploaded.get(document_type)
        if doc is None:
            missing_documents.append(document_type)
            errors.append(_error("DOCUMENT_REQUIRED", document_type, f"{DOCUMENT_TYPES[document_type]} is required."))
        elif not _document_is_intact(doc):
            missing_documents.append(document_type)
            errors.append(_error("DOCUMENT_UNAVAILABLE", document_type, f"{DOCUMENT_TYPES[document_type]} could not be read; upload it again."))
    completed = [
        field for field in required_fields(account_type if account_type in ACCOUNT_TYPES else None)
        if row[field] and field not in missing_fields
    ]
    return {
        "is_valid": not errors,
        "errors": errors,
        "missing_fields": missing_fields,
        "missing_documents": missing_documents,
        "completed_fields": completed,
    }


def _progress(row, validation):
    has_type = row["account_type"] in ACCOUNT_TYPES
    details_complete = has_type and not [f for f in validation["missing_fields"] if f != "account_type"] and not any(
        item["field"] in EDITABLE_FIELDS for item in validation["errors"]
    )
    documents_complete = has_type and not validation["missing_documents"] and not any(
        item["code"].startswith("DOCUMENT_") for item in validation["errors"]
    )
    submitted = row["status"] == "SUBMITTED"
    review_complete = submitted or row["status"] == "PENDING_CONFIRMATION"
    steps = [
        {"key": "account_type", "label": "Account type", "complete": has_type},
        {"key": "details", "label": "Your details", "complete": details_complete},
        {"key": "documents", "label": "KYC documents", "complete": documents_complete},
        {"key": "review", "label": "Review and confirm", "complete": review_complete},
    ]
    if submitted:
        current = "submitted"
    else:
        current = next((step["key"] for step in steps if not step["complete"]), "review")
        if current == "review" and not validation["is_valid"]:
            current = "details" if not details_complete else "documents"
    done = sum(1 for step in steps if step["complete"])
    return {"current_step": current, "steps": steps, "percent": int(done * 100 / len(steps))}


def _documents_for(conn, row):
    return conn.execute(
        "SELECT * FROM account_application_documents WHERE application_id = ? ORDER BY document_type",
        (row["application_id"],),
    ).fetchall()


def _detail(conn, row, *, include_personal):
    documents = _documents_for(conn, row)
    validation = _validate(row, documents)
    uploaded = {doc["document_type"]: doc for doc in documents}
    account_type = row["account_type"]
    required = required_documents(account_type)
    document_rows = []
    for document_type, label in DOCUMENT_TYPES.items():
        doc = uploaded.get(document_type)
        is_required = document_type in required
        if not is_required and doc is None and account_type is not None and document_type != "ADDRESS_PROOF_DEMO":
            continue
        document_rows.append({
            "document_type": document_type,
            "label": label,
            "required": is_required,
            "uploaded": doc is not None and _document_is_intact(doc),
            "media_type": doc["media_type"] if doc else None,
            "size_bytes": doc["size_bytes"] if doc else None,
            "uploaded_at": doc["uploaded_at"] if doc else None,
            "verification_status": "NOT_VERIFIED" if doc else "NOT_UPLOADED",
        })
    result = {
        "application_ref": row["application_ref"],
        "status": row["status"],
        "status_note": STATUS_NOTES[row["status"]],
        "account_type": account_type,
        "account_type_label": ACCOUNT_TYPES[account_type]["label"] if account_type in ACCOUNT_TYPES else None,
        "documents": document_rows,
        "validation": {
            "is_valid": validation["is_valid"],
            "missing_fields": validation["missing_fields"],
            "missing_documents": validation["missing_documents"],
            "completed_fields": validation["completed_fields"],
            "errors": (
                validation["errors"] if include_personal
                else [{"code": item["code"], "field": item["field"]} for item in validation["errors"]]
            ),
        },
        "progress": _progress(row, validation),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "submitted_at": row["submitted_at"],
        "is_synthetic_demo": True,
        "synthetic_notice": SYNTHETIC_NOTICE,
    }
    if include_personal:
        result["fields"] = {field: row[field] for field in _PERSONAL_FIELDS}
        result["review_digest"] = row["review_digest"]
    return result


def _review_digest(row, documents):
    payload = {
        "ref": row["application_ref"],
        "account_type": row["account_type"],
        "fields": {field: row[field] for field in _PERSONAL_FIELDS},
        "documents": sorted((doc["document_type"], doc["content_sha256"]) for doc in documents),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _invalidate_review(conn, row, owner, reason):
    """An edit after review returns the application to DRAFT."""
    if row["status"] == "PENDING_CONFIRMATION":
        conn.execute(
            "UPDATE account_applications SET status = 'DRAFT', review_digest = NULL, "
            "review_prepared_at = NULL WHERE application_id = ?",
            (row["application_id"],),
        )
        _event(conn, row, owner, "REVIEW_INVALIDATED", reason)


# ---------------------------------------------------------------------------
# Public static information
# ---------------------------------------------------------------------------

def list_account_options():
    """Synthetic account options and their demo document requirements."""
    return {
        "account_types": [
            {
                "account_type": key,
                "label": rule["label"],
                "description": rule["description"],
                "required_documents": [
                    {"document_type": doc, "label": DOCUMENT_TYPES[doc]} for doc in rule["required_documents"]
                ],
                "employment_rule": rule["employment_rule"],
            }
            for key, rule in ACCOUNT_TYPES.items()
        ],
        "minimum_age_years": MINIMUM_AGE_YEARS,
        "accepted_document_formats": ["PDF", "PNG", "JPEG"],
        "maximum_document_megabytes": MAX_DOCUMENT_BYTES // (1024 * 1024),
        "is_synthetic_demo": True,
        "synthetic_notice": SYNTHETIC_NOTICE,
    }


def get_required_documents(account_type=None):
    if account_type is not None and account_type not in ACCOUNT_TYPES:
        raise ApplicationValidationError([_error("INVALID_CHOICE", "account_type", "Choose Savings, Salary or Current.")])
    if account_type is None:
        return list_account_options()
    return {
        "account_type": account_type,
        "required_documents": [
            {"document_type": doc, "label": DOCUMENT_TYPES[doc]}
            for doc in required_documents(account_type)
        ],
        "accepted_document_formats": ["PDF", "PNG", "JPEG"],
        "maximum_document_megabytes": MAX_DOCUMENT_BYTES // (1024 * 1024),
        "is_synthetic_demo": True,
        "synthetic_notice": SYNTHETIC_NOTICE,
    }


# ---------------------------------------------------------------------------
# Application lifecycle
# ---------------------------------------------------------------------------

def has_active_application(principal):
    owner, _profile = _owner(principal)
    conn = database.get_connection()
    try:
        return conn.execute(
            "SELECT 1 FROM account_applications WHERE customer_key = ? "
            "AND status IN ('DRAFT', 'PENDING_CONFIRMATION')",
            (owner,),
        ).fetchone() is not None
    finally:
        conn.close()


def _apply_changes(conn, row, owner, changes):
    """Write already-normalized changes; returns True if anything changed."""
    actual = {key: value for key, value in changes.items() if row[key] != value}
    if not actual:
        return False
    assignments = ", ".join(f"{key} = ?" for key in actual)  # keys are allowlisted
    conn.execute(
        f"UPDATE account_applications SET {assignments}, updated_at = ? WHERE application_id = ?",
        (*actual.values(), _ts(), row["application_id"]),
    )
    return True


def start_application(principal, account_type=None, *, include_personal=True):
    """Resume the customer's unfinished application or start a new draft."""
    owner, profile = _owner(principal)
    if account_type is not None:
        account_type = normalize_field("account_type", account_type)
    for _attempt in range(2):
        try:
            with _transaction() as conn:
                row = conn.execute(
                    "SELECT * FROM account_applications WHERE customer_key = ? "
                    "AND status IN ('DRAFT', 'PENDING_CONFIRMATION')",
                    (owner,),
                ).fetchone()
                resumed = row is not None
                if row is None:
                    now = _ts()
                    ref = "VSA-" + uuid4().hex[:16].upper()
                    conn.execute(
                        "INSERT INTO account_applications (application_ref, customer_key, account_type, status, "
                        "full_name, phone, email, created_at, updated_at) VALUES (?, ?, ?, 'DRAFT', ?, ?, ?, ?, ?)",
                        (
                            ref, owner, account_type, profile["name"],
                            normalize_phone(profile.get("phone") or ""), profile.get("email"), now, now,
                        ),
                    )
                    row = conn.execute("SELECT * FROM account_applications WHERE application_ref = ?", (ref,)).fetchone()
                    _event(conn, row, owner, "CREATED", "Draft application started.", profile)
                detail = _detail(conn, row, include_personal=include_personal)
            detail["resumed"] = resumed
            return detail
        except sqlite3.IntegrityError:
            # A concurrent start won the one-active-application index; resume it.
            continue
    raise ApplicationConflict("An application is already in progress.")


def get_application(principal, application_ref=None, *, include_personal=True):
    owner, _profile = _owner(principal)
    conn = database.get_connection()
    try:
        return _detail(conn, _find(conn, owner, application_ref), include_personal=include_personal)
    finally:
        conn.close()


def get_current_application(principal, *, include_personal=True):
    owner, _profile = _owner(principal)
    conn = database.get_connection()
    try:
        return _detail(conn, _find(conn, owner, None, for_status="active"), include_personal=include_personal)
    finally:
        conn.close()


def list_applications(principal, limit=10):
    owner, _profile = _owner(principal)
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 50:
        raise ApplicationError("Invalid application limit.")
    conn = database.get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM account_applications WHERE customer_key = ? "
            "ORDER BY created_at DESC, application_id DESC LIMIT ?",
            (owner, limit),
        ).fetchall()
        items = []
        for row in rows:
            detail = _detail(conn, row, include_personal=False)
            items.append({
                key: detail[key] for key in (
                    "application_ref", "status", "status_note", "account_type",
                    "account_type_label", "created_at", "updated_at", "submitted_at",
                    "is_synthetic_demo",
                )
            } | {"progress": detail["progress"]})
        return items
    finally:
        conn.close()


def update_application(principal, application_ref, changes, *, include_personal=True):
    """Validate and save editable fields. Server-side validation is authoritative."""
    owner, profile = _owner(principal)
    if not isinstance(changes, dict) or not changes:
        raise ApplicationValidationError([_error("NO_CHANGES", "-", "No changes were provided.")])
    unknown = sorted(str(key)[:40] for key in set(changes) - set(EDITABLE_FIELDS))
    if unknown:
        raise ApplicationValidationError([_error("UNSUPPORTED_FIELD", name, "That field cannot be changed.") for name in unknown])
    normalized = {}
    errors = []
    for field, value in changes.items():
        try:
            normalized[field] = normalize_field(field, value)
        except ApplicationValidationError as exc:
            errors.extend(exc.errors)
    if errors:
        raise ApplicationValidationError(errors)
    with _transaction() as conn:
        row = _find(conn, owner, application_ref)
        if row["status"] == "SUBMITTED":
            raise ApplicationConflict("A submitted application cannot be changed.")
        merged = dict(row)
        merged.update(normalized)
        # Cross-field rule evaluated before anything is written.
        type_rule = ACCOUNT_TYPES.get(merged["account_type"])
        if type_rule and type_rule["required_employment"] and merged["employment_category"] and (
            merged["employment_category"] not in type_rule["required_employment"]
        ) and ("employment_category" in normalized or "account_type" in normalized):
            raise ApplicationValidationError([_error(
                "EMPLOYMENT_NOT_ELIGIBLE", "employment_category",
                type_rule["employment_rule"] + " (synthetic demo rule)",
            )])
        # A partially entered nominee may be saved as a draft; submission
        # validation (not saving) enforces that both nominee fields exist.
        changed = _apply_changes(conn, row, owner, normalized)
        if changed:
            _invalidate_review(conn, row, owner, "Application edited after review.")
            fresh = conn.execute("SELECT * FROM account_applications WHERE application_id = ?", (row["application_id"],)).fetchone()
            _event(conn, fresh, owner, "UPDATED", "Application details saved.", profile)
        row = conn.execute("SELECT * FROM account_applications WHERE application_id = ?", (row["application_id"],)).fetchone()
        return _detail(conn, row, include_personal=include_personal)


def validate_application(principal, application_ref=None, *, include_personal=True):
    owner, _profile = _owner(principal)
    conn = database.get_connection()
    try:
        row = _find(conn, owner, application_ref)
        return _detail(conn, row, include_personal=include_personal)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# KYC demo documents
# ---------------------------------------------------------------------------

def _validate_document_bytes(content, declared_media_type):
    if not isinstance(content, (bytes, bytearray)) or not content:
        raise DocumentRejected("The uploaded file is empty.")
    if len(content) > MAX_DOCUMENT_BYTES:
        raise DocumentRejected(f"The file exceeds the {MAX_DOCUMENT_BYTES // (1024 * 1024)} MB demo limit.")
    content = bytes(content)
    sniffed = next((media for media, (_ext, check) in _DOCUMENT_FORMATS.items() if check(content)), None)
    if sniffed is None:
        raise DocumentRejected("Only PDF, PNG or JPEG demo documents are accepted.")
    declared = (declared_media_type or "application/octet-stream").split(";", 1)[0].strip().lower()
    if declared == "image/jpg":
        declared = "image/jpeg"
    if declared not in {sniffed, "application/octet-stream", ""}:
        raise DocumentRejected("The file type does not match its content.")
    return content, sniffed


def store_document(principal, application_ref, document_type, content, declared_media_type=None):
    """Store one demo document under a generated identifier.

    The client filename is deliberately not a parameter: it is never used for
    storage, logging or display.
    """
    owner, profile = _owner(principal)
    if document_type not in DOCUMENT_TYPES:
        raise ApplicationValidationError([_error("INVALID_CHOICE", "document_type", "Unknown document type.")])
    content, media_type = _validate_document_bytes(content, declared_media_type)
    extension = _DOCUMENT_FORMATS[media_type][0]
    storage_id = uuid4().hex
    root = Path(DOCUMENT_ROOT)
    root.mkdir(parents=True, exist_ok=True)
    path = root.resolve() / f"{storage_id}{extension}"
    old_path = None
    try:
        with path.open("xb") as handle:
            handle.write(content)
        with _transaction() as conn:
            row = _find(conn, owner, application_ref)
            if row["status"] == "SUBMITTED":
                raise ApplicationConflict("A submitted application cannot be changed.")
            if row["account_type"] not in ACCOUNT_TYPES:
                raise ApplicationConflict("Choose an account type before uploading documents.")
            existing = conn.execute(
                "SELECT * FROM account_application_documents WHERE application_id = ? AND document_type = ?",
                (row["application_id"], document_type),
            ).fetchone()
            if (existing is not None
                    and existing["content_sha256"] == hashlib.sha256(content).hexdigest()
                    and _document_is_intact(existing)):
                # A retry of the same upload must not replace the document or
                # invalidate a review of identical content.
                path.unlink(missing_ok=True)
                return _detail(conn, row, include_personal=True)
            if existing is not None:
                old_path = _document_path(existing["storage_id"], existing["media_type"])
                conn.execute("DELETE FROM account_application_documents WHERE document_id = ?", (existing["document_id"],))
            conn.execute(
                "INSERT INTO account_application_documents (application_id, customer_key, document_type, "
                "storage_id, media_type, size_bytes, content_sha256, uploaded_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    row["application_id"], owner, document_type, storage_id, media_type,
                    len(content), hashlib.sha256(content).hexdigest(), _ts(),
                ),
            )
            _invalidate_review(conn, row, owner, "Documents changed after review.")
            conn.execute("UPDATE account_applications SET updated_at = ? WHERE application_id = ?", (_ts(), row["application_id"]))
            fresh = conn.execute("SELECT * FROM account_applications WHERE application_id = ?", (row["application_id"],)).fetchone()
            _event(conn, fresh, owner, "DOCUMENT_UPLOADED", f"{DOCUMENT_TYPES[document_type]} uploaded.", profile)
            detail = _detail(conn, fresh, include_personal=True)
    except Exception:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    if old_path is not None and old_path != path:
        try:
            old_path.unlink(missing_ok=True)
        except OSError:
            pass
    return detail


def delete_document(principal, application_ref, document_type):
    owner, profile = _owner(principal)
    if document_type not in DOCUMENT_TYPES:
        raise ApplicationNotFound("Document not found.")
    with _transaction() as conn:
        row = _find(conn, owner, application_ref)
        if row["status"] == "SUBMITTED":
            raise ApplicationConflict("A submitted application cannot be changed.")
        existing = conn.execute(
            "SELECT * FROM account_application_documents WHERE application_id = ? AND document_type = ?",
            (row["application_id"], document_type),
        ).fetchone()
        if existing is None:
            raise ApplicationNotFound("Document not found.")
        stored_path = _document_path(existing["storage_id"], existing["media_type"])
        conn.execute("DELETE FROM account_application_documents WHERE document_id = ?", (existing["document_id"],))
        _invalidate_review(conn, row, owner, "Documents changed after review.")
        conn.execute("UPDATE account_applications SET updated_at = ? WHERE application_id = ?", (_ts(), row["application_id"]))
        fresh = conn.execute("SELECT * FROM account_applications WHERE application_id = ?", (row["application_id"],)).fetchone()
        _event(conn, fresh, owner, "DOCUMENT_REMOVED", f"{DOCUMENT_TYPES[document_type]} removed.", profile)
        detail = _detail(conn, fresh, include_personal=True)
    if stored_path is not None:
        try:
            stored_path.unlink(missing_ok=True)
        except OSError:
            pass
    return detail


def get_document_status(principal, application_ref=None):
    """Document checklist only; never document contents or storage identifiers."""
    detail = get_application(principal, application_ref, include_personal=False)
    return {
        "application_ref": detail["application_ref"],
        "status": detail["status"],
        "account_type": detail["account_type"],
        "documents": detail["documents"],
        "missing_documents": detail["validation"]["missing_documents"],
        "is_synthetic_demo": True,
        "synthetic_notice": SYNTHETIC_NOTICE,
    }


def open_document(principal, application_ref, document_type):
    """Owner-only document read for the customer's own download."""
    owner, _profile = _owner(principal)
    if document_type not in DOCUMENT_TYPES:
        raise ApplicationNotFound("Document not found.")
    conn = database.get_connection()
    try:
        row = _find(conn, owner, application_ref)
        doc = conn.execute(
            "SELECT * FROM account_application_documents WHERE application_id = ? AND document_type = ? "
            "AND customer_key = ?",
            (row["application_id"], document_type, owner),
        ).fetchone()
    finally:
        conn.close()
    if doc is None or not _document_is_intact(doc):
        raise ApplicationNotFound("Document not found.")
    path = _document_path(doc["storage_id"], doc["media_type"])
    return path.read_bytes(), doc["media_type"]


# ---------------------------------------------------------------------------
# Review, confirmation and submission
# ---------------------------------------------------------------------------

def prepare_submission(principal, application_ref=None, *, include_personal=True):
    """Validate and move to PENDING_CONFIRMATION. This never submits."""
    owner, profile = _owner(principal)
    with _transaction() as conn:
        row = _find(conn, owner, application_ref)
        if row["status"] == "SUBMITTED":
            raise ApplicationConflict("This application has already been submitted.")
        documents = _documents_for(conn, row)
        validation = _validate(row, documents)
        if not validation["is_valid"]:
            _invalidate_review(conn, row, owner, "Application is no longer complete.")
            row = conn.execute("SELECT * FROM account_applications WHERE application_id = ?", (row["application_id"],)).fetchone()
            return {
                "ready": False,
                "status": row["status"],
                "review_digest": None,
                "application": _detail(conn, row, include_personal=include_personal),
            }
        digest = _review_digest(row, documents)
        was_draft = row["status"] == "DRAFT"
        conn.execute(
            "UPDATE account_applications SET status = 'PENDING_CONFIRMATION', review_digest = ?, "
            "review_prepared_at = ?, updated_at = ? WHERE application_id = ?",
            (digest, _ts(), _ts(), row["application_id"]),
        )
        row = conn.execute("SELECT * FROM account_applications WHERE application_id = ?", (row["application_id"],)).fetchone()
        if was_draft:
            _event(conn, row, owner, "REVIEW_PREPARED", "Application review prepared; confirmation required.", profile)
        return {
            "ready": True,
            "status": "PENDING_CONFIRMATION",
            "review_digest": digest,
            "application": _detail(conn, row, include_personal=include_personal),
        }


def return_to_draft(principal, application_ref):
    owner, profile = _owner(principal)
    with _transaction() as conn:
        row = _find(conn, owner, application_ref)
        if row["status"] == "SUBMITTED":
            raise ApplicationConflict("A submitted application cannot be changed.")
        if row["status"] == "PENDING_CONFIRMATION":
            _invalidate_review(conn, row, owner, "Customer returned the application to draft.")
        row = conn.execute("SELECT * FROM account_applications WHERE application_id = ?", (row["application_id"],)).fetchone()
        return _detail(conn, row, include_personal=True)


def submit_application(
    principal, application_ref, *, confirmed, confirmation_method, review_digest,
):
    """Submit after explicit, authenticated button confirmation. Idempotent."""
    if confirmed is not True or confirmation_method != "button":
        raise ApplicationConflict("Explicit button confirmation is required to submit.")
    return _submit_reviewed_application(principal, application_ref, review_digest, "button")


def submit_application_from_conversation(principal, application_ref, *, user_text, input_mode, review_digest):
    """Submission is restricted to the existing Review & Submit page."""
    raise ApplicationConflict("Review and submit the application on the Review & Submit page.")


def _submit_reviewed_application(principal, application_ref, review_digest, confirmation_method):
    owner, profile = _owner(principal)
    with _transaction() as conn:
        row = _find(conn, owner, application_ref)
        if row["status"] == "SUBMITTED":
            return {
                "application_ref": row["application_ref"],
                "status": "SUBMITTED",
                "submitted_at": row["submitted_at"],
                "already_submitted": True,
                "status_note": STATUS_NOTES["SUBMITTED"],
                "is_synthetic_demo": True,
            }
        if row["status"] != "PENDING_CONFIRMATION" or not row["review_digest"]:
            raise ApplicationConflict("Review the application before confirming submission.")
        if not isinstance(review_digest, str) or not hmac.compare_digest(row["review_digest"], review_digest):
            raise ApplicationConflict("The application changed after review. Review it again.")
        documents = _documents_for(conn, row)
        validation = _validate(row, documents)
        if not validation["is_valid"] or not hmac.compare_digest(row["review_digest"], _review_digest(row, documents)):
            _invalidate_review(conn, row, owner, "Application is no longer complete.")
            raise ApplicationConflict("The application changed after review. Review it again.")
        now = _ts()
        cursor = conn.execute(
            "UPDATE account_applications SET status = 'SUBMITTED', submitted_at = ?, updated_at = ?, "
            "review_digest = NULL WHERE application_id = ? AND status = 'PENDING_CONFIRMATION'",
            (now, now, row["application_id"]),
        )
        if cursor.rowcount != 1:
            raise ApplicationConflict("The application could not be submitted.")
        _event(conn, row, owner, "SUBMITTED", f"Application submitted after explicit {confirmation_method} confirmation.", profile)
        return {
            "application_ref": row["application_ref"],
            "status": "SUBMITTED",
            "submitted_at": now,
            "already_submitted": False,
            "status_note": STATUS_NOTES["SUBMITTED"],
            "is_synthetic_demo": True,
        }


def get_application_status(principal, application_ref=None, account_type=None):
    """Owner-scoped status lookup for the tracker and the assistant."""
    owner, _profile = _owner(principal)
    if account_type is not None and account_type not in ACCOUNT_TYPES:
        raise ApplicationValidationError([_error("INVALID_CHOICE", "account_type", "Choose Savings, Salary or Current.")])
    conn = database.get_connection()
    try:
        if application_ref is not None or account_type is None:
            try:
                row = _find(conn, owner, application_ref)
            except ApplicationNotFound:
                return {"found": False, "is_synthetic_demo": True, "synthetic_notice": SYNTHETIC_NOTICE}
        else:
            row = conn.execute(
                "SELECT * FROM account_applications WHERE customer_key = ? AND account_type = ? "
                "ORDER BY created_at DESC, application_id DESC LIMIT 1",
                (owner, account_type),
            ).fetchone()
            if row is None:
                return {"found": False, "is_synthetic_demo": True, "synthetic_notice": SYNTHETIC_NOTICE}
        detail = _detail(conn, row, include_personal=False)
        events = conn.execute(
            "SELECT event_type, summary, occurred_at FROM account_application_events "
            "WHERE application_id = ? ORDER BY occurred_at DESC, event_id DESC LIMIT 10",
            (row["application_id"],),
        ).fetchall()
    finally:
        conn.close()
    return {
        "found": True,
        "application_ref": detail["application_ref"],
        "status": detail["status"],
        "status_note": detail["status_note"],
        "account_type": detail["account_type"],
        "account_type_label": detail["account_type_label"],
        "progress": detail["progress"],
        "created_at": detail["created_at"],
        "submitted_at": detail["submitted_at"],
        "updated_at": detail["updated_at"],
        "events": [dict(event) for event in events],
        "is_synthetic_demo": True,
        "synthetic_notice": SYNTHETIC_NOTICE,
    }
