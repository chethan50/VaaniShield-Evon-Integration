"""Pydantic contracts for VaaniShield v1 APIs and feature modules."""

from datetime import date, datetime
from typing import Annotated, Any, Generic, Literal, TypeVar, Union
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator


LanguageHint = Literal["auto", "kn-IN", "en-IN", "hi-IN"]
LanguageCode = Literal["kn-IN", "en-IN", "hi-IN"]
InputMode = Literal["text", "speech"]
OutputMode = Literal["text", "speech", "speech_and_text"]
DemoCustomerKey = Literal["demo-arjun", "demo-priya", "demo-rahul"]
SupportTicketCategory = Literal[
    "account", "card", "transaction", "digital_banking", "loan", "deposit", "other"
]
SupportTicketPriority = Literal["LOW", "MEDIUM", "HIGH", "URGENT"]
SupportTicketStatus = Literal[
    "OPEN", "IN_PROGRESS", "WAITING_FOR_CUSTOMER", "RESOLVED", "CLOSED"
]
SupportTicketOrigin = Literal["customer_form", "assistant_confirmed", "legacy_import"]


class ContractModel(BaseModel):
    """Base model with strict fields for predictable API contracts."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class DemoSessionRequest(ContractModel):
    """Select one of the fixed synthetic accounts in DEMO mode."""

    customer_key: DemoCustomerKey


class DemoLoginRequest(ContractModel):
    """Authenticate with demo credentials (email and password)."""

    email: str = Field(min_length=3, max_length=255)
    password: str = Field(min_length=1, max_length=128)


class SigninAuditLogEntry(ContractModel):
    log_id: int
    customer_id: int | None = None
    email_attempted: str
    timestamp_utc: str
    timestamp_ist: str
    login_result: str
    ip_address: str | None = None
    user_agent: str | None = None
    device_platform: str | None = None


class DemoSessionResponse(ContractModel):
    customer_key: DemoCustomerKey
    display_name: str
    expires_at: datetime
    auth_mode: Literal["DEMO"] = "DEMO"


class DemoLogoutResponse(ContractModel):
    status: Literal["signed_out"] = "signed_out"


class ConversationCreateRequest(ContractModel):
    title: str | None = Field(default=None, min_length=1, max_length=160)


class ConversationSummary(ContractModel):
    conversation_id: UUID
    title: str | None = None
    created_at: datetime
    updated_at: datetime
    status: Literal["active", "archived", "closed"]


class ConversationDetail(ConversationSummary):
    turn_count: int = Field(ge=0)


class ConversationTurn(ContractModel):
    turn_id: UUID
    conversation_id: UUID
    sequence_number: int = Field(gt=0)
    role: Literal["user", "assistant"]
    input_type: Literal["text", "audio"] | None = None
    user_text_or_transcript: str | None = None
    input_language: LanguageCode | None = None
    detected_languages: list[LanguageCode] = Field(default_factory=list)
    is_code_mixed: bool = False
    assistant_response_text: str | None = None
    response_language: LanguageCode | None = None
    requested_output_mode: Literal["text", "speech", "both"] | None = None
    processing_status: Literal["pending", "processing", "completed", "failed"]
    created_at: datetime


class AudioUploadResponse(ContractModel):
    audio_upload_id: UUID
    media_type: str
    size_bytes: int = Field(gt=0)
    expires_at: datetime


class DocumentAttachment(ContractModel):
    attachment_id: str | None = None
    file_name: str
    file_type: str
    file_size_bytes: int = 0
    extracted_text: str = ""
    summary_metadata: dict[str, Any] = Field(default_factory=dict)


class AccountReviewBinding(ContractModel):
    application_ref: str = Field(pattern=r"^VSA-[A-F0-9]{16}$")
    review_digest: str = Field(pattern=r"^[a-f0-9]{64}$")


class AssistantTurnRequest(ContractModel):
    conversation_id: UUID | None = None
    # Optional owner-scoped reference to an assessment returned earlier by
    # the scam API. It is a resource pointer, never identity authority.
    scam_assessment_id: UUID | None = None
    client_turn_id: UUID
    input_mode: InputMode
    output_mode: OutputMode
    text: str | None = Field(default=None, min_length=1, max_length=4000)
    audio_upload_id: UUID | None = None
    language_hint: LanguageHint = "auto"
    response_language: LanguageCode | None = None
    attachments: list[DocumentAttachment] = Field(default_factory=list)
    account_review: AccountReviewBinding | None = None

    @model_validator(mode="after")
    def validate_input_source(self):
        if self.input_mode == "text":
            if not self.text:
                raise ValueError("text is required when input_mode is 'text'.")
            if self.audio_upload_id is not None:
                raise ValueError("audio_upload_id is only valid for speech input.")
        else:
            if self.audio_upload_id is None:
                raise ValueError("audio_upload_id is required for speech input.")
            if self.text is not None:
                raise ValueError("text must be omitted for speech input.")
        return self


class DetectedLanguage(ContractModel):
    language: LanguageCode
    confidence: float | None = Field(default=None, ge=0, le=1)


class SpeechOutput(ContractModel):
    audio_url: str
    media_type: Literal["audio/wav", "audio/mpeg", "audio/ogg"]
    language: LanguageCode
    expires_at: datetime


class ToolExecutionResult(ContractModel):
    tool_call_id: UUID
    tool_name: str
    status: Literal["succeeded", "failed", "needs_confirmation"]
    summary: str
    data: dict[str, Any] | None = None
    error_code: str | None = None


class CardBlockAction(ContractModel):
    action_type: Literal["block_card"]
    card_id: str
    reason: str | None = Field(default=None, max_length=500)


class TransactionInvestigationAction(ContractModel):
    action_type: Literal["investigate_transaction"]
    transaction_id: str
    reason: str = Field(min_length=1, max_length=1000)


class SupportTicketCreateAction(ContractModel):
    action_type: Literal["create_support_ticket"]
    category: SupportTicketCategory
    priority: SupportTicketPriority
    subject: str = Field(min_length=1, max_length=160)
    description: str | None = Field(default=None, max_length=4000)
    language: LanguageCode | None = None
    amount: float | None = None


InvestigationStatus = Literal[
    "OPEN", "UNDER_REVIEW", "WAITING_FOR_INFORMATION", "RESOLVED", "CLOSED"
]
InvestigationCategory = Literal[
    "MESSAGE_SCAM", "CALL_SCAM", "TRANSACTION_RISK", "ACCOUNT_SECURITY", "OTHER"
]
InvestigationEvidenceType = Literal[
    "MESSAGE_ASSESSMENT", "CALL_ASSESSMENT", "TRANSACTION_RISK_ASSESSMENT"
]


class InvestigationEvidenceReference(ContractModel):
    evidence_type: InvestigationEvidenceType
    assessment_id: UUID


class CreateInvestigationAction(ContractModel):
    action_type: Literal["create_investigation"]
    title: str = Field(min_length=3, max_length=120)
    category: InvestigationCategory
    summary: str = Field(min_length=8, max_length=1500)
    evidence: list[InvestigationEvidenceReference] = Field(default_factory=list, max_length=5)


class LinkInvestigationEvidenceAction(ContractModel):
    action_type: Literal["link_investigation_evidence"]
    investigation_ref: str = Field(pattern=r"^INV-[A-F0-9]{32}$")
    evidence: InvestigationEvidenceReference
    expected_updated_at: datetime | None = None


class AddInvestigationNoteAction(ContractModel):
    action_type: Literal["add_investigation_note"]
    investigation_ref: str = Field(pattern=r"^INV-[A-F0-9]{32}$")
    note: str = Field(min_length=3, max_length=1500)
    expected_updated_at: datetime | None = None


class ChangeInvestigationStatusAction(ContractModel):
    action_type: Literal["change_investigation_status"]
    investigation_ref: str = Field(pattern=r"^INV-[A-F0-9]{32}$")
    status: Literal["UNDER_REVIEW", "WAITING_FOR_INFORMATION", "RESOLVED"]
    expected_updated_at: datetime | None = None


class CloseInvestigationAction(ContractModel):
    action_type: Literal["close_investigation"]
    investigation_ref: str = Field(pattern=r"^INV-[A-F0-9]{32}$")
    expected_updated_at: datetime | None = None


InvestigationActionProposal = Annotated[
    Union[
        CreateInvestigationAction, LinkInvestigationEvidenceAction,
        AddInvestigationNoteAction, ChangeInvestigationStatusAction,
        CloseInvestigationAction,
    ],
    Field(discriminator="action_type"),
]


class SupportTicketCloseConfirmation(ContractModel):
    ticket_ref: str = Field(pattern=r"^VS-[A-F0-9]{32}$")
    subject: str
    current_status: SupportTicketStatus
    conversation_id: UUID
    confirmation_method: Literal["button"] = "button"


ActionProposal = Annotated[
    Union[
        CardBlockAction, TransactionInvestigationAction, SupportTicketCreateAction,
        CreateInvestigationAction, LinkInvestigationEvidenceAction,
        AddInvestigationNoteAction, ChangeInvestigationStatusAction,
        CloseInvestigationAction,
    ],
    Field(discriminator="action_type"),
]


class PendingAction(ContractModel):
    confirmation_id: UUID
    conversation_id: UUID
    original_turn_id: UUID
    summary: str
    action: ActionProposal
    expires_at: datetime


class AssistantTurnResponse(ContractModel):
    conversation_id: UUID
    turn_id: UUID
    assistant_turn_id: UUID | None = None
    input_mode: InputMode
    output_mode: OutputMode
    transcript: str | None = None
    detected_languages: list[DetectedLanguage] = Field(default_factory=list)
    is_code_mixed: bool = False
    assistant_text: str
    response_language: LanguageCode | None = None
    speech_output: SpeechOutput | None = None
    tool_results: list[ToolExecutionResult] = Field(default_factory=list)
    scam_assessment: "ScamAssessmentDetail | None" = None
    transaction_risk_assessment: "TransactionRiskAssessmentDetail | None" = None
    pending_action: PendingAction | None = None
    support_close_confirmation: SupportTicketCloseConfirmation | None = None
    created_at: datetime

    @model_validator(mode="after")
    def validate_response_mode(self):
        if self.input_mode == "speech" and not self.transcript:
            raise ValueError("transcript is required for speech input.")
        if not self.assistant_text:
            raise ValueError("assistant_text is required for every output_mode.")
        return self


class ConfirmationRequest(ContractModel):
    confirmed: bool
    confirmation_method: Literal["button", "voice"] = "button"
    conversation_id: UUID
    original_turn_id: UUID
    action_type: Literal["create_support_ticket"]


class ConfirmationResponse(ContractModel):
    confirmation_id: UUID
    status: Literal["executed", "declined", "expired"]
    action_result: ToolExecutionResult | None = None


class PageQuery(ContractModel):
    limit: int = Field(default=25, ge=1, le=100)
    cursor: str | None = Field(default=None, max_length=512)


ItemT = TypeVar("ItemT")


class CursorPage(ContractModel, Generic[ItemT]):
    items: list[ItemT]
    next_cursor: str | None = None
    has_more: bool


class DatePeriod(ContractModel):
    start_date: date | None = None
    end_date: date | None = None
    year: int | None = Field(default=None, ge=2000, le=2100)
    month: int | None = Field(default=None, ge=1, le=12)

    @model_validator(mode="after")
    def validate_period(self):
        has_start = self.start_date is not None
        has_end = self.end_date is not None
        has_year = self.year is not None
        has_month = self.month is not None

        if has_start != has_end:
            raise ValueError("start_date and end_date must be supplied together.")
        if has_year != has_month:
            raise ValueError("year and month must be supplied together.")
        if has_start and has_year:
            raise ValueError("Use a date range or a calendar month, not both.")
        if has_start and self.start_date > self.end_date:
            raise ValueError("start_date must be on or before end_date.")
        return self


class CalendarMonth(ContractModel):
    year: int = Field(ge=2000, le=2100)
    month: int = Field(ge=1, le=12)


class TransactionFilters(DatePeriod, PageQuery):
    merchant: str | None = Field(default=None, max_length=160)
    category: str | None = Field(default=None, max_length=80)
    transaction_type: Literal["DEBIT", "CREDIT"] | None = None
    minimum_amount: float | None = Field(default=None, ge=0)
    maximum_amount: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_amount_range(self):
        if (
            self.minimum_amount is not None
            and self.maximum_amount is not None
            and self.minimum_amount > self.maximum_amount
        ):
            raise ValueError("minimum_amount cannot exceed maximum_amount.")
        return self


class SelfServiceAccount(ContractModel):
    account_type: str
    account_number: str  # Masked value; full account numbers are never returned.
    balance: float
    status: str


class SelfServiceBalance(ContractModel):
    total_balance: float
    currency: str = "INR"


class SelfServiceTransaction(ContractModel):
    transaction_type: Literal["DEBIT", "CREDIT"]
    amount: float
    merchant: str | None = None
    category: str | None = None
    status: str
    reference_id: str | None = None
    # Additive opaque handle for authenticated transaction-risk workflows.
    # Existing reference_id values are preserved for dashboard compatibility.
    transaction_ref: str | None = Field(default=None, pattern=r"^TR-[A-F0-9]{48}$")
    transaction_date: str
    description: str | None = None


class SelfServiceCard(ContractModel):
    card_type: str
    last_four: str = Field(pattern=r"^\d{4}$")
    status: str
    expiry_date: str | None = None
    card_network: str | None = None
    cardholder_name: str | None = None
    daily_limit: float | None = Field(default=None, ge=0)
    monthly_limit: float | None = Field(default=None, ge=0)
    credit_limit: float | None = Field(default=None, ge=0)
    available_credit_limit: float | None = Field(default=None, ge=0)
    outstanding_amount: float | None = Field(default=None, ge=0)
    account_balance: float | None = Field(default=None, ge=0)


class SelfServiceLoan(ContractModel):
    loan_type: str
    principal_amount: float = Field(ge=0)
    interest_rate: float = Field(ge=0)
    tenure_months: int = Field(gt=0)
    outstanding_amount: float | None = Field(default=None, ge=0)
    status: str
    monthly_emi: float | None = Field(default=None, ge=0)
    remaining_tenure_months: int | None = Field(default=None, ge=0)
    next_emi_date: str | None = None
    completion_date: str | None = None


class SelfServiceFinancialSummary(ContractModel):
    total_spent: float
    total_credited: float
    net_cash_flow: float
    insights: list[dict[str, Any]] = Field(default_factory=list)


class AccountBalance(ContractModel):
    account_id: str
    account_type: str
    masked_account_number: str
    balance: float
    currency: str = Field(default="INR", min_length=3, max_length=3)
    status: str


class TransactionRecord(ContractModel):
    transaction_id: str
    transaction_type: Literal["DEBIT", "CREDIT"]
    amount: float
    currency: str = Field(default="INR", min_length=3, max_length=3)
    merchant: str | None = None
    category: str | None = None
    status: str
    transaction_date: datetime
    description: str | None = None


class AccountListResponse(ContractModel):
    items: list[AccountBalance]


class CardSummary(ContractModel):
    card_id: str
    card_type: str
    last_four: str = Field(pattern=r"^\d{4}$")
    status: str
    expiry_date: str | None = None


class LoanSummary(ContractModel):
    loan_id: str
    loan_type: str
    principal_amount: float = Field(ge=0)
    interest_rate: float = Field(ge=0)
    tenure_months: int = Field(gt=0)
    outstanding_amount: float | None = Field(default=None, ge=0)
    status: str


class CardListResponse(ContractModel):
    items: list[CardSummary]


class LoanListResponse(ContractModel):
    items: list[LoanSummary]


class TransactionPage(CursorPage[TransactionRecord]):
    pass


class TransactionRiskAssessmentRequest(ContractModel):
    transaction_ref: str = Field(pattern=r"^TR-[A-F0-9]{48}$", max_length=51)


class TransactionRiskIndicator(ContractModel):
    indicator_code: Literal[
        "unusual_amount", "first_seen_merchant", "unusual_frequency",
        "rapid_transaction_cluster", "unusual_timing",
    ]
    severity: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    explanation: str
    evidence: list[str]
    score_contribution: int = Field(ge=0, le=100)


class TransactionRiskHistorySufficiency(ContractModel):
    level: Literal["ADEQUATE", "LIMITED", "INSUFFICIENT"]
    matching_history_count: int = Field(ge=0)
    minimum_for_amount_baseline: int = Field(ge=0)
    time_coverage_days: int | None = Field(default=None, ge=0)
    frequency_baseline_ready: bool
    excluded_history_count: int = Field(ge=0)


class TransactionRiskAssessmentResult(ContractModel):
    risk_level: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL", "UNKNOWN"]
    risk_score: int = Field(ge=0, le=100)
    indicators: list[TransactionRiskIndicator]
    evidence: list[str]
    evidence_strength: Literal["NONE", "LIMITED", "MODERATE", "STRONG"]
    explanation: str
    warning: str
    recommended_actions: list[str]
    history_sufficiency: TransactionRiskHistorySufficiency
    engine_version: str
    context_notes: list[str] = Field(default_factory=list)


class TransactionRiskTransactionProjection(ContractModel):
    transaction_ref: str = Field(pattern=r"^TR-[A-F0-9]{48}$")
    transaction_type: Literal["DEBIT", "CREDIT"]
    amount: float = Field(ge=0)
    merchant: str | None = None
    category: str | None = None
    status: str
    transaction_date: str | None = None


class TransactionRiskAssessmentResponse(ContractModel):
    assessment_id: UUID
    transaction: TransactionRiskTransactionProjection
    assessment: TransactionRiskAssessmentResult
    created_at: datetime
    reused: bool = False


class TransactionRiskAssessmentSummary(ContractModel):
    assessment_id: UUID
    transaction_ref: str = Field(pattern=r"^TR-[A-F0-9]{48}$")
    transaction: TransactionRiskTransactionProjection
    risk_level: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL", "UNKNOWN"]
    risk_score: int = Field(ge=0, le=100)
    evidence_strength: Literal["NONE", "LIMITED", "MODERATE", "STRONG"]
    engine_version: str
    created_at: datetime


class TransactionRiskAssessmentDetail(ContractModel):
    assessment_id: UUID
    transaction_ref: str = Field(pattern=r"^TR-[A-F0-9]{48}$")
    transaction: TransactionRiskTransactionProjection
    assessment: TransactionRiskAssessmentResult
    created_at: datetime


class FinancialSummaryResponse(ContractModel):
    period: DatePeriod
    total_spent: float
    total_credited: float
    net_cash_flow: float
    currency: str = "INR"


class CategorySpending(ContractModel):
    category: str
    total_amount: float
    transaction_count: int = Field(ge=0)
    currency: str = "INR"


class MerchantSpending(ContractModel):
    merchant: str
    total_spent: float
    transaction_count: int = Field(ge=0)
    currency: str = "INR"


class MonthlyComparisonResponse(ContractModel):
    current_period: CalendarMonth
    previous_period: CalendarMonth
    current_spending: float
    previous_spending: float
    difference: float
    percentage_change: float | None = None
    currency: str = "INR"


class BudgetRecommendation(ContractModel):
    code: str
    message: str
    category: str | None = None
    amount: float | None = None
    currency: str = "INR"


class BudgetRecommendationsResponse(ContractModel):
    period: DatePeriod
    methodology: str
    recommendations: list[BudgetRecommendation]


class ScamMessageAssessmentRequest(ContractModel):
    message_text: str = Field(min_length=1, max_length=12000)
    language_hint: LanguageHint = "auto"


class ScamCallAssessmentRequest(ContractModel):
    audio_upload_id: UUID
    language_hint: LanguageHint = "auto"


class TransactionScamAssessmentRequest(ContractModel):
    transaction_id: str
    reported_reason: str | None = Field(default=None, max_length=1000)


class ScamRiskAssessment(ContractModel):
    assessment_id: UUID
    risk_level: Literal["low", "medium", "high", "critical", "unknown"]
    confidence: float | None = Field(default=None, ge=0, le=1)
    indicators: list[str] = Field(default_factory=list)
    customer_warning: str
    recommended_steps: list[str] = Field(default_factory=list)
    transcript: str | None = None
    detected_languages: list[DetectedLanguage] = Field(default_factory=list)
    created_at: datetime


class ScamMessageAssessmentCreate(ContractModel):
    """Authenticated Phase 2 message assessment; text is transient input only."""

    message_text: str = Field(min_length=1, max_length=12000)
    language_hint: LanguageHint = "auto"


class ScamAssessmentIndicator(ContractModel):
    category: str
    evidence: list[str]
    severity: Literal["UNKNOWN", "LOW", "MEDIUM", "HIGH", "CRITICAL"]
    explanation: str


class ScamAssessmentDetail(ContractModel):
    assessment_id: UUID
    source_type: Literal["typed_message", "voice_call"]
    risk_level: Literal["UNKNOWN", "LOW", "MEDIUM", "HIGH", "CRITICAL"]
    indicators: list[ScamAssessmentIndicator]
    explanation: str
    evidence_strength: Literal["NONE", "LIMITED", "MODERATE", "STRONG"]
    warning: str
    recommended_steps: list[str]
    detected_languages: list[LanguageCode]
    is_code_mixed: bool
    language_status: Literal["supported", "uncertain", "unsupported"]
    engine_version: str
    created_at: datetime
    redacted_transcript: str | None = None
    transcript_expires_at: datetime | None = None
    transcript_status: Literal["NOT_STORED", "AVAILABLE", "EXPIRED"] = "NOT_STORED"


class ScamAssessmentSummary(ContractModel):
    assessment_id: UUID
    source_type: Literal["typed_message", "voice_call"]
    risk_level: Literal["UNKNOWN", "LOW", "MEDIUM", "HIGH", "CRITICAL"]
    evidence_strength: Literal["NONE", "LIMITED", "MODERATE", "STRONG"]
    detected_languages: list[LanguageCode]
    language_status: Literal["supported", "uncertain", "unsupported"]
    created_at: datetime


class ScamVoiceAssessmentResponse(ScamAssessmentDetail):
    """Redacted transcript is returned for immediate review and retained 30 days."""

    transcript: str
    transcript_expires_at: datetime
    transcription_status: Literal["completed"] = "completed"
    transcription_warning: str = (
        "Automated transcription may contain errors. Review the transcript before relying on this assessment."
    )


class ScamReport(ContractModel):
    report_id: UUID
    source_type: Literal["MANUAL_MESSAGE", "INCOMING_SMS", "VOICE_CALL"]
    risk_level: Literal["CRITICAL", "HIGH", "MEDIUM", "LOW", "UNKNOWN"]
    risk_score: float | None = None
    analysis_timestamp: datetime
    indicators: list[ScamAssessmentIndicator]
    explanation: str
    warning: str
    recommended_actions: list[str]
    detected_languages: list[LanguageCode]
    language_status: Literal["supported", "uncertain", "unsupported"]
    is_code_mixed: bool
    engine_version: str
    redacted_message_preview: str | None = None
    redacted_call_transcript: str | None = None
    transcript_expires_at: datetime | None = None
    content_availability: Literal[
        "REDACTED_SMS_PREVIEW", "REDACTED_TRANSCRIPT", "TRANSCRIPT_EXPIRED", "SOURCE_CONTENT_NOT_STORED"
    ]
    sender: str | None = None
    source_package: str | None = None
    device_id: str | None = None


class ScamReportPage(CursorPage[ScamReport]):
    pass


class InvestigationEvidenceSummary(ContractModel):
    evidence_type: InvestigationEvidenceType
    assessment_id: UUID
    risk_level: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL", "UNKNOWN"]
    evidence_strength: Literal["NONE", "LIMITED", "MODERATE", "STRONG"]
    summary: str
    created_at: datetime


class InvestigationEvent(ContractModel):
    event_type: Literal[
        "CREATED", "EVIDENCE_LINKED", "INFORMATION_ADDED",
        "STATUS_CHANGED", "CLOSED",
    ]
    summary: str
    occurred_at: datetime
    note: str | None = None
    from_status: InvestigationStatus | None = None
    to_status: InvestigationStatus | None = None
    evidence_type: InvestigationEvidenceType | None = None
    assessment_id: UUID | None = None


class InvestigationSummary(ContractModel):
    is_synthetic_demo: Literal[True] = True
    investigation_ref: str = Field(pattern=r"^INV-[A-F0-9]{32}$")
    title: str
    summary: str
    category: InvestigationCategory
    status: InvestigationStatus
    created_at: datetime
    updated_at: datetime
    resolved_at: datetime | None = None
    evidence_count: int = Field(ge=0)


class InvestigationDetail(InvestigationSummary):
    evidence: list[InvestigationEvidenceSummary] = Field(default_factory=list)
    events: list[InvestigationEvent] = Field(default_factory=list)


class InvestigationCreateRequest(ContractModel):
    title: str = Field(min_length=3, max_length=120)
    category: InvestigationCategory
    summary: str = Field(min_length=8, max_length=1500)
    evidence: list[InvestigationEvidenceReference] = Field(default_factory=list, max_length=5)
    confirmed: Literal[True]


class InvestigationEvidenceLinkRequest(ContractModel):
    evidence: InvestigationEvidenceReference
    confirmed: Literal[True]


class InvestigationNoteRequest(ContractModel):
    note: str = Field(min_length=3, max_length=1500)
    confirmed: Literal[True]


class InvestigationStatusUpdateRequest(ContractModel):
    status: Literal["UNDER_REVIEW", "WAITING_FOR_INFORMATION", "RESOLVED"]
    confirmed: Literal[True]


class InvestigationCloseRequest(ContractModel):
    confirmed: Literal[True]


class InvestigationConfirmationRequest(ContractModel):
    confirmed: bool
    confirmation_method: Literal["button"]
    conversation_id: UUID
    original_turn_id: UUID
    action_type: Literal[
        "create_investigation", "link_investigation_evidence",
        "add_investigation_note", "change_investigation_status",
        "close_investigation",
    ]


class SupportTicketCreateRequest(ContractModel):
    category: SupportTicketCategory
    priority: SupportTicketPriority
    subject: str = Field(min_length=1, max_length=160)
    description: str = Field(min_length=1, max_length=4000)
    conversation_id: UUID | None = None
    language: LanguageCode | None = None


class SupportTicketSummary(ContractModel):
    ticket_ref: str = Field(pattern=r"^VS-[A-F0-9]{32}$")
    category: SupportTicketCategory
    subject: str
    priority: SupportTicketPriority
    status: SupportTicketStatus
    created_at: datetime
    updated_at: datetime
    origin: SupportTicketOrigin
    is_synthetic_demo: bool
    conversation_id: UUID | None = None
    language: LanguageCode | None = None


class SupportTicketDetail(SupportTicketSummary):
    description: str


class SupportTicketStatusEvent(ContractModel):
    from_status: SupportTicketStatus | None = None
    to_status: SupportTicketStatus
    actor_type: Literal["system", "customer"]
    occurred_at: datetime
    reason_code: Literal["legacy_import", "case_created", "customer_closed"]


class SupportTicketStatusUpdateRequest(ContractModel):
    status: Literal["CLOSED"]
    confirmed: Literal[True]
    confirmation_method: Literal["button"]


class ResolutionDraftRequest(ContractModel):
    customer_message: str | None = Field(default=None, max_length=4000)


class ResolutionDraftResponse(ContractModel):
    ticket_ref: str
    draft: str
    informational_only: Literal[True] = True
    official_bank_resolution: Literal[False] = False
    ticket_status_changed: Literal[False] = False


AccountApplicationType = Literal["SAVINGS", "SALARY", "CURRENT"]
AccountApplicationStatus = Literal["DRAFT", "PENDING_CONFIRMATION", "SUBMITTED"]
AccountApplicationEmploymentCategory = Literal[
    "SALARIED", "SELF_EMPLOYED", "BUSINESS_OWNER",
    "STUDENT", "RETIRED", "GOVERNMENT_EMPLOYEE", "OTHER",
]
AccountApplicationDocumentType = Literal[
    "AADHAAR_DEMO", "PAN_DEMO", "ADDRESS_PROOF_DEMO",
]


class AccountApplicationStartRequest(ContractModel):
    account_type: AccountApplicationType | None = None


class AccountApplicationUpdateRequest(ContractModel):
    account_type: AccountApplicationType | None = None
    date_of_birth: str | None = Field(default=None, max_length=32)
    phone: str | None = Field(default=None, max_length=24)
    address_line: str | None = Field(default=None, max_length=200)
    city: str | None = Field(default=None, max_length=80)
    state: str | None = Field(default=None, max_length=80)
    postal_code: str | None = Field(default=None, max_length=16)
    employment_category: AccountApplicationEmploymentCategory | None = None
    nominee_name: str | None = Field(default=None, max_length=120)
    nominee_relationship: str | None = Field(default=None, max_length=80)


class AccountApplicationSubmitRequest(ContractModel):
    confirmed: bool
    confirmation_method: Literal["button"]
    review_digest: str = Field(min_length=64, max_length=64)


DeviceStatus = Literal["PENDING", "ACTIVE", "PAUSED", "REVOKED"]
DeviceStatusTransition = Literal[
    "ACTIVE", "PAUSED", "REVOKED", "PAUSE", "RESUME", "REVOKE"
]
SmsSenderType = Literal[
    "dlt_header",
    "phone_number",
    "phone",
    "alphanumeric",
    "short_code",
    "shortcode",
    "contact",
    "unknown",
]
SmsRiskLevel = Literal["UNKNOWN", "LOW", "MEDIUM", "HIGH", "CRITICAL"]
SmsEvidenceStrength = Literal["NONE", "LIMITED", "MODERATE", "STRONG"]


class PairTokenCreateRequest(ContractModel):
    device_name: str | None = Field(default=None, min_length=1, max_length=120)


class PairTokenCreateResponse(ContractModel):
    pairing_code: str = Field(min_length=8, max_length=8)
    expires_at: datetime
    expires_in_seconds: int = Field(gt=0)


class DevicePairRequest(ContractModel):
    pairing_code: str = Field(min_length=1, max_length=32)
    device_id: str = Field(min_length=3, max_length=128, pattern=r"^[A-Za-z0-9._:-]+$")
    device_name: str = Field(min_length=1, max_length=120)
    device_model: str = Field(min_length=1, max_length=120)
    android_version: str = Field(min_length=1, max_length=32)
    app_version: str = Field(min_length=1, max_length=32)


class DevicePairResponse(ContractModel):
    device_id: str
    device_name: str
    device_model: str
    android_version: str
    app_version: str
    status: Literal["ACTIVE"] = "ACTIVE"
    device_token: str
    paired_at: datetime


class PairedDeviceSummary(ContractModel):
    device_id: str
    device_name: str
    device_model: str | None = None
    android_version: str | None = None
    app_version: str | None = None
    status: DeviceStatus
    created_at: datetime
    paired_at: datetime | None = None
    last_seen_at: datetime | None = None
    revoked_at: datetime | None = None
    updated_at: datetime


class PairedDeviceListResponse(ContractModel):
    items: list[PairedDeviceSummary]


class DeviceStatusUpdateRequest(ContractModel):
    status: DeviceStatusTransition


class DeviceTokenRotateResponse(ContractModel):
    device_id: str
    status: DeviceStatus
    device_token: str
    updated_at: datetime


class DeviceHeartbeatRequest(ContractModel):
    app_version: str | None = Field(default=None, min_length=1, max_length=32)


class DeviceHeartbeatResponse(ContractModel):
    device_id: str
    status: Literal["ACTIVE"]
    forwarding_permitted: bool = True
    last_seen_at: datetime
    server_time: datetime


class SmsIngestRequest(ContractModel):
    source_package: str = Field(min_length=1, max_length=128)
    sender: str = Field(min_length=1, max_length=120)
    sender_type: str = Field(min_length=1, max_length=32)
    message_text: str = Field(min_length=1, max_length=12000)
    message_timestamp: str = Field(min_length=1, max_length=64)
    client_dedup_key: str = Field(min_length=1, max_length=256)


class SmsIngestResponse(ContractModel):
    sms_event_id: UUID
    device_id: str
    deduplicated: bool = False
    risk_level: SmsRiskLevel
    evidence_strength: SmsEvidenceStrength
    alert_generated: bool
    received_at: datetime


class SmsThreatSummary(ContractModel):
    sms_event_id: UUID
    device_id: str
    source_package: str
    sender: str
    sender_type: SmsSenderType
    redacted_message: str
    message_timestamp: datetime
    received_at: datetime
    risk_level: SmsRiskLevel
    evidence_strength: SmsEvidenceStrength
    detected_languages: list[LanguageCode]
    code_mixed: bool
    language_status: Literal["supported", "uncertain", "unsupported"]
    alert_generated: bool
    acknowledged: bool
    acknowledged_at: datetime | None = None


class SmsThreatDetail(SmsThreatSummary):
    indicators: list[ScamAssessmentIndicator]
    explanation: str
    warning: str
    recommendations: list[str]
    engine_version: str


class FieldValidationError(ContractModel):
    field: str
    message: str


class ApiError(ContractModel):
    code: str
    message: str
    request_id: UUID
    retryable: bool = False
    field_errors: list[FieldValidationError] = Field(default_factory=list)


class ErrorResponse(ContractModel):
    error: ApiError


# Resolve the response's forward reference after all feature contracts have
# been declared in this module.
AssistantTurnResponse.model_rebuild()
