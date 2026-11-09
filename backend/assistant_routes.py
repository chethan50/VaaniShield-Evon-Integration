"""Authenticated entry point for unified assistant turns."""

from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from backend.api_models import (
    AssistantTurnRequest,
    AssistantTurnResponse,
    ConfirmationRequest,
    ConfirmationResponse,
    InvestigationConfirmationRequest,
)
from backend.auth import get_authenticated_principal
from services.assistant_service import (
    AssistantServiceError,
    process_assistant_turn,
)
from services.conversation_service import ConversationNotFoundError
from services.pending_action_service import (
    PendingActionConfigurationError,
    PendingActionConflict,
    PendingActionNotFound,
    confirm_pending_action,
)
from services.principal import CustomerPrincipal
from services.investigation_service import (
    InvestigationConfirmationConflict,
    InvestigationConfirmationNotFound,
    confirm_investigation_action,
)


router = APIRouter(prefix="/api/v1/assistant", tags=["Assistant"])


@router.post("/turns", response_model=AssistantTurnResponse)
def post_assistant_turn(
    payload: AssistantTurnRequest,
    principal: CustomerPrincipal = Depends(get_authenticated_principal),
):
    try:
        return process_assistant_turn(principal, payload)
    except ConversationNotFoundError:
        raise HTTPException(status_code=404, detail="Conversation not found.") from None
    except PermissionError:
        raise HTTPException(status_code=401, detail="Authentication required.") from None
    except AssistantServiceError as exc:
        status_by_code = {
            "PROVIDER_REQUEST_TOO_LARGE": 502,
            "PROVIDER_RATE_LIMITED": 429,
            "PROVIDER_TIMEOUT": 504,
            "PROVIDER_AUTHENTICATION_FAILED": 502,
            "PROVIDER_CONNECTION_FAILED": 503,
            "PROVIDER_UNAVAILABLE": 503,
            "TURN_ALREADY_COMPLETED": 409,
            "TURN_CONFLICT": 409,
            "TURN_IN_PROGRESS": 409,
        }
        raise HTTPException(
            status_code=status_by_code.get(exc.code, 503), detail=str(exc),
            headers={"X-Vaani-Error-Code": exc.code},
        ) from None


@router.post("/attachments")
def post_assistant_attachment(
    file: UploadFile = File(...),
    conversation_id: str | None = Form(default=None),
    principal: CustomerPrincipal = Depends(get_authenticated_principal),
):
    from services.document_processing_service import process_uploaded_attachment
    content = file.file.read() if hasattr(file, "file") else getattr(file, "content", b"")
    result = process_uploaded_attachment(
        file_name=file.filename or "uploaded_document",
        file_bytes=content,
        customer_key=principal.customer_key,
        conversation_id=conversation_id,
    )
    if not result.is_safe and result.error:
        raise HTTPException(status_code=400, detail=result.error)
    return result.to_dict()


@router.post(
    "/confirmations/{confirmation_id}",
    response_model=ConfirmationResponse,
)
def post_assistant_confirmation(
    confirmation_id: UUID,
    payload: ConfirmationRequest,
    principal: CustomerPrincipal = Depends(get_authenticated_principal),
):
    try:
        return confirm_pending_action(principal, confirmation_id, payload)
    except PermissionError:
        raise HTTPException(status_code=401, detail="Authentication required.") from None
    except PendingActionNotFound:
        raise HTTPException(status_code=404, detail="Confirmation not found.") from None
    except PendingActionConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from None
    except PendingActionConfigurationError:
        raise HTTPException(
            status_code=503,
            detail="Pending-action confirmation is temporarily unavailable.",
        ) from None


@router.post(
    "/investigation-confirmations/{confirmation_id}",
    response_model=ConfirmationResponse,
)
def post_investigation_confirmation(
    confirmation_id: UUID,
    payload: InvestigationConfirmationRequest,
    principal: CustomerPrincipal = Depends(get_authenticated_principal),
):
    try:
        return confirm_investigation_action(principal, confirmation_id, payload)
    except PermissionError:
        raise HTTPException(status_code=401, detail="Authentication required.") from None
    except InvestigationConfirmationNotFound:
        raise HTTPException(status_code=404, detail="Confirmation not found.") from None
    except InvestigationConfirmationConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from None
