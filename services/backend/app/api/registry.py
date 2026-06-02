from fastapi import APIRouter, HTTPException, Request
from sqlmodel import select

from app.core.registry_proposals import (
    registry_proposal_response,
    serialize_validation,
    validate_registry_proposal,
)
from app.core.registry_store import apply_registry_patch_proposal
from app.db.models import RegistryPatchProposalRecord
from app.db.session import new_session
from app.schemas.registry import (
    RegistryPatchProposalPatchRequest,
    RegistryPatchProposalResponse,
    RegistryPatchProposalValidationResponse,
)

router = APIRouter(prefix="/registry", tags=["registry"])


def _engine(request: Request):
    return request.app.state.engine


def _proposal_or_404(db, proposal_id: str) -> RegistryPatchProposalRecord:
    record = db.get(RegistryPatchProposalRecord, proposal_id)
    if record is None:
        raise HTTPException(
            status_code=404,
            detail={"code": "proposal_not_found", "message": "Patch proposal not found."},
        )
    return record


def _ensure_pending(record: RegistryPatchProposalRecord) -> None:
    if record.status != "pending":
        raise HTTPException(
            status_code=409,
            detail={
                "code": "proposal_already_closed",
                "message": "Patch proposal is already closed.",
            },
        )


@router.get("/patch-proposals", response_model=list[RegistryPatchProposalResponse])
def list_registry_patch_proposals(request: Request) -> list[RegistryPatchProposalResponse]:
    with new_session(_engine(request)) as db:
        records = db.exec(
            select(RegistryPatchProposalRecord).order_by(
                RegistryPatchProposalRecord.created_at.desc()
            )
        ).all()
        return [registry_proposal_response(record) for record in records]


@router.patch(
    "/patch-proposals/{proposal_id}",
    response_model=RegistryPatchProposalResponse,
)
def patch_registry_patch_proposal(
    proposal_id: str,
    payload: RegistryPatchProposalPatchRequest,
    request: Request,
) -> RegistryPatchProposalResponse:
    with new_session(_engine(request)) as db:
        record = _proposal_or_404(db, proposal_id)
        _ensure_pending(record)
        if payload.proposed_content is not None:
            record.proposed_content = payload.proposed_content
        if payload.diff_text is not None:
            record.diff_text = payload.diff_text
        if payload.summary is not None:
            record.summary = payload.summary
        if payload.item_id is not None:
            record.target_id = payload.item_id.strip() or None
        record.validation_json = None
        db.add(record)
        db.commit()
        db.refresh(record)
        return registry_proposal_response(record)


@router.post(
    "/patch-proposals/{proposal_id}/validate",
    response_model=RegistryPatchProposalValidationResponse,
)
def validate_registry_patch_proposal(
    proposal_id: str,
    request: Request,
) -> RegistryPatchProposalValidationResponse:
    with new_session(_engine(request)) as db:
        record = _proposal_or_404(db, proposal_id)
        validation = validate_registry_proposal(record)
        record.validation_json = serialize_validation(validation)
        db.add(record)
        db.commit()
        return validation


@router.post(
    "/patch-proposals/{proposal_id}/approve",
    response_model=RegistryPatchProposalResponse,
)
def approve_registry_patch_proposal(
    proposal_id: str,
    request: Request,
) -> RegistryPatchProposalResponse:
    with new_session(_engine(request)) as db:
        record = _proposal_or_404(db, proposal_id)
        _ensure_pending(record)
        validation = validate_registry_proposal(record)
        record.validation_json = serialize_validation(validation)
        if not validation.valid:
            db.add(record)
            db.commit()
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "proposal_invalid",
                    "message": "Patch proposal did not pass validation.",
                    "errors": validation.errors,
                },
            )
        apply_registry_patch_proposal(db, record)
        record.status = "approved"
        db.add(record)
        db.commit()
        db.refresh(record)
        return registry_proposal_response(record)


@router.post(
    "/patch-proposals/{proposal_id}/reject",
    response_model=RegistryPatchProposalResponse,
)
def reject_registry_patch_proposal(
    proposal_id: str,
    request: Request,
) -> RegistryPatchProposalResponse:
    with new_session(_engine(request)) as db:
        record = _proposal_or_404(db, proposal_id)
        _ensure_pending(record)
        record.status = "rejected"
        db.add(record)
        db.commit()
        db.refresh(record)
        return registry_proposal_response(record)
