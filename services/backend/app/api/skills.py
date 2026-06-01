from fastapi import APIRouter, HTTPException, Request
from sqlmodel import select

from app.core.session_workspace import new_id
from app.db.models import RegistryPatchProposalRecord, SkillVersionRecord
from app.db.session import new_session
from app.schemas.registry import (
    RegistryItemResponse,
    RegistryPatchProposalCreateRequest,
    RegistryPatchProposalResponse,
)

router = APIRouter(prefix="/skills", tags=["skills"])


def _engine(request: Request):
    return request.app.state.engine


def _proposal_response(record: RegistryPatchProposalRecord) -> RegistryPatchProposalResponse:
    return RegistryPatchProposalResponse(
        proposal_id=record.proposal_id,
        registry_kind="skill",
        status=record.status,
        diff_text=record.diff_text,
        created_at=record.created_at,
    )


def _item_response(record: SkillVersionRecord) -> RegistryItemResponse:
    return RegistryItemResponse(
        registry_kind="skill",
        item_id=record.skill_id,
        latest_version_id=record.skill_version_id,
        content=record.content,
        created_at=record.created_at,
    )


@router.get("", response_model=list[RegistryItemResponse])
def list_skills(request: Request) -> list[RegistryItemResponse]:
    with new_session(_engine(request)) as db:
        records = db.exec(
            select(SkillVersionRecord).order_by(
                SkillVersionRecord.created_at,
                SkillVersionRecord.skill_version_id,
            )
        ).all()
        latest_by_id: dict[str, SkillVersionRecord] = {}
        for record in records:
            latest_by_id[record.skill_id] = record
        return [_item_response(record) for record in latest_by_id.values()]


@router.get("/{skill_id}", response_model=RegistryItemResponse)
def get_skill(skill_id: str, request: Request) -> RegistryItemResponse:
    with new_session(_engine(request)) as db:
        record = db.exec(
            select(SkillVersionRecord)
            .where(SkillVersionRecord.skill_id == skill_id)
            .order_by(
                SkillVersionRecord.created_at.desc(),
                SkillVersionRecord.skill_version_id.desc(),
            )
        ).first()
        if record is None:
            raise HTTPException(
                status_code=404,
                detail={"code": "skill_not_found", "message": "Skill not found."},
            )
        return _item_response(record)


@router.post("/patch-proposals", response_model=RegistryPatchProposalResponse)
def create_skill_patch_proposal(
    payload: RegistryPatchProposalCreateRequest,
    request: Request,
) -> RegistryPatchProposalResponse:
    diff_text = payload.diff_text.strip()
    if not diff_text:
        raise HTTPException(
            status_code=422,
            detail={"code": "empty_patch_proposal", "message": "Patch proposal cannot be empty."},
        )
    with new_session(_engine(request)) as db:
        record = RegistryPatchProposalRecord(
            proposal_id=new_id("proposal"),
            registry_kind="skill",
            status="pending",
            diff_text=diff_text,
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        return _proposal_response(record)


@router.post("/patch-proposals/{proposal_id}/approve", response_model=RegistryPatchProposalResponse)
def approve_skill_patch_proposal(proposal_id: str, request: Request) -> RegistryPatchProposalResponse:
    return _update_proposal_status(proposal_id, "approved", request)


@router.post("/patch-proposals/{proposal_id}/reject", response_model=RegistryPatchProposalResponse)
def reject_skill_patch_proposal(proposal_id: str, request: Request) -> RegistryPatchProposalResponse:
    return _update_proposal_status(proposal_id, "rejected", request)


def _update_proposal_status(
    proposal_id: str,
    status: str,
    request: Request,
) -> RegistryPatchProposalResponse:
    with new_session(_engine(request)) as db:
        record = db.get(RegistryPatchProposalRecord, proposal_id)
        if record is None or record.registry_kind != "skill":
            raise HTTPException(
                status_code=404,
                detail={"code": "proposal_not_found", "message": "Patch proposal not found."},
            )
        record.status = status
        db.add(record)
        db.commit()
        db.refresh(record)
        return _proposal_response(record)
