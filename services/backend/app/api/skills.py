from fastapi import APIRouter, HTTPException, Request
from sqlmodel import select

from app.core.registry_store import apply_registry_patch_proposal
from app.core.session_workspace import new_id
from app.db.models import AppSettingRecord, RegistryPatchProposalRecord, SkillVersionRecord
from app.db.session import new_session
from app.schemas.registry import (
    RegistryItemResponse,
    RegistryPatchProposalCreateRequest,
    RegistryPatchProposalResponse,
    SkillEnabledPatchRequest,
)

router = APIRouter(prefix="/skills", tags=["skills"])


def _engine(request: Request):
    return request.app.state.engine


def _proposal_response(record: RegistryPatchProposalRecord) -> RegistryPatchProposalResponse:
    return RegistryPatchProposalResponse(
        proposal_id=record.proposal_id,
        registry_kind="skill",
        item_id=record.target_id,
        status=record.status,
        diff_text=record.diff_text,
        proposed_content=record.proposed_content,
        applied_version_id=record.applied_version_id,
        created_at=record.created_at,
    )


def _item_response(record: SkillVersionRecord) -> RegistryItemResponse:
    return RegistryItemResponse(
        registry_kind="skill",
        item_id=record.skill_id,
        latest_version_id=record.skill_version_id,
        content=record.content,
        enabled=True,
        created_at=record.created_at,
    )


def _clean_optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _skill_enabled_key(skill_id: str) -> str:
    return f"skill_enabled:{skill_id}"


def _is_skill_enabled(db, skill_id: str) -> bool:
    record = db.get(AppSettingRecord, _skill_enabled_key(skill_id))
    if record is None:
        return True
    return record.value != "0"


def _latest_skill(db, skill_id: str) -> SkillVersionRecord | None:
    return db.exec(
        select(SkillVersionRecord)
        .where(SkillVersionRecord.skill_id == skill_id)
        .order_by(
            SkillVersionRecord.created_at.desc(),
            SkillVersionRecord.skill_version_id.desc(),
        )
    ).first()


def _skill_response(db, record: SkillVersionRecord) -> RegistryItemResponse:
    response = _item_response(record)
    response.enabled = _is_skill_enabled(db, record.skill_id)
    return response


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
        return [_skill_response(db, record) for record in latest_by_id.values()]


@router.get("/patch-proposals", response_model=list[RegistryPatchProposalResponse])
def list_skill_patch_proposals(request: Request) -> list[RegistryPatchProposalResponse]:
    with new_session(_engine(request)) as db:
        records = db.exec(
            select(RegistryPatchProposalRecord)
            .where(RegistryPatchProposalRecord.registry_kind == "skill")
            .order_by(RegistryPatchProposalRecord.created_at.desc())
        ).all()
        return [_proposal_response(record) for record in records]


@router.get("/{skill_id}", response_model=RegistryItemResponse)
def get_skill(skill_id: str, request: Request) -> RegistryItemResponse:
    with new_session(_engine(request)) as db:
        record = _latest_skill(db, skill_id)
        if record is None:
            raise HTTPException(
                status_code=404,
                detail={"code": "skill_not_found", "message": "Skill not found."},
            )
        return _skill_response(db, record)


@router.patch("/{skill_id}/enabled", response_model=RegistryItemResponse)
def patch_skill_enabled(
    skill_id: str,
    payload: SkillEnabledPatchRequest,
    request: Request,
) -> RegistryItemResponse:
    with new_session(_engine(request)) as db:
        record = _latest_skill(db, skill_id)
        if record is None:
            raise HTTPException(
                status_code=404,
                detail={"code": "skill_not_found", "message": "Skill not found."},
            )
        setting = AppSettingRecord(
            key=_skill_enabled_key(skill_id),
            value="1" if payload.enabled else "0",
        )
        db.merge(setting)
        db.commit()
        return _skill_response(db, record)


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
    target_id = _clean_optional_text(payload.item_id)
    if payload.proposed_content is not None:
        if not payload.proposed_content.strip():
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "empty_proposed_content",
                    "message": "Proposed content cannot be empty.",
                },
            )
        if target_id is None:
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "missing_registry_item_id",
                    "message": "item_id is required when proposed_content is provided.",
                },
            )
    with new_session(_engine(request)) as db:
        record = RegistryPatchProposalRecord(
            proposal_id=new_id("proposal"),
            registry_kind="skill",
            target_id=target_id,
            status="pending",
            diff_text=diff_text,
            proposed_content=payload.proposed_content,
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
        if record.status != "pending":
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "proposal_already_closed",
                    "message": "Patch proposal is already closed.",
                },
            )
        if status == "approved":
            apply_registry_patch_proposal(db, record)
        record.status = status
        db.add(record)
        db.commit()
        db.refresh(record)
        return _proposal_response(record)
