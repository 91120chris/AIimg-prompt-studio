import copy
import json

from fastapi import APIRouter, HTTPException, Request
from sqlmodel import select

from app.core.registry_store import apply_registry_patch_proposal
from app.core.session_workspace import new_id
from app.db.models import RegistryPatchProposalRecord, TemplateVersionRecord
from app.db.session import new_session
from app.schemas.registry import (
    RegistryItemResponse,
    RegistryPatchProposalCreateRequest,
    RegistryPatchProposalResponse,
    TemplateContentRequest,
    TemplateDuplicateRequest,
    TemplatePreviewResponse,
    TemplateValidationResponse,
)

router = APIRouter(prefix="/templates", tags=["templates"])


def _engine(request: Request):
    return request.app.state.engine


def _proposal_response(record: RegistryPatchProposalRecord) -> RegistryPatchProposalResponse:
    return RegistryPatchProposalResponse(
        proposal_id=record.proposal_id,
        registry_kind="template",
        item_id=record.target_id,
        status=record.status,
        diff_text=record.diff_text,
        proposed_content=record.proposed_content,
        applied_version_id=record.applied_version_id,
        created_at=record.created_at,
    )


def _item_response(record: TemplateVersionRecord) -> RegistryItemResponse:
    return RegistryItemResponse(
        registry_kind="template",
        item_id=record.template_id,
        latest_version_id=record.template_version_id,
        content=record.content,
        created_at=record.created_at,
    )


def _clean_optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _latest_template(db, template_id: str) -> TemplateVersionRecord | None:
    return db.exec(
        select(TemplateVersionRecord)
        .where(TemplateVersionRecord.template_id == template_id)
        .order_by(
            TemplateVersionRecord.created_at.desc(),
            TemplateVersionRecord.template_version_id.desc(),
        )
    ).first()


def _template_exists(db, template_id: str) -> bool:
    return _latest_template(db, template_id) is not None


def _parse_template_content(content: str) -> tuple[dict[str, object] | None, list[str]]:
    stripped = content.strip()
    if not stripped:
        return None, ["Template content cannot be empty."]
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError as error:
        return None, [f"Template content must be valid JSON: {error.msg}."]
    if not isinstance(payload, dict):
        return None, ["Template content must be a JSON object."]

    errors: list[str] = []
    template_id = payload.get("id")
    if not isinstance(template_id, str) or not template_id.strip():
        errors.append("id must be a non-empty string.")
    name = payload.get("name")
    if not isinstance(name, str) or not name.strip():
        errors.append("name must be a non-empty string.")
    applies_to = payload.get("applies_to")
    if not isinstance(applies_to, list) or not applies_to:
        errors.append("applies_to must be a non-empty array.")
    elif any(item not in {"t2i", "i2i"} for item in applies_to):
        errors.append("applies_to may only contain t2i or i2i.")
    questions = payload.get("questions")
    if not isinstance(questions, list) or not questions:
        errors.append("questions must be a non-empty array.")
    else:
        for index, question in enumerate(questions, start=1):
            if not isinstance(question, dict):
                errors.append(f"questions[{index}] must be an object.")
                continue
            for key in ["id", "type", "label"]:
                if not isinstance(question.get(key), str) or not question.get(key, "").strip():
                    errors.append(f"questions[{index}].{key} must be a non-empty string.")
            question_type = question.get("type")
            if question_type not in {"text", "textarea", "single_choice", "choice", "boolean", "scale"}:
                errors.append(f"questions[{index}].type is not supported.")
            if question_type in {"single_choice", "choice"}:
                options = question.get("options")
                if not isinstance(options, list) or not options:
                    errors.append(f"questions[{index}].options must be a non-empty array.")
    if not isinstance(payload.get("prompt_structure"), dict):
        errors.append("prompt_structure must be an object.")
    return payload if not errors else None, errors


def _template_id_from_payload(payload: dict[str, object]) -> str:
    return str(payload["id"]).strip()


def _template_name_from_payload(payload: dict[str, object]) -> str:
    return str(payload["name"]).strip()


def _question_option(option: object) -> dict[str, str | None]:
    if isinstance(option, dict):
        value = str(option.get("value") or option.get("id") or option.get("label") or "").strip()
        label = str(option.get("label") or value).strip()
        description = option.get("description")
        return {
            "value": value,
            "label": label,
            "description": str(description).strip() if description is not None else None,
        }
    value = str(option).strip()
    return {"value": value, "label": value, "description": None}


def _preview_question(question: dict[str, object]) -> dict[str, object]:
    question_id = str(question["id"]).strip()
    label = str(question["label"]).strip()
    prompt = str(question.get("prompt") or question.get("description") or label).strip()
    required = bool(question.get("required", True))
    question_type = question["type"]
    base = {
        "question_id": question_id,
        "label": label,
        "prompt": prompt,
        "required": required,
    }
    if question_type in {"text", "textarea"}:
        return {
            **base,
            "kind": "text",
            "placeholder": question.get("placeholder"),
            "max_length": question.get("max_length"),
        }
    if question_type in {"single_choice", "choice"}:
        return {
            **base,
            "kind": "choice",
            "options": [_question_option(option) for option in question.get("options", [])],
            "allow_multiple": bool(question.get("allow_multiple", False)),
        }
    if question_type == "boolean":
        return {
            **base,
            "kind": "boolean",
            "true_label": str(question.get("true_label") or "是"),
            "false_label": str(question.get("false_label") or "否"),
        }
    return {
        **base,
        "kind": "scale",
        "min_value": int(question.get("min_value", 1)),
        "max_value": int(question.get("max_value", 5)),
        "step": int(question.get("step", 1)),
    }


def _preview_payload(payload: dict[str, object]) -> dict[str, object]:
    return {
        "questionnaire_id": f"preview_{_template_id_from_payload(payload)}",
        "title": f"{_template_name_from_payload(payload)} Preview",
        "description": str(payload.get("description") or ""),
        "questions": [
            _preview_question(question)
            for question in payload.get("questions", [])
            if isinstance(question, dict)
        ],
    }


def _validation_response(content: str) -> TemplateValidationResponse:
    payload, errors = _parse_template_content(content)
    if payload is None:
        return TemplateValidationResponse(valid=False, errors=errors)
    return TemplateValidationResponse(
        valid=True,
        template_id=_template_id_from_payload(payload),
        name=_template_name_from_payload(payload),
        errors=[],
    )


def _validated_payload_or_422(content: str) -> dict[str, object]:
    payload, errors = _parse_template_content(content)
    if payload is None:
        raise HTTPException(
            status_code=422,
            detail={"code": "invalid_template", "message": "Template is invalid.", "errors": errors},
        )
    return payload


@router.get("", response_model=list[RegistryItemResponse])
def list_templates(request: Request) -> list[RegistryItemResponse]:
    with new_session(_engine(request)) as db:
        records = db.exec(
            select(TemplateVersionRecord).order_by(
                TemplateVersionRecord.created_at,
                TemplateVersionRecord.template_version_id,
            )
        ).all()
        latest_by_id: dict[str, TemplateVersionRecord] = {}
        for record in records:
            latest_by_id[record.template_id] = record
        return [_item_response(record) for record in latest_by_id.values()]


@router.post("", response_model=RegistryItemResponse)
def create_template(payload: TemplateContentRequest, request: Request) -> RegistryItemResponse:
    template_payload = _validated_payload_or_422(payload.content)
    template_id = _template_id_from_payload(template_payload)
    with new_session(_engine(request)) as db:
        if _template_exists(db, template_id):
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "template_id_exists",
                    "message": "Template ID already exists.",
                },
            )
        record = TemplateVersionRecord(
            template_version_id=new_id("tmplv"),
            template_id=template_id,
            content=json.dumps(template_payload, ensure_ascii=False, indent=2, sort_keys=True),
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        return _item_response(record)


@router.post("/validate", response_model=TemplateValidationResponse)
def validate_template(payload: TemplateContentRequest) -> TemplateValidationResponse:
    return _validation_response(payload.content)


@router.post("/preview", response_model=TemplatePreviewResponse)
def preview_template(payload: TemplateContentRequest) -> TemplatePreviewResponse:
    template_payload, errors = _parse_template_content(payload.content)
    if template_payload is None:
        return TemplatePreviewResponse(valid=False, errors=errors)
    return TemplatePreviewResponse(
        valid=True,
        template_id=_template_id_from_payload(template_payload),
        questionnaire=_preview_payload(template_payload),
        errors=[],
    )


@router.get("/patch-proposals", response_model=list[RegistryPatchProposalResponse])
def list_template_patch_proposals(request: Request) -> list[RegistryPatchProposalResponse]:
    with new_session(_engine(request)) as db:
        records = db.exec(
            select(RegistryPatchProposalRecord)
            .where(RegistryPatchProposalRecord.registry_kind == "template")
            .order_by(RegistryPatchProposalRecord.created_at.desc())
        ).all()
        return [_proposal_response(record) for record in records]


@router.get("/{template_id}", response_model=RegistryItemResponse)
def get_template(template_id: str, request: Request) -> RegistryItemResponse:
    with new_session(_engine(request)) as db:
        record = _latest_template(db, template_id)
        if record is None:
            raise HTTPException(
                status_code=404,
                detail={"code": "template_not_found", "message": "Template not found."},
            )
        return _item_response(record)


@router.put("/{template_id}", response_model=RegistryItemResponse)
def update_template(
    template_id: str,
    payload: TemplateContentRequest,
    request: Request,
) -> RegistryItemResponse:
    template_payload = _validated_payload_or_422(payload.content)
    content_id = _template_id_from_payload(template_payload)
    if content_id != template_id:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "template_id_mismatch",
                "message": "Template JSON id must match the URL template_id.",
            },
        )
    with new_session(_engine(request)) as db:
        if not _template_exists(db, template_id):
            raise HTTPException(
                status_code=404,
                detail={"code": "template_not_found", "message": "Template not found."},
            )
        record = TemplateVersionRecord(
            template_version_id=new_id("tmplv"),
            template_id=template_id,
            content=json.dumps(template_payload, ensure_ascii=False, indent=2, sort_keys=True),
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        return _item_response(record)


@router.post("/{template_id}/duplicate", response_model=RegistryItemResponse)
def duplicate_template(
    template_id: str,
    payload: TemplateDuplicateRequest,
    request: Request,
) -> RegistryItemResponse:
    with new_session(_engine(request)) as db:
        source_record = _latest_template(db, template_id)
        if source_record is None:
            raise HTTPException(
                status_code=404,
                detail={"code": "template_not_found", "message": "Template not found."},
            )
        source_payload = _validated_payload_or_422(source_record.content)
        requested_id = payload.new_template_id.strip() if payload.new_template_id else ""
        new_template_id = requested_id or f"{template_id}-copy"
        if _template_exists(db, new_template_id):
            suffix = 2
            while _template_exists(db, f"{new_template_id}-{suffix}"):
                suffix += 1
            new_template_id = f"{new_template_id}-{suffix}"
        duplicated = copy.deepcopy(source_payload)
        duplicated["id"] = new_template_id
        duplicated["name"] = (
            payload.new_name.strip()
            if payload.new_name and payload.new_name.strip()
            else f"{_template_name_from_payload(source_payload)} Copy"
        )
        record = TemplateVersionRecord(
            template_version_id=new_id("tmplv"),
            template_id=new_template_id,
            content=json.dumps(duplicated, ensure_ascii=False, indent=2, sort_keys=True),
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        return _item_response(record)


@router.post("/patch-proposals", response_model=RegistryPatchProposalResponse)
def create_template_patch_proposal(
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
            registry_kind="template",
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
def approve_template_patch_proposal(
    proposal_id: str,
    request: Request,
) -> RegistryPatchProposalResponse:
    return _update_proposal_status(proposal_id, "approved", request)


@router.post("/patch-proposals/{proposal_id}/reject", response_model=RegistryPatchProposalResponse)
def reject_template_patch_proposal(
    proposal_id: str,
    request: Request,
) -> RegistryPatchProposalResponse:
    return _update_proposal_status(proposal_id, "rejected", request)


def _update_proposal_status(
    proposal_id: str,
    status: str,
    request: Request,
) -> RegistryPatchProposalResponse:
    with new_session(_engine(request)) as db:
        record = db.get(RegistryPatchProposalRecord, proposal_id)
        if record is None or record.registry_kind != "template":
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
