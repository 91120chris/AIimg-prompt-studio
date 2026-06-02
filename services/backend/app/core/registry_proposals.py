import json

from app.db.models import RegistryPatchProposalRecord
from app.schemas.registry import (
    RegistryPatchProposalResponse,
    RegistryPatchProposalValidationResponse,
)


def registry_proposal_response(
    record: RegistryPatchProposalRecord,
) -> RegistryPatchProposalResponse:
    return RegistryPatchProposalResponse(
        proposal_id=record.proposal_id,
        registry_kind=record.registry_kind,  # type: ignore[arg-type]
        change_kind=record.change_kind,  # type: ignore[arg-type]
        item_id=record.target_id,
        status=record.status,  # type: ignore[arg-type]
        summary=record.summary,
        diff_text=record.diff_text,
        proposed_content=record.proposed_content,
        validation=_json_dict(record.validation_json),
        applied_version_id=record.applied_version_id,
        created_at=record.created_at,
    )


def validate_registry_proposal(
    record: RegistryPatchProposalRecord,
) -> RegistryPatchProposalValidationResponse:
    errors: list[str] = []
    target_id = (record.target_id or "").strip()
    content = (record.proposed_content or "").strip()

    if record.registry_kind not in {"skill", "template"}:
        errors.append("registry_kind must be skill or template.")
    if record.change_kind not in {"create", "update"}:
        errors.append("change_kind must be create or update.")
    if not target_id:
        errors.append("target_id is required.")
    if not content:
        errors.append("proposed_content is required.")

    if record.registry_kind == "template" and content:
        template_id, template_errors = validate_template_content(content)
        errors.extend(template_errors)
        if target_id and template_id and template_id != target_id:
            errors.append("Template JSON id must match target_id.")
    elif record.registry_kind == "skill" and content:
        errors.extend(validate_skill_content(content, target_id))

    return RegistryPatchProposalValidationResponse(
        valid=not errors,
        registry_kind=record.registry_kind,  # type: ignore[arg-type]
        item_id=target_id or None,
        errors=errors,
    )


def validate_template_content(content: str) -> tuple[str | None, list[str]]:
    try:
        payload = json.loads(content)
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
                value = question.get(key)
                if not isinstance(value, str) or not value.strip():
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
    return template_id.strip() if isinstance(template_id, str) else None, errors


def validate_skill_content(content: str, target_id: str | None = None) -> list[str]:
    stripped = content.strip()
    errors: list[str] = []
    if not stripped:
        errors.append("Skill content cannot be empty.")
    if len(stripped) < 20:
        errors.append("Skill content is too short to be useful.")
    if "#" not in stripped[:120]:
        errors.append("Skill content should start with a Markdown heading.")
    if target_id and target_id.lower().endswith(".md"):
        errors.append("Skill target_id should be an id, not a filename.")
    return errors


def serialize_validation(
    validation: RegistryPatchProposalValidationResponse,
) -> str:
    return json.dumps(validation.model_dump(), ensure_ascii=False, sort_keys=True)


def _json_dict(raw: str | None) -> dict | None:
    if not raw:
        return None
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None
