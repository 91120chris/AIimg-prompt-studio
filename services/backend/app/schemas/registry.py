from typing import Literal

from app.schemas.base import StrictBaseModel

RegistryKind = Literal["skill", "template"]
ProposalStatus = Literal["pending", "approved", "rejected"]
ProposalChangeKind = Literal["create", "update"]
WorkflowMode = Literal["t2i", "i2i"]


class RegistryItemResponse(StrictBaseModel):
    registry_kind: RegistryKind
    item_id: str
    latest_version_id: str | None = None
    content: str
    enabled: bool | None = None
    created_at: str | None = None


class RegistryPatchProposalCreateRequest(StrictBaseModel):
    diff_text: str
    item_id: str | None = None
    proposed_content: str | None = None
    change_kind: ProposalChangeKind = "update"
    summary: str | None = None


class RegistryPatchProposalResponse(StrictBaseModel):
    proposal_id: str
    registry_kind: RegistryKind
    change_kind: ProposalChangeKind = "update"
    item_id: str | None = None
    status: ProposalStatus
    summary: str | None = None
    diff_text: str
    proposed_content: str | None = None
    validation: dict | None = None
    applied_version_id: str | None = None
    created_at: str


class RegistryPatchProposalPatchRequest(StrictBaseModel):
    proposed_content: str | None = None
    diff_text: str | None = None
    summary: str | None = None
    item_id: str | None = None


class RegistryPatchProposalValidationResponse(StrictBaseModel):
    valid: bool
    registry_kind: RegistryKind
    item_id: str | None = None
    errors: list[str] = []


class SkillEnabledPatchRequest(StrictBaseModel):
    enabled: bool


class TemplateContentRequest(StrictBaseModel):
    content: str
    sample_prompt: str | None = None
    mode: WorkflowMode | None = None


class TemplateDuplicateRequest(StrictBaseModel):
    new_template_id: str | None = None
    new_name: str | None = None


class TemplateValidationResponse(StrictBaseModel):
    valid: bool
    template_id: str | None = None
    name: str | None = None
    errors: list[str] = []


class TemplatePreviewResponse(StrictBaseModel):
    valid: bool
    template_id: str | None = None
    questionnaire: dict | None = None
    sample_prompt: str | None = None
    mode: WorkflowMode | None = None
    prompt_structure_preview: dict | None = None
    errors: list[str] = []
