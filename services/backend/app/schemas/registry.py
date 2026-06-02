from typing import Literal

from app.schemas.base import StrictBaseModel

RegistryKind = Literal["skill", "template"]
ProposalStatus = Literal["pending", "approved", "rejected"]


class RegistryItemResponse(StrictBaseModel):
    registry_kind: RegistryKind
    item_id: str
    latest_version_id: str | None = None
    content: str
    created_at: str | None = None


class RegistryPatchProposalCreateRequest(StrictBaseModel):
    diff_text: str
    item_id: str | None = None
    proposed_content: str | None = None


class RegistryPatchProposalResponse(StrictBaseModel):
    proposal_id: str
    registry_kind: RegistryKind
    item_id: str | None = None
    status: ProposalStatus
    diff_text: str
    proposed_content: str | None = None
    applied_version_id: str | None = None
    created_at: str
