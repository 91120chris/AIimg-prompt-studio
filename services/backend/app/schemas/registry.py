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


class RegistryPatchProposalResponse(StrictBaseModel):
    proposal_id: str
    registry_kind: RegistryKind
    status: ProposalStatus
    diff_text: str
    created_at: str
