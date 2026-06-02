from app.schemas.base import StrictBaseModel
from app.schemas.files import GeneratedImageResponse, ReferenceImageResponse


class SessionCreateRequest(StrictBaseModel):
    title: str | None = None


class SessionResponse(StrictBaseModel):
    session_id: str
    title: str | None
    created_at: str
    reference_images: list[ReferenceImageResponse] = []
    generated_images: list[GeneratedImageResponse] = []


class PromptVersionResponse(StrictBaseModel):
    prompt_version_id: str
    session_id: str
    prompt_text: str
    title: str | None
    source: str
    metadata: dict
    is_current: bool
    created_at: str


class CurrentPromptVersionPatchRequest(StrictBaseModel):
    prompt_version_id: str
