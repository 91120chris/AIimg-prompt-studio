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
