from app.schemas.base import StrictBaseModel


class ReferenceImageResponse(StrictBaseModel):
    reference_image_id: str
    session_id: str
    slot: int
    role: str
    url: str
    thumbnail_url: str | None
    filename: str
    width: int
    height: int
    created_at: str


class GeneratedImageResponse(StrictBaseModel):
    image_id: str
    session_id: str
    role: str
    url: str
    thumbnail_url: str | None
    filename: str
    width: int
    height: int
    seed: int | None
    provider: str
    created_at: str
