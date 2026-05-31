from typing import Literal

from app.schemas.base import StrictBaseModel
from app.schemas.errors import StructuredError


class GenerationImage(StrictBaseModel):
    image_id: str
    session_id: str
    role: str
    url: str
    thumbnail_url: str | None = None
    filename: str
    width: int
    height: int
    seed: int | None = None
    provider: str
    created_at: str


class GenerationResult(StrictBaseModel):
    status: Literal["succeeded", "failed", "cancelled"]
    session_id: str
    job_id: str
    provider: str
    images: list[GenerationImage]
    error: StructuredError | None = None
