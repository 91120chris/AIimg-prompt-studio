from typing import Literal

from app.schemas.base import StrictBaseModel
from app.schemas.errors import StructuredError

WorkflowMode = Literal["t2i", "i2i"]
ImageProvider = Literal["codex_cli_gpt_image", "diffusers_flux2"]
GenerationJobStatus = Literal["queued", "running", "succeeded", "failed", "cancelled"]


class GenerationParameters(StrictBaseModel):
    steps: int
    guidance: float
    seed: int | None = None


class GenerationConfirmRequest(StrictBaseModel):
    session_id: str
    provider: ImageProvider = "codex_cli_gpt_image"
    mode: WorkflowMode = "t2i"
    original_prompt: str
    optimized_prompt: str
    parameters: GenerationParameters
    reference_image_ids: list[str] = []


class GenerationCancelRequest(StrictBaseModel):
    job_id: str


class CodexImageResponse(StrictBaseModel):
    status: Literal["succeeded", "failed"]
    image_files: list[str]
    error: StructuredError | None = None


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


class GenerationJobResponse(StrictBaseModel):
    job_id: str
    session_id: str
    provider: str
    mode: str
    status: GenerationJobStatus
    images: list[GenerationImage]
    error: StructuredError | None = None
    created_at: str
