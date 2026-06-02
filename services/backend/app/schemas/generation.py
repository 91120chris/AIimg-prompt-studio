from typing import Literal

from pydantic import Field

from app.schemas.base import StrictBaseModel
from app.schemas.errors import StructuredError
from app.schemas.provider import CodexReasoningEffort, CodexReasoningSummary, CodexVerbosity

WorkflowMode = Literal["t2i", "i2i"]
ImageProvider = Literal["codex_cli_gpt_image", "diffusers_flux2"]
GenerationJobStatus = Literal["queued", "running", "succeeded", "failed", "cancelled"]


class GenerationParameters(StrictBaseModel):
    steps: int = Field(ge=1, le=120)
    guidance: float = Field(ge=0, le=50)
    seed: int | None = None
    width: int = Field(default=1024, ge=256, le=2048)
    height: int = Field(default=1024, ge=256, le=2048)


class GenerationConfirmRequest(StrictBaseModel):
    session_id: str
    provider: ImageProvider = "codex_cli_gpt_image"
    mode: WorkflowMode = "t2i"
    original_prompt: str
    optimized_prompt: str
    parameters: GenerationParameters
    reference_image_ids: list[str] = []
    codex_model: str | None = None
    codex_reasoning_effort: CodexReasoningEffort | None = None
    codex_reasoning_summary: CodexReasoningSummary | None = None
    codex_verbosity: CodexVerbosity | None = None


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
