from typing import Literal

from app.schemas.base import StrictBaseModel

CodexReasoningEffort = Literal["low", "medium", "high", "xhigh"]
CodexReasoningSummary = Literal["auto", "concise", "detailed", "none"]
CodexVerbosity = Literal["low", "medium", "high"]
LocalFluxWorkflowFormat = Literal["api", "ui", "unknown"]


class CodexStatusResponse(StrictBaseModel):
    provider: Literal["codex_cli"]
    available: bool
    configured_binary: str
    resolved_kind: Literal["native", "cmd", "ps1", "not_found", "unknown"]
    version: str | None = None
    warning: str | None = None
    error: str | None = None


class CodexModelsResponse(StrictBaseModel):
    default_model: str
    model_options: list[str]
    default_reasoning_effort: CodexReasoningEffort
    reasoning_effort_options: list[CodexReasoningEffort]
    default_reasoning_summary: CodexReasoningSummary
    default_verbosity: CodexVerbosity | None = None


class OllamaStatusResponse(StrictBaseModel):
    provider: Literal["ollama_local_llm"]
    available: bool
    base_url: str
    model_count: int
    error: str | None = None


class OllamaModelsResponse(StrictBaseModel):
    selected_model: str | None
    models: list[str]


class LocalFluxStatusResponse(StrictBaseModel):
    provider: Literal["local_flux"]
    available: bool
    base_url: str
    message: str
    error: str | None = None


class LocalFluxSettingsResponse(StrictBaseModel):
    provider: Literal["local_flux"]
    base_url: str
    workflow_path: str
    i2i_one_workflow_path: str
    i2i_two_workflow_path: str
    model_path: str
    vae_path: str
    text_encoder_path: str
    width: int
    height: int
    seed: int | None
    steps: int
    cfg: float
    sampler_name: str
    scheduler: str
    denoise: float
    guidance: float
    output_prefix: str
    timeout_seconds: int
    lora_dir: str


class LocalFluxSettingsPatch(StrictBaseModel):
    base_url: str | None = None
    workflow_path: str | None = None
    i2i_one_workflow_path: str | None = None
    i2i_two_workflow_path: str | None = None
    model_path: str | None = None
    vae_path: str | None = None
    text_encoder_path: str | None = None
    width: int | None = None
    height: int | None = None
    seed: int | None = None
    steps: int | None = None
    cfg: float | None = None
    sampler_name: str | None = None
    scheduler: str | None = None
    denoise: float | None = None
    guidance: float | None = None
    output_prefix: str | None = None
    timeout_seconds: int | None = None
    lora_dir: str | None = None


class LocalFluxWorkflowValidationRequest(StrictBaseModel):
    workflow_path: str | None = None
    mode: Literal["t2i", "i2i"] = "t2i"
    reference_count: int = 0


class LocalFluxWorkflowValidationResponse(StrictBaseModel):
    valid: bool
    workflow_path: str
    workflow_format: LocalFluxWorkflowFormat
    missing_bindings: list[str]
    message: str
