from typing import Literal

from app.schemas.base import StrictBaseModel

CodexReasoningEffort = Literal["low", "medium", "high", "xhigh"]
CodexReasoningSummary = Literal["auto", "concise", "detailed", "none"]
CodexVerbosity = Literal["low", "medium", "high"]


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
