from fastapi import APIRouter, HTTPException, Request

from app.providers.codex.codex_capabilities import get_codex_status
from app.providers.ollama.ollama_client import get_ollama_models, get_ollama_status
from app.schemas.base import StrictBaseModel
from app.schemas.provider import (
    CodexModelsResponse,
    CodexStatusResponse,
    OllamaModelsResponse,
    OllamaStatusResponse,
)
from app.settings import Settings

router = APIRouter()


class CodexModelOptionsPatch(StrictBaseModel):
    model_options: list[str]


class CodexDefaultModelPatch(StrictBaseModel):
    default_model: str


class OllamaDefaultModelPatch(StrictBaseModel):
    default_model: str | None


def get_request_settings(request: Request) -> Settings:
    return request.app.state.settings


def normalize_model_options(options: list[str]) -> list[str]:
    normalized = []
    seen = set()
    for option in options:
        value = option.strip()
        if not value or value in seen:
            continue
        normalized.append(value)
        seen.add(value)
    if not normalized:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "empty_codex_model_options",
                "message": "Codex model options must include at least one model label.",
            },
        )
    return normalized


def select_ollama_model(settings: Settings, models: list[str]) -> str | None:
    if settings.ollama_selected_model in models:
        return settings.ollama_selected_model
    return models[0] if models else None


@router.get("/providers/codex/status", response_model=CodexStatusResponse)
def codex_status(request: Request) -> CodexStatusResponse:
    return get_codex_status(get_request_settings(request))


@router.get("/providers/codex/models", response_model=CodexModelsResponse)
def codex_models(request: Request) -> CodexModelsResponse:
    settings = get_request_settings(request)
    return CodexModelsResponse(
        default_model=settings.codex_default_model,
        model_options=settings.codex_model_options,
    )


@router.patch("/providers/codex/model-options", response_model=CodexModelsResponse)
def patch_codex_model_options(
    payload: CodexModelOptionsPatch, request: Request
) -> CodexModelsResponse:
    settings = get_request_settings(request)
    options = normalize_model_options(payload.model_options)
    settings.codex_model_options = options
    if settings.codex_default_model not in options:
        settings.codex_default_model = options[0]
    return CodexModelsResponse(
        default_model=settings.codex_default_model,
        model_options=settings.codex_model_options,
    )


@router.patch("/providers/codex/default-model", response_model=CodexModelsResponse)
def patch_codex_default_model(
    payload: CodexDefaultModelPatch, request: Request
) -> CodexModelsResponse:
    settings = get_request_settings(request)
    default_model = payload.default_model.strip()
    if not default_model:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "empty_codex_default_model",
                "message": "Codex default model cannot be empty.",
            },
        )
    if default_model not in settings.codex_model_options:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "unknown_codex_model",
                "message": "Codex default model must be one of the configured options.",
            },
        )
    settings.codex_default_model = default_model
    return CodexModelsResponse(
        default_model=settings.codex_default_model,
        model_options=settings.codex_model_options,
    )


@router.get("/providers/ollama/status", response_model=OllamaStatusResponse)
def ollama_status(request: Request) -> OllamaStatusResponse:
    return get_ollama_status(get_request_settings(request))


@router.get("/providers/ollama/models", response_model=OllamaModelsResponse)
def ollama_models(request: Request) -> OllamaModelsResponse:
    settings = get_request_settings(request)
    models = get_ollama_models(settings)
    return OllamaModelsResponse(selected_model=select_ollama_model(settings, models), models=models)


@router.patch("/providers/ollama/default-model", response_model=OllamaModelsResponse)
def patch_ollama_default_model(
    payload: OllamaDefaultModelPatch, request: Request
) -> OllamaModelsResponse:
    settings = get_request_settings(request)
    models = get_ollama_models(settings)
    selected_model = payload.default_model.strip() if payload.default_model else None

    if selected_model is not None and selected_model not in models:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "unknown_ollama_model",
                "message": "Ollama selected model must be one of the currently installed models.",
            },
        )

    settings.ollama_selected_model = selected_model
    return OllamaModelsResponse(selected_model=select_ollama_model(settings, models), models=models)
