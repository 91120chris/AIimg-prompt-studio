from fastapi import APIRouter, HTTPException, Request

from app.core.app_settings_store import persist_app_settings
from app.providers.codex.codex_capabilities import get_codex_status
from app.providers.local_flux.client import LocalFluxClient, LocalFluxClientError
from app.providers.local_flux.workflow import validate_workflow_for_settings
from app.providers.ollama.ollama_client import get_ollama_models, get_ollama_status
from app.schemas.base import StrictBaseModel
from app.schemas.provider import (
    CodexModelsResponse,
    CodexReasoningEffort,
    CodexReasoningSummary,
    CodexStatusResponse,
    CodexVerbosity,
    LocalFluxSettingsPatch,
    LocalFluxSettingsResponse,
    LocalFluxStatusResponse,
    LocalFluxWorkflowValidationRequest,
    LocalFluxWorkflowValidationResponse,
    OllamaModelsResponse,
    OllamaStatusResponse,
)
from app.settings import (
    CODEX_MODEL_OPTION_SET,
    DEFAULT_CODEX_MODEL_OPTIONS,
    Settings,
    normalize_local_flux_model_value,
)

router = APIRouter()


class CodexModelOptionsPatch(StrictBaseModel):
    model_options: list[str]


class CodexDefaultModelPatch(StrictBaseModel):
    default_model: str


class CodexRuntimeOptionsPatch(StrictBaseModel):
    default_model: str | None = None
    default_reasoning_effort: CodexReasoningEffort | None = None
    default_reasoning_summary: CodexReasoningSummary | None = None
    default_verbosity: CodexVerbosity | None = None


class OllamaDefaultModelPatch(StrictBaseModel):
    default_model: str | None


def get_request_settings(request: Request) -> Settings:
    return request.app.state.settings


def persist_request_settings(request: Request, values: dict[str, object]) -> None:
    persist_app_settings(request.app.state.engine, values)


def normalize_model_options(options: list[str]) -> list[str]:
    seen = set()
    had_nonblank_option = False
    for option in options:
        value = option.strip()
        if not value:
            continue
        had_nonblank_option = True
        if value in seen:
            continue
        if value not in CODEX_MODEL_OPTION_SET:
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "unknown_codex_model_option",
                    "message": "Codex model options must use the bundled Codex CLI model catalog.",
                },
            )
        seen.add(value)
    if not had_nonblank_option:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "empty_codex_model_options",
                "message": "Codex model options must include at least one model label.",
            },
        )
    return DEFAULT_CODEX_MODEL_OPTIONS.copy()


def normalize_reasoning_effort_options(options: list[str]) -> list[CodexReasoningEffort]:
    allowed = {"low", "medium", "high", "xhigh"}
    normalized = []
    seen = set()
    for option in options:
        value = option.strip()
        if not value or value in seen:
            continue
        if value not in allowed:
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "unknown_codex_reasoning_effort",
                    "message": "Codex reasoning effort must be low, medium, high, or xhigh.",
                },
            )
        normalized.append(value)
        seen.add(value)
    if not normalized:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "empty_codex_reasoning_effort_options",
                "message": "Codex reasoning effort options must include at least one value.",
            },
        )
    return normalized


def codex_models_response(settings: Settings) -> CodexModelsResponse:
    return CodexModelsResponse(
        default_model=settings.codex_default_model,
        model_options=settings.codex_model_options,
        default_reasoning_effort=settings.codex_default_reasoning_effort,
        reasoning_effort_options=normalize_reasoning_effort_options(
            settings.codex_reasoning_effort_options
        ),
        default_reasoning_summary=settings.codex_default_reasoning_summary,
        default_verbosity=settings.codex_default_verbosity,
    )


def select_ollama_model(settings: Settings, models: list[str]) -> str | None:
    if settings.ollama_selected_model in models:
        return settings.ollama_selected_model
    return models[0] if models else None


def local_flux_settings_response(settings: Settings) -> LocalFluxSettingsResponse:
    return LocalFluxSettingsResponse(
        provider="local_flux",
        base_url=settings.local_flux_base_url,
        workflow_path=settings.local_flux_workflow_path,
        i2i_one_workflow_path=settings.local_flux_i2i_one_workflow_path,
        i2i_two_workflow_path=settings.local_flux_i2i_two_workflow_path,
        model_path=settings.local_flux_model_path,
        vae_path=settings.local_flux_vae_path,
        text_encoder_path=settings.local_flux_text_encoder_path,
        width=settings.local_flux_width,
        height=settings.local_flux_height,
        seed=settings.local_flux_seed,
        steps=settings.local_flux_steps,
        cfg=settings.local_flux_cfg,
        sampler_name=settings.local_flux_sampler_name,
        scheduler=settings.local_flux_scheduler,
        denoise=settings.local_flux_denoise,
        guidance=settings.local_flux_guidance,
        output_prefix=settings.local_flux_output_prefix,
        timeout_seconds=settings.local_flux_timeout_seconds,
    )


def _nonblank(value: str | None, field_name: str) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        raise HTTPException(
            status_code=422,
            detail={
                "code": f"empty_{field_name}",
                "message": f"{field_name} cannot be empty.",
            },
        )
    return stripped


def _int_range(value: int | None, field_name: str, minimum: int, maximum: int) -> int | None:
    if value is None:
        return None
    if minimum <= value <= maximum:
        return value
    raise HTTPException(
        status_code=422,
        detail={
            "code": f"invalid_{field_name}",
            "message": f"{field_name} must be between {minimum} and {maximum}.",
        },
    )


def _float_range(value: float | None, field_name: str, minimum: float, maximum: float) -> float | None:
    if value is None:
        return None
    if minimum <= value <= maximum:
        return value
    raise HTTPException(
        status_code=422,
        detail={
            "code": f"invalid_{field_name}",
            "message": f"{field_name} must be between {minimum} and {maximum}.",
        },
    )


@router.get("/providers/codex/status", response_model=CodexStatusResponse)
def codex_status(request: Request) -> CodexStatusResponse:
    return get_codex_status(get_request_settings(request))


@router.get("/providers/codex/models", response_model=CodexModelsResponse)
def codex_models(request: Request) -> CodexModelsResponse:
    settings = get_request_settings(request)
    return codex_models_response(settings)


@router.patch("/providers/codex/model-options", response_model=CodexModelsResponse)
def patch_codex_model_options(
    payload: CodexModelOptionsPatch, request: Request
) -> CodexModelsResponse:
    settings = get_request_settings(request)
    options = normalize_model_options(payload.model_options)
    settings.codex_model_options = options
    if settings.codex_default_model not in options:
        settings.codex_default_model = options[0]
    persist_request_settings(
        request,
        {
            "codex_model_options": settings.codex_model_options,
            "codex_default_model": settings.codex_default_model,
        },
    )
    return codex_models_response(settings)


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
    persist_request_settings(request, {"codex_default_model": settings.codex_default_model})
    return codex_models_response(settings)


@router.patch("/providers/codex/runtime-options", response_model=CodexModelsResponse)
def patch_codex_runtime_options(
    payload: CodexRuntimeOptionsPatch, request: Request
) -> CodexModelsResponse:
    settings = get_request_settings(request)
    fields_set = payload.model_fields_set
    if "default_model" in fields_set and payload.default_model is not None:
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
    if (
        "default_reasoning_effort" in fields_set
        and payload.default_reasoning_effort is not None
    ):
        settings.codex_default_reasoning_effort = payload.default_reasoning_effort
    if (
        "default_reasoning_summary" in fields_set
        and payload.default_reasoning_summary is not None
    ):
        settings.codex_default_reasoning_summary = payload.default_reasoning_summary
    if "default_verbosity" in fields_set:
        settings.codex_default_verbosity = payload.default_verbosity
    persist_request_settings(
        request,
        {
            "codex_default_model": settings.codex_default_model,
            "codex_default_reasoning_effort": settings.codex_default_reasoning_effort,
            "codex_default_reasoning_summary": settings.codex_default_reasoning_summary,
            "codex_default_verbosity": settings.codex_default_verbosity,
        },
    )
    return codex_models_response(settings)


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
    persist_request_settings(request, {"ollama_selected_model": settings.ollama_selected_model})
    return OllamaModelsResponse(selected_model=select_ollama_model(settings, models), models=models)


@router.get("/providers/local-flux/status", response_model=LocalFluxStatusResponse)
def local_flux_status(request: Request) -> LocalFluxStatusResponse:
    settings = get_request_settings(request)
    try:
        LocalFluxClient(settings).get_system_stats()
    except LocalFluxClientError as error:
        return LocalFluxStatusResponse(
            provider="local_flux",
            available=False,
            base_url=settings.local_flux_base_url,
            message="Local Flux 未連線",
            error=str(error),
        )
    return LocalFluxStatusResponse(
        provider="local_flux",
        available=True,
        base_url=settings.local_flux_base_url,
        message="Local Flux 已連線",
        error=None,
    )


@router.get("/providers/local-flux/settings", response_model=LocalFluxSettingsResponse)
def local_flux_settings(request: Request) -> LocalFluxSettingsResponse:
    return local_flux_settings_response(get_request_settings(request))


@router.patch("/providers/local-flux/settings", response_model=LocalFluxSettingsResponse)
def patch_local_flux_settings(
    payload: LocalFluxSettingsPatch,
    request: Request,
) -> LocalFluxSettingsResponse:
    settings = get_request_settings(request)
    updates: dict[str, object] = {}
    fields_set = payload.model_fields_set
    string_fields = {
        "base_url": "local_flux_base_url",
        "workflow_path": "local_flux_workflow_path",
        "i2i_one_workflow_path": "local_flux_i2i_one_workflow_path",
        "i2i_two_workflow_path": "local_flux_i2i_two_workflow_path",
        "model_path": "local_flux_model_path",
        "vae_path": "local_flux_vae_path",
        "text_encoder_path": "local_flux_text_encoder_path",
        "sampler_name": "local_flux_sampler_name",
        "scheduler": "local_flux_scheduler",
        "output_prefix": "local_flux_output_prefix",
    }
    for payload_field, settings_field in string_fields.items():
        if payload_field not in fields_set:
            continue
        value = _nonblank(getattr(payload, payload_field), settings_field)
        if value is not None:
            value = normalize_local_flux_model_value(value)
            setattr(settings, settings_field, value)
            updates[settings_field] = value

    int_fields = {
        "width": ("local_flux_width", 64, 4096),
        "height": ("local_flux_height", 64, 4096),
        "steps": ("local_flux_steps", 1, 150),
        "timeout_seconds": ("local_flux_timeout_seconds", 1, 7200),
    }
    for payload_field, (settings_field, minimum, maximum) in int_fields.items():
        if payload_field not in fields_set:
            continue
        value = _int_range(getattr(payload, payload_field), settings_field, minimum, maximum)
        if value is not None:
            setattr(settings, settings_field, value)
            updates[settings_field] = value

    if "seed" in fields_set:
        settings.local_flux_seed = payload.seed
        updates["local_flux_seed"] = payload.seed

    float_fields = {
        "cfg": ("local_flux_cfg", 0, 30),
        "denoise": ("local_flux_denoise", 0, 1),
        "guidance": ("local_flux_guidance", 0, 30),
    }
    for payload_field, (settings_field, minimum, maximum) in float_fields.items():
        if payload_field not in fields_set:
            continue
        value = _float_range(getattr(payload, payload_field), settings_field, minimum, maximum)
        if value is not None:
            setattr(settings, settings_field, value)
            updates[settings_field] = value

    if updates:
        persist_request_settings(request, updates)
    return local_flux_settings_response(settings)


@router.post(
    "/providers/local-flux/workflows/validate",
    response_model=LocalFluxWorkflowValidationResponse,
)
def validate_local_flux_workflow(
    payload: LocalFluxWorkflowValidationRequest,
    request: Request,
) -> LocalFluxWorkflowValidationResponse:
    valid, path, workflow_format, missing, message = validate_workflow_for_settings(
        get_request_settings(request),
        workflow_path=payload.workflow_path,
        mode=payload.mode,
        reference_count=payload.reference_count,
    )
    return LocalFluxWorkflowValidationResponse(
        valid=valid,
        workflow_path=str(path),
        workflow_format=workflow_format,
        missing_bindings=missing,
        message=message,
    )
