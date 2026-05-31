from fastapi import APIRouter, HTTPException, Request

from app.core.app_settings_store import persist_app_settings
from app.schemas.settings import SafeSettingsPatch, SafeSettingsResponse
from app.settings import Settings, parse_csv

router = APIRouter()


def get_request_settings(request: Request) -> Settings:
    return request.app.state.settings


@router.get("/settings/safe", response_model=SafeSettingsResponse)
def safe_settings(request: Request) -> SafeSettingsResponse:
    return SafeSettingsResponse.from_settings(get_request_settings(request))


@router.patch("/settings/safe", response_model=SafeSettingsResponse)
def patch_safe_settings(payload: SafeSettingsPatch, request: Request) -> SafeSettingsResponse:
    settings = get_request_settings(request)
    fields_set = payload.model_fields_set
    updates: dict[str, object] = {}

    if "selected_agent_provider" in fields_set and payload.selected_agent_provider is not None:
        settings.selected_agent_provider = payload.selected_agent_provider
        updates["selected_agent_provider"] = payload.selected_agent_provider
    if "selected_image_provider" in fields_set and payload.selected_image_provider is not None:
        settings.selected_image_provider = payload.selected_image_provider
        updates["selected_image_provider"] = payload.selected_image_provider
    if "cors_allow_origins" in fields_set and payload.cors_allow_origins is not None:
        origins = parse_csv(payload.cors_allow_origins)
        if not origins:
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "empty_cors_allow_origins",
                    "message": "CORS allow origins must include at least one origin.",
                },
            )
        settings.cors_allow_origins = origins
        updates["cors_allow_origins"] = origins
    if "frontend_api_base_url" in fields_set and payload.frontend_api_base_url is not None:
        frontend_api_base_url = payload.frontend_api_base_url.strip()
        if not frontend_api_base_url:
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "empty_frontend_api_base_url",
                    "message": "Frontend API base URL cannot be empty.",
                },
            )
        settings.frontend_api_base_url = frontend_api_base_url
        updates["frontend_api_base_url"] = frontend_api_base_url
    if "ollama_base_url" in fields_set and payload.ollama_base_url is not None:
        ollama_base_url = payload.ollama_base_url.strip()
        if not ollama_base_url:
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "empty_ollama_base_url",
                    "message": "Ollama base URL cannot be empty.",
                },
            )
        settings.ollama_base_url = ollama_base_url
        updates["ollama_base_url"] = ollama_base_url
    if "ollama_timeout_seconds" in fields_set and payload.ollama_timeout_seconds is not None:
        if not 1 <= payload.ollama_timeout_seconds <= 3600:
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "invalid_ollama_timeout_seconds",
                    "message": "Ollama timeout seconds must be between 1 and 3600.",
                },
            )
        settings.ollama_timeout_seconds = payload.ollama_timeout_seconds
        updates["ollama_timeout_seconds"] = payload.ollama_timeout_seconds
    if (
        "ollama_agent_temperature" in fields_set
        and payload.ollama_agent_temperature is not None
    ):
        if not 0 <= payload.ollama_agent_temperature <= 2:
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "invalid_ollama_agent_temperature",
                    "message": "Ollama agent temperature must be between 0 and 2.",
                },
            )
        settings.ollama_agent_temperature = payload.ollama_agent_temperature
        updates["ollama_agent_temperature"] = payload.ollama_agent_temperature

    if updates:
        persist_app_settings(request.app.state.engine, updates)
    return SafeSettingsResponse.from_settings(settings)
