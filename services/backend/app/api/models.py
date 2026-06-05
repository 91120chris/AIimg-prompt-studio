from pathlib import Path

from fastapi import APIRouter, Request

from app.providers.local_flux.client import LocalFluxClient, LocalFluxClientError
from app.schemas.model_management import ModelInfoResponse
from app.settings import Settings

router = APIRouter(prefix="/models", tags=["models"])


def _settings(request: Request) -> Settings:
    return request.app.state.settings


def _path_label(value: str) -> str:
    return Path(value).name or "configured"


@router.get("", response_model=list[ModelInfoResponse])
def list_models(request: Request) -> list[ModelInfoResponse]:
    settings = _settings(request)
    try:
        LocalFluxClient(settings).get_system_stats()
        available = True
        status = "available"
        message = "Local Flux backend is reachable."
    except LocalFluxClientError as error:
        available = False
        status = "offline"
        message = f"Local Flux backend is not reachable: {error}"

    return [
        ModelInfoResponse(
            provider="local_flux",
            label="Local Flux",
            status=status,
            installed=available,
            path_configured=bool(settings.local_flux_model_path),
            path_label=_path_label(settings.local_flux_model_path),
            message=message,
        )
    ]


@router.get("/loras", response_model=list[str])
def list_loras(request: Request) -> list[str]:
    settings = _settings(request)
    try:
        return LocalFluxClient(settings).get_models("loras")
    except LocalFluxClientError:
        return []
