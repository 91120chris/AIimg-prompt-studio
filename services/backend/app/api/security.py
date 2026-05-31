from fastapi import APIRouter, Request

from app.schemas.base import StrictBaseModel
from app.settings import Settings

router = APIRouter()


class SecretStatusResponse(StrictBaseModel):
    hf_token_configured: bool
    hf_home_configured: bool
    hf_hub_cache_configured: bool


def get_request_settings(request: Request) -> Settings:
    return request.app.state.settings


@router.get("/security/secrets/status", response_model=SecretStatusResponse)
def secret_status(request: Request) -> SecretStatusResponse:
    settings = get_request_settings(request)
    return SecretStatusResponse(
        hf_token_configured=settings.hf_token is not None,
        hf_home_configured=settings.hf_home is not None,
        hf_hub_cache_configured=settings.hf_hub_cache is not None,
    )
