from fastapi import APIRouter, Request

from app.schemas.settings import SafeSettingsResponse
from app.settings import Settings

router = APIRouter()


def get_request_settings(request: Request) -> Settings:
    return request.app.state.settings


@router.get("/settings/safe", response_model=SafeSettingsResponse)
def safe_settings(request: Request) -> SafeSettingsResponse:
    return SafeSettingsResponse.from_settings(get_request_settings(request))
