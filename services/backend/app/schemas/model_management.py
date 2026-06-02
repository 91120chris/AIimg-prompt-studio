from typing import Literal

from app.schemas.base import StrictBaseModel

FluxProvider = Literal["diffusers_flux2_klein_9b_fp8"]
FluxStatus = Literal["not_installed", "path_selected", "install_pending", "unloaded"]


class ModelInfoResponse(StrictBaseModel):
    provider: str
    label: str
    status: str
    installed: bool
    path_configured: bool
    path_label: str | None = None
    message: str | None = None


class FluxStatusResponse(StrictBaseModel):
    provider: FluxProvider
    label: str
    status: FluxStatus
    installed: bool
    path_configured: bool
    path_label: str | None = None
    message: str


class FluxReadinessResponse(StrictBaseModel):
    provider: FluxProvider
    label: str
    hf_token_configured: bool
    hf_cache_configured: bool
    path_configured: bool
    path_label: str | None = None
    can_queue_install: bool
    message: str


class FluxPathRequest(StrictBaseModel):
    model_path: str
