from app.schemas.base import StrictBaseModel


class ModelInfoResponse(StrictBaseModel):
    provider: str
    label: str
    status: str
    installed: bool
    path_configured: bool
    path_label: str | None = None
    message: str | None = None
