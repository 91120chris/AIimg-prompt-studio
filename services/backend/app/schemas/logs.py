from app.schemas.base import StrictBaseModel


class LogResponse(StrictBaseModel):
    log_id: str
    level: str
    message: str
    created_at: str
