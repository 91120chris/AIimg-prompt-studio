from app.schemas.base import StrictBaseModel


class StructuredError(StrictBaseModel):
    code: str
    message: str
    suggestion: str | None = None
