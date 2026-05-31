from typing import Literal

from app.schemas.base import StrictBaseModel


class HealthResponse(StrictBaseModel):
    status: Literal["ok"]
    app_name: str
    version: str
    environment: str
