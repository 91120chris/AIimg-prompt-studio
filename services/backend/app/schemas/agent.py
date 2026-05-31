from typing import Annotated, Literal

from pydantic import Field

from app.schemas.base import StrictBaseModel
from app.schemas.errors import StructuredError
from app.schemas.questionnaire import Questionnaire


class MessageTurnResponse(StrictBaseModel):
    kind: Literal["message"]
    message: str
    warnings: list[str] = Field(default_factory=list)


class QuestionnaireTurnResponse(StrictBaseModel):
    kind: Literal["questionnaire"]
    message: str
    questionnaire: Questionnaire
    warnings: list[str] = Field(default_factory=list)


class OptimizedPromptTurnResponse(StrictBaseModel):
    kind: Literal["optimized_prompt"]
    message: str
    optimized_prompt: str
    prompt_version_title: str | None = None
    warnings: list[str] = Field(default_factory=list)


class ErrorTurnResponse(StrictBaseModel):
    kind: Literal["error"]
    error: StructuredError


AgentTurnResponse = Annotated[
    MessageTurnResponse
    | QuestionnaireTurnResponse
    | OptimizedPromptTurnResponse
    | ErrorTurnResponse,
    Field(discriminator="kind"),
]
