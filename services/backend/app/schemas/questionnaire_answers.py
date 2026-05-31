from typing import Annotated, Literal

from pydantic import Field

from app.schemas.base import StrictBaseModel


class AnswerBase(StrictBaseModel):
    question_id: str


class TextQuestionnaireAnswer(AnswerBase):
    kind: Literal["text"]
    value: str


class ChoiceQuestionnaireAnswer(AnswerBase):
    kind: Literal["choice"]
    value: str


class MultiChoiceQuestionnaireAnswer(AnswerBase):
    kind: Literal["multi_choice"]
    values: list[str] = Field(min_length=1)


class BooleanQuestionnaireAnswer(AnswerBase):
    kind: Literal["boolean"]
    value: bool


class ScaleQuestionnaireAnswer(AnswerBase):
    kind: Literal["scale"]
    value: int


QuestionnaireAnswer = Annotated[
    TextQuestionnaireAnswer
    | ChoiceQuestionnaireAnswer
    | MultiChoiceQuestionnaireAnswer
    | BooleanQuestionnaireAnswer
    | ScaleQuestionnaireAnswer,
    Field(discriminator="kind"),
]


class QuestionnaireAnswerPayload(StrictBaseModel):
    session_id: str
    questionnaire_id: str
    answers: list[QuestionnaireAnswer] = Field(min_length=1)
