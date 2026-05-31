from typing import Annotated, Literal

from pydantic import Field, model_validator

from app.schemas.base import StrictBaseModel


class QuestionOption(StrictBaseModel):
    value: str
    label: str
    description: str | None = None


class QuestionBase(StrictBaseModel):
    question_id: str
    label: str
    prompt: str
    required: bool = True


class TextQuestion(QuestionBase):
    kind: Literal["text"]
    placeholder: str | None = None
    max_length: int | None = Field(default=None, ge=1)


class ChoiceQuestion(QuestionBase):
    kind: Literal["choice"]
    options: list[QuestionOption] = Field(min_length=1)
    allow_multiple: bool = False


class BooleanQuestion(QuestionBase):
    kind: Literal["boolean"]
    true_label: str = "是"
    false_label: str = "否"


class ScaleQuestion(QuestionBase):
    kind: Literal["scale"]
    min_value: int
    max_value: int
    step: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def validate_range(self) -> "ScaleQuestion":
        if self.max_value <= self.min_value:
            raise ValueError("max_value must be greater than min_value")
        return self


Question = Annotated[
    TextQuestion | ChoiceQuestion | BooleanQuestion | ScaleQuestion,
    Field(discriminator="kind"),
]


class Questionnaire(StrictBaseModel):
    questionnaire_id: str
    title: str
    description: str | None = None
    questions: list[Question] = Field(min_length=1)
