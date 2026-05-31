import pytest
from pydantic import TypeAdapter, ValidationError

from app.schemas.questionnaire import Question, Questionnaire


def test_choice_question_requires_options() -> None:
    adapter = TypeAdapter(Question)

    with pytest.raises(ValidationError):
        adapter.validate_python(
            {
                "kind": "choice",
                "question_id": "palette",
                "label": "配色",
                "prompt": "選擇配色",
                "options": [],
            }
        )


def test_scale_question_requires_valid_range() -> None:
    adapter = TypeAdapter(Question)

    with pytest.raises(ValidationError):
        adapter.validate_python(
            {
                "kind": "scale",
                "question_id": "strength",
                "label": "強度",
                "prompt": "強度",
                "min_value": 5,
                "max_value": 5,
            }
        )


def test_questionnaire_rejects_extra_properties() -> None:
    with pytest.raises(ValidationError):
        Questionnaire.model_validate(
            {
                "questionnaire_id": "q_001",
                "title": "問卷",
                "questions": [
                    {
                        "kind": "boolean",
                        "question_id": "needs_text",
                        "label": "文字",
                        "prompt": "是否包含文字？",
                    }
                ],
                "extra": "forbidden",
            }
        )
