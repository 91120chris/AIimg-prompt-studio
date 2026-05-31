import pytest
from pydantic import TypeAdapter, ValidationError

from app.schemas.questionnaire_answers import QuestionnaireAnswer, QuestionnaireAnswerPayload


def test_questionnaire_answer_discriminated_union_validates() -> None:
    adapter = TypeAdapter(QuestionnaireAnswer)

    answer = adapter.validate_python(
        {
            "kind": "multi_choice",
            "question_id": "palette",
            "values": ["warm", "cinematic"],
        }
    )

    assert answer.kind == "multi_choice"
    assert answer.values == ["warm", "cinematic"]


def test_questionnaire_answer_rejects_unknown_fields() -> None:
    adapter = TypeAdapter(QuestionnaireAnswer)

    with pytest.raises(ValidationError):
        adapter.validate_python(
            {
                "kind": "boolean",
                "question_id": "needs_text",
                "value": False,
                "raw_notes": "not allowed",
            }
        )


def test_questionnaire_answer_payload_requires_answers() -> None:
    with pytest.raises(ValidationError):
        QuestionnaireAnswerPayload.model_validate(
            {
                "session_id": "sess_001",
                "questionnaire_id": "q_001",
                "answers": [],
            }
        )
