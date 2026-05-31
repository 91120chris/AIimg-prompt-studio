import pytest
from pydantic import TypeAdapter, ValidationError

from app.schemas.agent import AgentTurnResponse


def test_agent_turn_response_questionnaire_variant_validates() -> None:
    adapter = TypeAdapter(AgentTurnResponse)

    payload = {
        "kind": "questionnaire",
        "message": "請補充需求。",
        "questionnaire": {
            "questionnaire_id": "q_001",
            "title": "影像需求",
            "questions": [
                {
                    "kind": "text",
                    "question_id": "style",
                    "label": "風格",
                    "prompt": "想要什麼風格？",
                    "required": True,
                }
            ],
        },
    }

    parsed = adapter.validate_python(payload)

    assert parsed.kind == "questionnaire"
    assert parsed.questionnaire.questions[0].kind == "text"


def test_agent_turn_response_rejects_unknown_fields() -> None:
    adapter = TypeAdapter(AgentTurnResponse)

    with pytest.raises(ValidationError):
        adapter.validate_python(
            {
                "kind": "message",
                "message": "hello",
                "unexpected": "nope",
            }
        )


def test_agent_turn_response_requires_known_discriminator() -> None:
    adapter = TypeAdapter(AgentTurnResponse)

    with pytest.raises(ValidationError):
        adapter.validate_python({"kind": "freeform", "message": "nope"})
