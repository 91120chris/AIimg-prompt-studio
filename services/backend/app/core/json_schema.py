import json
from pathlib import Path
from typing import TypeAlias

from pydantic import TypeAdapter

from app.schemas.agent import AgentTurnResponse
from app.schemas.generation import GenerationResult
from app.schemas.questionnaire_answers import QuestionnaireAnswerPayload

SchemaName: TypeAlias = str


SCHEMA_ADAPTERS: dict[SchemaName, TypeAdapter[object]] = {
    "agent_turn_response.schema.json": TypeAdapter(AgentTurnResponse),
    "questionnaire_answer_payload.schema.json": TypeAdapter(QuestionnaireAnswerPayload),
    "codex_image_response.schema.json": TypeAdapter(GenerationResult),
}


def schema_output_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "providers" / "codex" / "schemas"


def generate_schema_files(output_dir: Path | None = None) -> list[Path]:
    target = output_dir or schema_output_dir()
    target.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for filename, adapter in SCHEMA_ADAPTERS.items():
        path = target / filename
        if filename == "agent_turn_response.schema.json":
            schema = codex_agent_turn_response_schema()
        else:
            schema = adapter.json_schema(ref_template="#/$defs/{model}")
            require_all_declared_properties(schema)
        path.write_text(
            json.dumps(schema, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        written.append(path)
    return written


def nullable(schema: dict[str, object]) -> dict[str, object]:
    schema = dict(schema)
    schema_type = schema.get("type")
    if isinstance(schema_type, str):
        schema["type"] = [schema_type, "null"]
    return schema


def codex_agent_turn_response_schema() -> dict[str, object]:
    string_or_null = {"type": ["string", "null"]}
    int_or_null = {"type": ["integer", "null"]}
    bool_or_null = {"type": ["boolean", "null"]}
    question_option_schema: dict[str, object] = {
        "additionalProperties": False,
        "properties": {
            "description": string_or_null,
            "label": {"type": "string"},
            "value": {"type": "string"},
        },
        "required": ["description", "label", "value"],
        "type": "object",
    }
    question_schema: dict[str, object] = {
        "additionalProperties": False,
        "properties": {
            "allow_multiple": bool_or_null,
            "false_label": string_or_null,
            "kind": {"enum": ["text", "choice", "boolean", "scale"], "type": "string"},
            "label": {"type": "string"},
            "max_length": int_or_null,
            "max_value": int_or_null,
            "min_value": int_or_null,
            "options": nullable({"items": question_option_schema, "type": "array"}),
            "placeholder": string_or_null,
            "prompt": {"type": "string"},
            "question_id": {"type": "string"},
            "required": {"type": "boolean"},
            "step": int_or_null,
            "true_label": string_or_null,
        },
        "required": [
            "allow_multiple",
            "false_label",
            "kind",
            "label",
            "max_length",
            "max_value",
            "min_value",
            "options",
            "placeholder",
            "prompt",
            "question_id",
            "required",
            "step",
            "true_label",
        ],
        "type": "object",
    }
    questionnaire_schema: dict[str, object] = {
        "additionalProperties": False,
        "properties": {
            "description": string_or_null,
            "questionnaire_id": {"type": "string"},
            "questions": {"items": question_schema, "minItems": 1, "type": "array"},
            "title": {"type": "string"},
        },
        "required": ["description", "questionnaire_id", "questions", "title"],
        "type": "object",
    }
    error_schema: dict[str, object] = {
        "additionalProperties": False,
        "properties": {
            "code": {"type": "string"},
            "message": {"type": "string"},
            "suggestion": string_or_null,
        },
        "required": ["code", "message", "suggestion"],
        "type": "object",
    }
    return {
        "additionalProperties": False,
        "properties": {
            "error": nullable(error_schema),
            "kind": {"enum": ["message", "questionnaire", "optimized_prompt", "error"], "type": "string"},
            "message": string_or_null,
            "optimized_prompt": string_or_null,
            "prompt_version_title": string_or_null,
            "questionnaire": nullable(questionnaire_schema),
            "warnings": {"items": {"type": "string"}, "type": "array"},
        },
        "required": [
            "error",
            "kind",
            "message",
            "optimized_prompt",
            "prompt_version_title",
            "questionnaire",
            "warnings",
        ],
        "title": "AgentTurnResponse",
        "type": "object",
    }


def require_all_declared_properties(schema: object) -> None:
    if isinstance(schema, dict):
        properties = schema.get("properties")
        if isinstance(properties, dict) and properties:
            schema["required"] = sorted(properties)
        for child in schema.values():
            require_all_declared_properties(child)
    elif isinstance(schema, list):
        for child in schema:
            require_all_declared_properties(child)


def iter_object_schemas(schema: object) -> list[dict[str, object]]:
    found: list[dict[str, object]] = []

    def walk(value: object) -> None:
        if isinstance(value, dict):
            if (value.get("type") == "object" and "oneOf" not in value) or "properties" in value:
                found.append(value)
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(schema)
    return found
