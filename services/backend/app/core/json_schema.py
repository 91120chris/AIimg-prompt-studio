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
        schema = adapter.json_schema(ref_template="#/$defs/{model}")
        path.write_text(
            json.dumps(schema, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        written.append(path)
    return written


def iter_object_schemas(schema: object) -> list[dict[str, object]]:
    found: list[dict[str, object]] = []

    def walk(value: object) -> None:
        if isinstance(value, dict):
            if value.get("type") == "object" or "properties" in value:
                found.append(value)
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(schema)
    return found
