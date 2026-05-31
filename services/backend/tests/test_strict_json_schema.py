import json
from pathlib import Path

from app.core.json_schema import generate_schema_files, iter_object_schemas


def test_generated_codex_json_schemas_are_strict() -> None:
    schema_paths = generate_schema_files()

    assert schema_paths
    for schema_path in schema_paths:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        object_schemas = iter_object_schemas(schema)
        assert object_schemas, f"{schema_path} should contain object schemas"
        for object_schema in object_schemas:
            assert object_schema.get("additionalProperties") is False, (
                f"{schema_path} has non-strict object schema: "
                f"{object_schema.get('title', '<anonymous>')}"
            )


def test_codex_schema_files_exist_in_provider_directory() -> None:
    schema_dir = Path("app/providers/codex/schemas")

    assert (schema_dir / "agent_turn_response.schema.json").exists()
    assert (schema_dir / "questionnaire_answer_payload.schema.json").exists()
    assert (schema_dir / "codex_image_response.schema.json").exists()


def test_agent_turn_schema_has_codex_compatible_object_root() -> None:
    schema_path = generate_schema_files()[0]
    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    assert schema_path.name == "agent_turn_response.schema.json"
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert "oneOf" not in schema
    assert set(schema["required"]) == set(schema["properties"])
