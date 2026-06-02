from sqlalchemy import inspect

from app.main import create_app
from app.settings import Settings


def test_database_includes_required_tables(tmp_path) -> None:
    app = create_app(
        Settings(
            storage_root=str(tmp_path / "storage"),
            database_url=f"sqlite:///{tmp_path / 'app.sqlite3'}",
            _env_file=None,
        )
    )

    table_names = set(inspect(app.state.engine).get_table_names())

    assert {
        "sessions",
        "agent_turns",
        "questionnaires",
        "questionnaire_answers",
        "prompts",
        "prompt_versions",
        "reference_images",
        "generation_jobs",
        "generated_images",
        "skill_versions",
        "template_versions",
        "registry_patch_proposals",
        "model_status",
        "app_settings",
        "logs",
    }.issubset(table_names)
