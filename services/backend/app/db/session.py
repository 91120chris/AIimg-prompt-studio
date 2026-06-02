from pathlib import Path

from sqlalchemy import Engine, inspect, text
from sqlmodel import SQLModel, Session, create_engine

from app.db import models as _models  # noqa: F401
from app.settings import Settings


def database_url_for_settings(settings: Settings) -> str:
    if settings.database_url:
        return settings.database_url
    storage_root = Path(settings.storage_root)
    storage_root.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{storage_root / 'app.sqlite3'}"


def create_db_engine(settings: Settings) -> Engine:
    return create_engine(
        database_url_for_settings(settings),
        connect_args={"check_same_thread": False},
    )


def init_db(engine: Engine) -> None:
    SQLModel.metadata.create_all(engine)
    _ensure_prompt_version_columns(engine)
    _ensure_registry_patch_columns(engine)


def new_session(engine: Engine) -> Session:
    return Session(engine)


def _ensure_registry_patch_columns(engine: Engine) -> None:
    inspector = inspect(engine)
    table_names = set(inspector.get_table_names())
    if "registry_patch_proposals" not in table_names:
        return

    existing_columns = {
        column["name"] for column in inspector.get_columns("registry_patch_proposals")
    }
    columns_to_add = {
        "change_kind": "TEXT NOT NULL DEFAULT 'update'",
        "target_id": "TEXT",
        "summary": "TEXT",
        "proposed_content": "TEXT",
        "validation_json": "TEXT",
        "source_json": "TEXT",
        "applied_version_id": "TEXT",
    }
    with engine.begin() as connection:
        for column_name, column_type in columns_to_add.items():
            if column_name not in existing_columns:
                connection.execute(
                    text(
                        "ALTER TABLE registry_patch_proposals "
                        f"ADD COLUMN {column_name} {column_type}"
                    )
                )


def _ensure_prompt_version_columns(engine: Engine) -> None:
    inspector = inspect(engine)
    table_names = set(inspector.get_table_names())
    if "prompt_versions" not in table_names:
        return

    existing_columns = {
        column["name"] for column in inspector.get_columns("prompt_versions")
    }
    columns_to_add = {
        "title": "TEXT",
        "source": "TEXT NOT NULL DEFAULT 'optimized_prompt'",
        "metadata_json": "TEXT NOT NULL DEFAULT '{}'",
    }
    with engine.begin() as connection:
        for column_name, column_type in columns_to_add.items():
            if column_name not in existing_columns:
                connection.execute(
                    text(
                        "ALTER TABLE prompt_versions "
                        f"ADD COLUMN {column_name} {column_type}"
                    )
                )
