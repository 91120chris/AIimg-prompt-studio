from pathlib import Path

from sqlalchemy import Engine
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


def new_session(engine: Engine) -> Session:
    return Session(engine)
