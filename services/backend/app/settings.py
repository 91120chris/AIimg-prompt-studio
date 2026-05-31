from functools import lru_cache
from typing import Annotated

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


DEFAULT_CORS_ALLOW_ORIGINS = [
    "http://localhost:1420",
    "http://127.0.0.1:1420",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "tauri://localhost",
]

DEFAULT_CODEX_MODEL_OPTIONS = ["auto", "gpt-5.5", "gpt-5.4"]


def parse_csv(value: str | list[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in value.split(",") if item.strip()]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_name: str = "Prompt Optimizer Studio"
    app_version: str = "0.1.0"
    app_env: str = "development"

    backend_host: str = "127.0.0.1"
    backend_port: int = 8000
    storage_root: str = "storage"
    database_url: str | None = None

    cors_allow_origins: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: DEFAULT_CORS_ALLOW_ORIGINS.copy()
    )

    codex_binary_path: str = "codex"
    codex_default_model: str = "auto"
    codex_model_options: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: DEFAULT_CODEX_MODEL_OPTIONS.copy()
    )
    codex_timeout_seconds: int = 300
    run_codex_smoke: bool = False

    ollama_base_url: str = "http://localhost:11434"
    ollama_selected_model: str | None = None

    hf_token: str | None = Field(default=None, repr=False)
    hf_home: str | None = None
    hf_hub_cache: str | None = None

    frontend_api_base_url: str = "http://127.0.0.1:8000"

    @field_validator("cors_allow_origins", "codex_model_options", mode="before")
    @classmethod
    def parse_csv_fields(cls, value: str | list[str] | None) -> list[str]:
        return parse_csv(value)

    @field_validator(
        "database_url",
        "hf_token",
        "hf_home",
        "hf_hub_cache",
        "ollama_selected_model",
        mode="before",
    )
    @classmethod
    def empty_secretish_values_to_none(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None


@lru_cache
def get_settings() -> Settings:
    return Settings()
