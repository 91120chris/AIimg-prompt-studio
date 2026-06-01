import json
from collections.abc import Mapping

from sqlalchemy import Engine
from sqlmodel import select

from app.db.models import AppSettingRecord, utc_now_iso
from app.db.session import new_session
from app.settings import (
    DEFAULT_AGENT_PROVIDER_OPTIONS,
    DEFAULT_CODEX_REASONING_EFFORT_OPTIONS,
    DEFAULT_CODEX_REASONING_SUMMARY_OPTIONS,
    DEFAULT_CODEX_VERBOSITY_OPTIONS,
    DEFAULT_IMAGE_PROVIDER_OPTIONS,
    Settings,
    normalize_codex_default_model,
    normalize_codex_model_options,
    parse_csv,
)

PERSISTED_SETTING_KEYS = {
    "selected_agent_provider",
    "selected_image_provider",
    "cors_allow_origins",
    "codex_default_model",
    "codex_model_options",
    "codex_default_reasoning_effort",
    "codex_reasoning_effort_options",
    "codex_default_reasoning_summary",
    "codex_default_verbosity",
    "ollama_base_url",
    "ollama_selected_model",
    "ollama_timeout_seconds",
    "ollama_agent_temperature",
    "frontend_api_base_url",
}


def load_persisted_app_settings(engine: Engine, settings: Settings) -> None:
    with new_session(engine) as db:
        records = db.exec(select(AppSettingRecord)).all()

    values: dict[str, object] = {}
    for record in records:
        if record.key not in PERSISTED_SETTING_KEYS:
            continue
        try:
            values[record.key] = json.loads(record.value)
        except json.JSONDecodeError:
            continue

    apply_persisted_app_settings(settings, values)
    persist_normalized_codex_settings(engine, settings)


def persist_app_settings(engine: Engine, values: Mapping[str, object]) -> None:
    with new_session(engine) as db:
        for key, value in values.items():
            if key not in PERSISTED_SETTING_KEYS:
                continue
            record = db.get(AppSettingRecord, key)
            if record is None:
                record = AppSettingRecord(key=key, value="")
                db.add(record)
            record.value = json.dumps(value, ensure_ascii=False, sort_keys=True)
            record.updated_at = utc_now_iso()
        db.commit()


def persist_normalized_codex_settings(engine: Engine, settings: Settings) -> None:
    persist_app_settings(
        engine,
        {
            "codex_default_model": settings.codex_default_model,
            "codex_model_options": settings.codex_model_options,
        },
    )


def apply_persisted_app_settings(settings: Settings, values: Mapping[str, object]) -> None:
    if "selected_agent_provider" in values and values["selected_agent_provider"] in (
        DEFAULT_AGENT_PROVIDER_OPTIONS
    ):
        settings.selected_agent_provider = str(values["selected_agent_provider"])
    if "selected_image_provider" in values and values["selected_image_provider"] in (
        DEFAULT_IMAGE_PROVIDER_OPTIONS
    ):
        settings.selected_image_provider = str(values["selected_image_provider"])

    if "cors_allow_origins" in values:
        origins = parse_csv(_string_or_list(values["cors_allow_origins"]))
        if origins:
            settings.cors_allow_origins = origins

    if "codex_model_options" in values:
        options = normalize_codex_model_options(_string_or_list(values["codex_model_options"]))
        if options:
            settings.codex_model_options = options
            if settings.codex_default_model not in options:
                settings.codex_default_model = options[0]
    if "codex_default_model" in values and isinstance(values["codex_default_model"], str):
        model = normalize_codex_default_model(values["codex_default_model"])
        if model and model in settings.codex_model_options:
            settings.codex_default_model = model
    if "codex_reasoning_effort_options" in values:
        options = parse_csv(_string_or_list(values["codex_reasoning_effort_options"]))
        if options and all(option in DEFAULT_CODEX_REASONING_EFFORT_OPTIONS for option in options):
            settings.codex_reasoning_effort_options = options
    if "codex_default_reasoning_effort" in values:
        effort = values["codex_default_reasoning_effort"]
        if effort in DEFAULT_CODEX_REASONING_EFFORT_OPTIONS:
            settings.codex_default_reasoning_effort = str(effort)
    if "codex_default_reasoning_summary" in values:
        summary = values["codex_default_reasoning_summary"]
        if summary in DEFAULT_CODEX_REASONING_SUMMARY_OPTIONS:
            settings.codex_default_reasoning_summary = str(summary)
    if "codex_default_verbosity" in values:
        verbosity = values["codex_default_verbosity"]
        if verbosity is None or verbosity in DEFAULT_CODEX_VERBOSITY_OPTIONS:
            settings.codex_default_verbosity = None if verbosity is None else str(verbosity)

    if "ollama_base_url" in values and isinstance(values["ollama_base_url"], str):
        base_url = values["ollama_base_url"].strip()
        if base_url:
            settings.ollama_base_url = base_url
    if "ollama_selected_model" in values:
        model = values["ollama_selected_model"]
        settings.ollama_selected_model = model.strip() if isinstance(model, str) and model.strip() else None
    if "ollama_timeout_seconds" in values:
        timeout = _int_in_range(values["ollama_timeout_seconds"], minimum=1, maximum=3600)
        if timeout is not None:
            settings.ollama_timeout_seconds = timeout
    if "ollama_agent_temperature" in values:
        temperature = _float_in_range(values["ollama_agent_temperature"], minimum=0, maximum=2)
        if temperature is not None:
            settings.ollama_agent_temperature = temperature

    if "frontend_api_base_url" in values and isinstance(values["frontend_api_base_url"], str):
        frontend_api_base_url = values["frontend_api_base_url"].strip()
        if frontend_api_base_url:
            settings.frontend_api_base_url = frontend_api_base_url


def _string_or_list(value: object) -> str | list[str] | None:
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, list):
        return [str(item) for item in value]
    return None


def _int_in_range(value: object, *, minimum: int, maximum: int) -> int | None:
    if not isinstance(value, int):
        return None
    if minimum <= value <= maximum:
        return value
    return None


def _float_in_range(value: object, *, minimum: float, maximum: float) -> float | None:
    if not isinstance(value, int | float):
        return None
    as_float = float(value)
    if minimum <= as_float <= maximum:
        return as_float
    return None
