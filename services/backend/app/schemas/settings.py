from app.schemas.base import StrictBaseModel
from app.settings import Settings


class SafeSettingsResponse(StrictBaseModel):
    app_name: str
    app_version: str
    app_env: str
    backend_host: str
    backend_port: int
    cors_allow_origins: list[str]
    codex_binary_path: str
    codex_default_model: str
    codex_model_options: list[str]
    codex_default_reasoning_effort: str
    codex_reasoning_effort_options: list[str]
    codex_default_reasoning_summary: str
    codex_default_verbosity: str | None
    codex_timeout_seconds: int
    run_codex_smoke: bool
    ollama_base_url: str
    ollama_selected_model: str | None
    ollama_timeout_seconds: int
    ollama_agent_temperature: float
    hf_home_configured: bool
    hf_hub_cache_configured: bool
    frontend_api_base_url: str

    @classmethod
    def from_settings(cls, settings: Settings) -> "SafeSettingsResponse":
        return cls(
            app_name=settings.app_name,
            app_version=settings.app_version,
            app_env=settings.app_env,
            backend_host=settings.backend_host,
            backend_port=settings.backend_port,
            cors_allow_origins=settings.cors_allow_origins,
            codex_binary_path=settings.codex_binary_path,
            codex_default_model=settings.codex_default_model,
            codex_model_options=settings.codex_model_options,
            codex_default_reasoning_effort=settings.codex_default_reasoning_effort,
            codex_reasoning_effort_options=settings.codex_reasoning_effort_options,
            codex_default_reasoning_summary=settings.codex_default_reasoning_summary,
            codex_default_verbosity=settings.codex_default_verbosity,
            codex_timeout_seconds=settings.codex_timeout_seconds,
            run_codex_smoke=settings.run_codex_smoke,
            ollama_base_url=settings.ollama_base_url,
            ollama_selected_model=settings.ollama_selected_model,
            ollama_timeout_seconds=settings.ollama_timeout_seconds,
            ollama_agent_temperature=settings.ollama_agent_temperature,
            hf_home_configured=settings.hf_home is not None,
            hf_hub_cache_configured=settings.hf_hub_cache is not None,
            frontend_api_base_url=settings.frontend_api_base_url,
        )
