from functools import lru_cache
from typing import Annotated

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


DEFAULT_CORS_ALLOW_ORIGINS = [
    "http://localhost:1420",
    "http://127.0.0.1:1420",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "tauri://localhost",
]

DEFAULT_CODEX_MODEL_OPTIONS = [
    "gpt-5.5",
    "gpt-5.4",
    "gpt-5.4-mini",
    "gpt-5.3-codex",
    "gpt-5.3-codex-spark",
    "gpt-5.2",
]
CODEX_MODEL_OPTION_SET = set(DEFAULT_CODEX_MODEL_OPTIONS)
DEFAULT_CODEX_REASONING_EFFORT_OPTIONS = ["low", "medium", "high", "xhigh"]
DEFAULT_CODEX_REASONING_SUMMARY_OPTIONS = ["auto", "concise", "detailed", "none"]
DEFAULT_CODEX_VERBOSITY_OPTIONS = ["low", "medium", "high"]
DEFAULT_AGENT_PROVIDER_OPTIONS = {"codex_cli", "ollama_local_llm"}
DEFAULT_IMAGE_PROVIDER_OPTIONS = {"codex_cli_gpt_image", "local_flux"}
DEFAULT_LOCAL_FLUX_T2I_WORKFLOW_PATH = (
    "workflow/Flux 2 Klein 9B FP8 (Distilled)/"
    "Flux 2 Klein 9B FP8 (Distilled) - Image Generation.json"
)
DEFAULT_LOCAL_FLUX_I2I_ONE_WORKFLOW_PATH = (
    "workflow/Flux 2 Klein 9B FP8 (Distilled)/"
    "Flux 2 Klein 9B FP8 (Distilled) - One Image Edit.json"
)
DEFAULT_LOCAL_FLUX_I2I_TWO_WORKFLOW_PATH = (
    "workflow/Flux 2 Klein 9B FP8 (Distilled)/"
    "Flux 2 Klein 9B FP8 (Distilled) - Two Images Edit.json"
)


def parse_csv(value: str | list[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in value.split(",") if item.strip()]


def normalize_codex_model_options(_value: str | list[str] | None) -> list[str]:
    return DEFAULT_CODEX_MODEL_OPTIONS.copy()


def normalize_codex_default_model(value: str | None) -> str:
    if value is None:
        return DEFAULT_CODEX_MODEL_OPTIONS[0]
    model = value.strip()
    if model in CODEX_MODEL_OPTION_SET:
        return model
    return DEFAULT_CODEX_MODEL_OPTIONS[0]


def normalize_local_flux_model_value(value: str) -> str:
    return value


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
    load_persisted_settings: bool = False
    selected_agent_provider: str = "codex_cli"
    selected_image_provider: str = "codex_cli_gpt_image"

    cors_allow_origins: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: DEFAULT_CORS_ALLOW_ORIGINS.copy()
    )

    codex_binary_path: str = "codex"
    codex_default_model: str = DEFAULT_CODEX_MODEL_OPTIONS[0]
    codex_model_options: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: DEFAULT_CODEX_MODEL_OPTIONS.copy()
    )
    codex_default_reasoning_effort: str = "medium"
    codex_reasoning_effort_options: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: DEFAULT_CODEX_REASONING_EFFORT_OPTIONS.copy()
    )
    codex_default_reasoning_summary: str = "auto"
    codex_default_verbosity: str | None = None
    codex_timeout_seconds: int = 300
    run_codex_smoke: bool = False

    ollama_base_url: str = "http://localhost:11434"
    ollama_selected_model: str | None = None
    ollama_timeout_seconds: int = Field(default=300, ge=1, le=3600)
    ollama_agent_temperature: float = Field(default=0.2, ge=0, le=2)

    local_flux_base_url: str = "http://127.0.0.1:8188"
    local_flux_workflow_path: str = DEFAULT_LOCAL_FLUX_T2I_WORKFLOW_PATH
    local_flux_i2i_one_workflow_path: str = DEFAULT_LOCAL_FLUX_I2I_ONE_WORKFLOW_PATH
    local_flux_i2i_two_workflow_path: str = DEFAULT_LOCAL_FLUX_I2I_TWO_WORKFLOW_PATH
    local_flux_model_path: str = r"flux2\flux-2-klein-9b-fp8mixed.safetensors"
    local_flux_vae_path: str = r"flux\flux2-vae.safetensors"
    local_flux_text_encoder_path: str = r"qwen\qwen_3_8b_fp8mixed.safetensors"
    local_flux_width: int = Field(default=1024, ge=64, le=4096)
    local_flux_height: int = Field(default=1024, ge=64, le=4096)
    local_flux_seed: int | None = None
    local_flux_steps: int = Field(default=4, ge=1, le=150)
    local_flux_cfg: float = Field(default=1.0, ge=0, le=30)
    local_flux_sampler_name: str = "euler"
    local_flux_scheduler: str = "simple"
    local_flux_denoise: float = Field(default=1.0, ge=0, le=1)
    local_flux_guidance: float = Field(default=3.5, ge=0, le=30)
    local_flux_output_prefix: str = "aiimg"
    local_flux_timeout_seconds: int = Field(default=600, ge=1, le=7200)
    run_local_flux_smoke: bool = False

    hf_token: str | None = Field(default=None, repr=False)
    hf_home: str | None = None
    hf_hub_cache: str | None = None

    frontend_api_base_url: str = "http://127.0.0.1:8000"

    @field_validator(
        "cors_allow_origins",
        "codex_model_options",
        "codex_reasoning_effort_options",
        mode="before",
    )
    @classmethod
    def parse_csv_fields(cls, value: str | list[str] | None) -> list[str]:
        return parse_csv(value)

    @field_validator("codex_default_reasoning_effort")
    @classmethod
    def validate_codex_reasoning_effort(cls, value: str) -> str:
        if value not in DEFAULT_CODEX_REASONING_EFFORT_OPTIONS:
            allowed = ", ".join(DEFAULT_CODEX_REASONING_EFFORT_OPTIONS)
            raise ValueError(f"CODEX_DEFAULT_REASONING_EFFORT must be one of: {allowed}")
        return value

    @field_validator("codex_model_options")
    @classmethod
    def validate_codex_model_options(cls, value: list[str]) -> list[str]:
        return normalize_codex_model_options(value)

    @field_validator("codex_default_model")
    @classmethod
    def validate_codex_default_model(cls, value: str) -> str:
        return normalize_codex_default_model(value)

    @field_validator("codex_default_reasoning_summary")
    @classmethod
    def validate_codex_reasoning_summary(cls, value: str) -> str:
        if value not in DEFAULT_CODEX_REASONING_SUMMARY_OPTIONS:
            allowed = ", ".join(DEFAULT_CODEX_REASONING_SUMMARY_OPTIONS)
            raise ValueError(f"CODEX_DEFAULT_REASONING_SUMMARY must be one of: {allowed}")
        return value

    @field_validator("codex_default_verbosity")
    @classmethod
    def validate_codex_verbosity(cls, value: str | None) -> str | None:
        if value is not None and value not in DEFAULT_CODEX_VERBOSITY_OPTIONS:
            allowed = ", ".join(DEFAULT_CODEX_VERBOSITY_OPTIONS)
            raise ValueError(f"CODEX_DEFAULT_VERBOSITY must be one of: {allowed}")
        return value

    @field_validator("selected_agent_provider")
    @classmethod
    def validate_selected_agent_provider(cls, value: str) -> str:
        if value not in DEFAULT_AGENT_PROVIDER_OPTIONS:
            allowed = ", ".join(sorted(DEFAULT_AGENT_PROVIDER_OPTIONS))
            raise ValueError(f"SELECTED_AGENT_PROVIDER must be one of: {allowed}")
        return value

    @field_validator("selected_image_provider")
    @classmethod
    def validate_selected_image_provider(cls, value: str) -> str:
        if value == "diffusers_flux2":
            return "local_flux"
        if value not in DEFAULT_IMAGE_PROVIDER_OPTIONS:
            allowed = ", ".join(sorted(DEFAULT_IMAGE_PROVIDER_OPTIONS))
            raise ValueError(f"SELECTED_IMAGE_PROVIDER must be one of: {allowed}")
        return value

    @field_validator(
        "database_url",
        "hf_token",
        "hf_home",
        "hf_hub_cache",
        "codex_default_verbosity",
        "ollama_selected_model",
        "local_flux_seed",
        mode="before",
    )
    @classmethod
    def empty_secretish_values_to_none(cls, value: object) -> object:
        if value is None:
            return None
        if not isinstance(value, str):
            return value
        stripped = value.strip()
        return stripped or None

    @field_validator(
        "local_flux_base_url",
        "local_flux_workflow_path",
        "local_flux_i2i_one_workflow_path",
        "local_flux_i2i_two_workflow_path",
        "local_flux_model_path",
        "local_flux_vae_path",
        "local_flux_text_encoder_path",
        "local_flux_sampler_name",
        "local_flux_scheduler",
        "local_flux_output_prefix",
    )
    @classmethod
    def strip_required_local_flux_strings(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("Local Flux settings cannot be empty.")
        return normalize_local_flux_model_value(stripped)

    @model_validator(mode="after")
    def ensure_codex_default_model_is_listed(self) -> "Settings":
        if self.codex_default_model not in self.codex_model_options:
            self.codex_default_model = self.codex_model_options[0]
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
