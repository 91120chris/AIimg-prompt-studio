from fastapi.testclient import TestClient

from app.main import create_app
from app.schemas import SafeSettingsResponse
from app.settings import DEFAULT_CODEX_MODEL_OPTIONS, Settings


def test_cors_allow_origins_parses_csv_with_whitespace() -> None:
    settings = Settings(
        cors_allow_origins="http://localhost:1420, http://127.0.0.1:1420 , tauri://localhost",
        _env_file=None,
    )

    assert settings.cors_allow_origins == [
        "http://localhost:1420",
        "http://127.0.0.1:1420",
        "tauri://localhost",
    ]


def test_cors_allow_origins_parses_single_value() -> None:
    settings = Settings(cors_allow_origins="http://localhost:1420", _env_file=None)

    assert settings.cors_allow_origins == ["http://localhost:1420"]


def test_codex_model_options_parses_csv_with_whitespace() -> None:
    settings = Settings(
        codex_model_options="auto, gpt-5.5 , custom, gpt-5.3-codex",
        _env_file=None,
    )

    assert settings.codex_model_options == DEFAULT_CODEX_MODEL_OPTIONS


def test_codex_default_model_legacy_auto_uses_first_supported_model() -> None:
    settings = Settings(codex_default_model="auto", _env_file=None)

    assert settings.codex_default_model == "gpt-5.5"


def test_codex_reasoning_effort_options_parses_csv_with_whitespace() -> None:
    settings = Settings(
        codex_reasoning_effort_options="low, medium , high, xhigh",
        _env_file=None,
    )

    assert settings.codex_reasoning_effort_options == ["low", "medium", "high", "xhigh"]


def test_safe_settings_response_does_not_include_hf_token() -> None:
    secret = "hf_secret_should_not_appear"
    safe_settings = SafeSettingsResponse.from_settings(Settings(hf_token=secret, _env_file=None))

    serialized = safe_settings.model_dump_json()

    assert secret not in serialized
    assert "hf_token" not in serialized


def test_settings_safe_endpoint_does_not_include_hf_token() -> None:
    secret = "hf_secret_should_not_appear"
    client = TestClient(create_app(Settings(hf_token=secret, _env_file=None)))

    response = client.get("/settings/safe")

    assert response.status_code == 200
    assert secret not in response.text
    assert "hf_token" not in response.text


def test_safe_settings_patch_persists_across_app_restart(tmp_path) -> None:
    database_url = f"sqlite:///{tmp_path / 'app.sqlite3'}"
    settings = Settings(
        storage_root=str(tmp_path / "storage"),
        database_url=database_url,
        _env_file=None,
    )
    client = TestClient(create_app(settings))

    response = client.patch(
        "/settings/safe",
        json={
            "selected_agent_provider": "ollama_local_llm",
            "selected_image_provider": "diffusers_flux2",
            "ollama_timeout_seconds": 123,
            "ollama_agent_temperature": 0.4,
        },
    )

    assert response.status_code == 200

    restarted_client = TestClient(
        create_app(
            Settings(
                storage_root=str(tmp_path / "storage"),
                database_url=database_url,
                load_persisted_settings=True,
                _env_file=None,
            )
        )
    )
    persisted = restarted_client.get("/settings/safe")

    assert persisted.status_code == 200
    payload = persisted.json()
    assert payload["selected_agent_provider"] == "ollama_local_llm"
    assert payload["selected_image_provider"] == "diffusers_flux2"
    assert payload["ollama_timeout_seconds"] == 123
    assert payload["ollama_agent_temperature"] == 0.4


def test_safe_settings_patch_rejects_empty_cors_origins(tmp_path) -> None:
    client = TestClient(
        create_app(
            Settings(
                storage_root=str(tmp_path / "storage"),
                database_url=f"sqlite:///{tmp_path / 'app.sqlite3'}",
                _env_file=None,
            )
        )
    )

    response = client.patch("/settings/safe", json={"cors_allow_origins": []})

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "empty_cors_allow_origins"


def test_cors_preflight_uses_configured_origin() -> None:
    origin = "http://localhost:1420"
    client = TestClient(create_app(Settings(cors_allow_origins=origin, _env_file=None)))

    response = client.options(
        "/health",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "GET",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == origin
