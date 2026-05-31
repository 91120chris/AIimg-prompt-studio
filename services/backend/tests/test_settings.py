from fastapi.testclient import TestClient

from app.main import create_app
from app.schemas import SafeSettingsResponse
from app.settings import Settings


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
    settings = Settings(codex_model_options="auto, gpt-5.5 , custom", _env_file=None)

    assert settings.codex_model_options == ["auto", "gpt-5.5", "custom"]


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
