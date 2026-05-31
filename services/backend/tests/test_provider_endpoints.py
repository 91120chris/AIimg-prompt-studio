from fastapi.testclient import TestClient

from app.main import create_app
from app.settings import Settings


def test_codex_status_endpoint_handles_missing_binary() -> None:
    client = TestClient(
        create_app(Settings(codex_binary_path="definitely-not-real-codex", _env_file=None))
    )

    response = client.get("/providers/codex/status")

    assert response.status_code == 200
    payload = response.json()
    assert payload["provider"] == "codex_cli"
    assert payload["available"] is False
    assert "resolved_path" not in payload


def test_codex_model_options_patch_updates_in_memory_settings() -> None:
    client = TestClient(create_app(Settings(_env_file=None)))

    response = client.patch(
        "/providers/codex/model-options",
        json={"model_options": ["auto", " custom ", "auto"]},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["default_model"] == "auto"
    assert payload["model_options"] == ["auto", "custom"]
    assert payload["default_reasoning_effort"] == "medium"
    assert payload["reasoning_effort_options"] == ["low", "medium", "high", "xhigh"]


def test_codex_runtime_options_patch_updates_reasoning_effort() -> None:
    client = TestClient(create_app(Settings(_env_file=None)))

    response = client.patch(
        "/providers/codex/runtime-options",
        json={
            "default_model": "gpt-5.5",
            "default_reasoning_effort": "xhigh",
            "default_reasoning_summary": "concise",
            "default_verbosity": "high",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["default_model"] == "gpt-5.5"
    assert payload["default_reasoning_effort"] == "xhigh"
    assert payload["default_reasoning_summary"] == "concise"
    assert payload["default_verbosity"] == "high"


def test_codex_runtime_options_patch_can_clear_verbosity() -> None:
    client = TestClient(create_app(Settings(codex_default_verbosity="high", _env_file=None)))

    response = client.patch(
        "/providers/codex/runtime-options",
        json={"default_verbosity": None},
    )

    assert response.status_code == 200
    assert response.json()["default_verbosity"] is None


def test_secret_status_never_returns_secret_value() -> None:
    secret = "hf_secret_should_not_appear"
    client = TestClient(create_app(Settings(hf_token=secret, _env_file=None)))

    response = client.get("/security/secrets/status")

    assert response.status_code == 200
    assert response.json()["hf_token_configured"] is True
    assert secret not in response.text
    assert "hf_token" in response.text
    assert "hf_secret" not in response.text


def test_ollama_model_patch_rejects_unknown_live_model(monkeypatch) -> None:
    from app.api import providers

    monkeypatch.setattr(providers, "get_ollama_models", lambda settings: ["llama3.2"])
    client = TestClient(create_app(Settings(_env_file=None)))

    response = client.patch(
        "/providers/ollama/default-model",
        json={"default_model": "missing-model"},
    )

    assert response.status_code == 422


def test_ollama_models_returns_live_selected_model(monkeypatch) -> None:
    from app.api import providers

    monkeypatch.setattr(providers, "get_ollama_models", lambda settings: ["llama3.2", "qwen2.5"])
    client = TestClient(create_app(Settings(ollama_selected_model="qwen2.5", _env_file=None)))

    response = client.get("/providers/ollama/models")

    assert response.status_code == 200
    assert response.json() == {
        "selected_model": "qwen2.5",
        "models": ["llama3.2", "qwen2.5"],
    }
