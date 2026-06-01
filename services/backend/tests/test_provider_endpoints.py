import json

from fastapi.testclient import TestClient

from app.core.app_settings_store import persist_app_settings
from app.db.models import AppSettingRecord
from app.db.session import new_session
from app.main import create_app
from app.settings import DEFAULT_CODEX_MODEL_OPTIONS, Settings


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
        json={"model_options": ["gpt-5.5", " gpt-5.3-codex ", "gpt-5.5"]},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["default_model"] == "gpt-5.5"
    assert payload["model_options"] == DEFAULT_CODEX_MODEL_OPTIONS
    assert payload["default_reasoning_effort"] == "medium"
    assert payload["reasoning_effort_options"] == ["low", "medium", "high", "xhigh"]


def test_codex_model_options_patch_rejects_legacy_custom_labels() -> None:
    client = TestClient(create_app(Settings(_env_file=None)))

    response = client.patch(
        "/providers/codex/model-options",
        json={"model_options": ["auto", "custom"]},
    )

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "unknown_codex_model_option"


def test_legacy_codex_persisted_settings_are_normalized_on_startup(tmp_path) -> None:
    database_url = f"sqlite:///{tmp_path / 'app.sqlite3'}"
    seed_app = create_app(
        Settings(
            storage_root=str(tmp_path / "storage"),
            database_url=database_url,
            _env_file=None,
        )
    )
    persist_app_settings(
        seed_app.state.engine,
        {
            "codex_default_model": "auto",
            "codex_model_options": ["auto", "custom"],
        },
    )

    client = TestClient(
        create_app(
            Settings(
                storage_root=str(tmp_path / "storage"),
                database_url=database_url,
                load_persisted_settings=True,
                _env_file=None,
            )
        )
    )

    response = client.get("/providers/codex/models")

    assert response.status_code == 200
    payload = response.json()
    assert payload["default_model"] == DEFAULT_CODEX_MODEL_OPTIONS[0]
    assert payload["model_options"] == DEFAULT_CODEX_MODEL_OPTIONS

    with new_session(seed_app.state.engine) as db:
        default_model = db.get(AppSettingRecord, "codex_default_model")
        model_options = db.get(AppSettingRecord, "codex_model_options")

    assert default_model is not None
    assert json.loads(default_model.value) == DEFAULT_CODEX_MODEL_OPTIONS[0]
    assert model_options is None


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


def test_codex_runtime_options_patch_persists_across_app_restart(tmp_path) -> None:
    database_url = f"sqlite:///{tmp_path / 'app.sqlite3'}"
    client = TestClient(
        create_app(
            Settings(
                storage_root=str(tmp_path / "storage"),
                database_url=database_url,
                load_persisted_settings=True,
                _env_file=None,
            )
        )
    )

    response = client.patch(
        "/providers/codex/runtime-options",
        json={"default_reasoning_effort": "xhigh", "default_reasoning_summary": "detailed"},
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
    persisted = restarted_client.get("/providers/codex/models")

    assert persisted.status_code == 200
    payload = persisted.json()
    assert payload["default_reasoning_effort"] == "xhigh"
    assert payload["default_reasoning_summary"] == "detailed"


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


def test_ollama_default_model_patch_persists_across_app_restart(monkeypatch, tmp_path) -> None:
    from app.api import providers

    monkeypatch.setattr(providers, "get_ollama_models", lambda settings: ["llama3.2", "qwen2.5"])
    database_url = f"sqlite:///{tmp_path / 'app.sqlite3'}"
    client = TestClient(
        create_app(
            Settings(
                storage_root=str(tmp_path / "storage"),
                database_url=database_url,
                load_persisted_settings=True,
                _env_file=None,
            )
        )
    )

    response = client.patch(
        "/providers/ollama/default-model",
        json={"default_model": "qwen2.5"},
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
    persisted = restarted_client.get("/providers/ollama/models")

    assert persisted.status_code == 200
    assert persisted.json()["selected_model"] == "qwen2.5"
