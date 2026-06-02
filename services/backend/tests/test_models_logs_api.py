from fastapi.testclient import TestClient

from app.core.session_workspace import new_id
from app.db.models import LogRecord
from app.db.session import new_session
from app.main import create_app
from app.settings import Settings


def make_client(tmp_path, hf_token: str | None = None) -> tuple[object, TestClient]:
    app = create_app(
        Settings(
            storage_root=str(tmp_path / "storage"),
            database_url=f"sqlite:///{tmp_path / 'app.sqlite3'}",
            hf_token=hf_token,
            _env_file=None,
        )
    )
    return app, TestClient(app)


def test_models_endpoint_reports_local_flux_without_raw_secret(monkeypatch, tmp_path) -> None:
    from app.api import models

    class FakeLocalFluxClient:
        def __init__(self, settings):
            self.settings = settings

        def get_system_stats(self):
            return {"system": "ok"}

    monkeypatch.setattr(models, "LocalFluxClient", FakeLocalFluxClient)
    _, client = make_client(tmp_path, hf_token="hf_secret_token")

    response = client.get("/models")

    assert response.status_code == 200
    payload = response.json()
    assert payload[0]["provider"] == "local_flux"
    assert payload[0]["status"] == "available"
    assert payload[0]["path_label"] == "flux-2-klein-9b-fp8mixed.safetensors"
    assert "hf_secret_token" not in response.text


def test_local_flux_settings_patch_persists_and_returns_editable_paths(tmp_path) -> None:
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
    model_path = str(tmp_path / "models" / "flux.safetensors")

    response = client.patch(
        "/providers/local-flux/settings",
        json={
            "base_url": "http://127.0.0.1:8189",
            "model_path": model_path,
            "steps": 12,
            "guidance": 4.2,
            "seed": 123,
        },
    )

    assert response.status_code == 200
    assert response.json()["model_path"] == model_path

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
    persisted = restarted_client.get("/providers/local-flux/settings")

    assert persisted.status_code == 200
    payload = persisted.json()
    assert payload["base_url"] == "http://127.0.0.1:8189"
    assert payload["model_path"] == model_path
    assert payload["steps"] == 12
    assert payload["guidance"] == 4.2
    assert payload["seed"] == 123


def test_local_flux_settings_patch_normalizes_legacy_model_values(tmp_path) -> None:
    _, client = make_client(tmp_path)

    response = client.patch(
        "/providers/local-flux/settings",
        json={
            "vae_path": "flux2-vae.safetensors",
            "text_encoder_path": "qwen_3_8b_fp8mixed.safetensors",
        },
    )

    assert response.status_code == 200
    assert response.json()["vae_path"] == "flux\\flux2-vae.safetensors"
    assert response.json()["text_encoder_path"] == "qwen\\qwen_3_8b_fp8mixed.safetensors"


def test_local_flux_status_handles_offline_backend(monkeypatch, tmp_path) -> None:
    from app.api import providers
    from app.providers.local_flux.client import LocalFluxClientError

    class OfflineLocalFluxClient:
        def __init__(self, settings):
            self.settings = settings

        def get_system_stats(self):
            raise LocalFluxClientError("offline")

    monkeypatch.setattr(providers, "LocalFluxClient", OfflineLocalFluxClient)
    _, client = make_client(tmp_path)

    response = client.get("/providers/local-flux/status")

    assert response.status_code == 200
    assert response.json()["provider"] == "local_flux"
    assert response.json()["available"] is False
    assert response.json()["message"] == "Local Flux 未連線"


def test_logs_endpoint_returns_recent_logs(tmp_path) -> None:
    app, client = make_client(tmp_path)
    with new_session(app.state.engine) as db:
        db.add(LogRecord(log_id=new_id("log"), level="info", message="first"))
        db.add(LogRecord(log_id=new_id("log"), level="warning", message="second"))
        db.commit()

    response = client.get("/logs?limit=1")

    assert response.status_code == 200
    payload = response.json()
    assert len(payload) == 1
    assert payload[0]["level"] in {"info", "warning"}
    assert "message" in payload[0]
