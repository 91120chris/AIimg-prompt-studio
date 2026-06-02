from fastapi.testclient import TestClient

from app.api import models as models_api
from app.core.flux_model_manager import FluxInstallError, FluxInstallResult
from app.core.session_workspace import new_id
from app.db.models import LogRecord
from app.db.session import new_session
from app.main import create_app
from app.settings import Settings


def make_flux_pipeline_dir(tmp_path, name: str = "private-flux"):
    model_dir = tmp_path / "local_models" / name
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "model_index.json").write_text("{}", encoding="utf-8")
    return model_dir


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


def test_flux_status_and_set_path_do_not_return_raw_model_path(tmp_path) -> None:
    _, client = make_client(tmp_path)
    model_path = str(make_flux_pipeline_dir(tmp_path, "flux-secret-path"))

    initial = client.get("/models/flux/status")
    set_path = client.post("/models/flux/set-path", json={"model_path": model_path})
    listed = client.get("/models")

    assert initial.status_code == 200
    assert initial.json()["status"] == "not_installed"
    assert set_path.status_code == 200
    payload = set_path.json()
    assert payload["status"] == "path_selected"
    assert payload["path_configured"] is True
    assert payload["path_label"] == "flux-secret-path"
    assert model_path not in set_path.text
    assert listed.status_code == 200
    assert listed.json()[0]["provider"] == "diffusers_flux2_klein_9b_fp8"
    assert model_path not in listed.text


def fake_flux_install(tmp_path):
    def _install(_settings) -> FluxInstallResult:
        model_dir = make_flux_pipeline_dir(tmp_path, "downloaded-flux")
        return FluxInstallResult(
            model_path=str(model_dir),
            repo_id="black-forest-labs/FLUX.2-klein-9b",
            revision=None,
        )

    return _install


def test_flux_install_and_unload_update_status(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(models_api, "install_flux_snapshot", fake_flux_install(tmp_path))
    _, client = make_client(tmp_path, hf_token="hf_test_token")

    install = client.post("/models/flux/install")
    unload = client.post("/models/flux/unload")

    assert install.status_code == 200
    assert install.json()["status"] == "installed"
    assert install.json()["path_label"] == "downloaded-flux"
    assert unload.status_code == 200
    assert unload.json()["status"] == "unloaded"
    assert unload.json()["installed"] is True


def test_flux_readiness_does_not_return_hf_token_or_raw_path(tmp_path) -> None:
    _, client = make_client(tmp_path, hf_token="hf_secret_token")
    model_path = str(make_flux_pipeline_dir(tmp_path))

    client.post("/models/flux/set-path", json={"model_path": model_path})
    readiness = client.get("/models/flux/readiness")

    assert readiness.status_code == 200
    payload = readiness.json()
    assert payload["hf_token_configured"] is True
    assert payload["can_queue_install"] is True
    assert payload["path_configured"] is True
    assert payload["path_label"] == "private-flux"
    assert "hf_secret_token" not in readiness.text
    assert model_path not in readiness.text


def test_flux_install_requires_hf_token(tmp_path) -> None:
    _, client = make_client(tmp_path)

    response = client.post("/models/flux/install")

    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "hf_token_required"


def test_flux_actions_preserve_path_label_and_write_safe_logs(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(models_api, "install_flux_snapshot", fake_flux_install(tmp_path))
    _, client = make_client(tmp_path, hf_token="hf_test_token")
    model_path = str(make_flux_pipeline_dir(tmp_path))

    set_path = client.post("/models/flux/set-path", json={"model_path": model_path})
    install = client.post("/models/flux/install")
    unload = client.post("/models/flux/unload")
    logs = client.get("/logs")

    assert set_path.status_code == 200
    assert install.status_code == 200
    assert install.json()["path_configured"] is True
    assert install.json()["status"] == "installed"
    assert install.json()["path_label"] == "downloaded-flux"
    assert unload.status_code == 200
    assert unload.json()["path_configured"] is True
    assert unload.json()["path_label"] == "downloaded-flux"
    assert logs.status_code == 200
    assert "FLUX model path selected: private-flux" in logs.text
    assert "FLUX install started." in logs.text
    assert "FLUX install completed: downloaded-flux" in logs.text
    assert "FLUX provider unloaded." in logs.text
    assert model_path not in logs.text


def test_flux_install_failure_records_failed_status_without_secret(monkeypatch, tmp_path) -> None:
    def fail_install(_settings) -> FluxInstallResult:
        raise FluxInstallError(
            code="hf_access_denied",
            message="Hugging Face denied access to the configured FLUX repository.",
            suggestion="Accept the model terms and make sure the token has permission.",
        )

    monkeypatch.setattr(models_api, "install_flux_snapshot", fail_install)
    _, client = make_client(tmp_path, hf_token="hf_secret_token")

    response = client.post("/models/flux/install")
    status = client.get("/models/flux/status")
    logs = client.get("/logs")

    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "hf_access_denied"
    assert status.status_code == 200
    assert status.json()["status"] == "failed"
    assert status.json()["error_code"] == "hf_access_denied"
    assert "hf_secret_token" not in response.text
    assert "hf_secret_token" not in status.text
    assert "hf_secret_token" not in logs.text

    monkeypatch.setattr(models_api, "install_flux_snapshot", fake_flux_install(tmp_path))
    retry = client.post("/models/flux/install")

    assert retry.status_code == 200
    assert retry.json()["status"] == "installed"
    assert retry.json()["error_code"] is None


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
