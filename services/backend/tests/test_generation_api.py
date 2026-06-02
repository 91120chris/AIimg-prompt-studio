from fastapi.testclient import TestClient
from PIL import Image
from sqlmodel import select

from app.core.image_records import register_generated_image
from app.db.models import GeneratedImageRecord, GenerationJobRecord, ModelStatusRecord
from app.db.session import new_session
from app.main import create_app
from app.settings import Settings


def make_test_app(tmp_path):
    settings = Settings(
        storage_root=str(tmp_path / "storage"),
        database_url=f"sqlite:///{tmp_path / 'app.sqlite3'}",
        _env_file=None,
    )
    app = create_app(settings)
    return app, TestClient(app)


def generation_payload(session_id: str, optimized_prompt: str = "cinematic city") -> dict:
    return {
        "session_id": session_id,
        "provider": "codex_cli_gpt_image",
        "mode": "t2i",
        "original_prompt": "城市",
        "optimized_prompt": optimized_prompt,
        "parameters": {"steps": 28, "guidance": 3.5, "seed": None},
        "reference_image_ids": [],
    }


def flux_generation_payload(session_id: str, optimized_prompt: str = "cinematic city") -> dict:
    payload = generation_payload(session_id, optimized_prompt)
    payload["provider"] = "diffusers_flux2"
    payload["parameters"] = {
        "steps": 8,
        "guidance": 1.0,
        "seed": 123,
        "width": 512,
        "height": 512,
    }
    return payload


def fake_success_provider(tmp_path):
    class FakeCodexImageProvider:
        def __init__(self, settings):
            self.settings = settings

        def generate(self, db, *, job, payload, reference_images):
            source_path = tmp_path / "fake-output.png"
            Image.new("RGB", (32, 24), color="purple").save(source_path)
            return [
                register_generated_image(
                    db,
                    self.settings,
                    session_id=payload.session_id,
                    source_path=source_path,
                    provider=payload.provider,
                    seed=payload.parameters.seed,
                )
            ]

    return FakeCodexImageProvider


def fake_success_flux_provider(tmp_path):
    class FakeDiffusersFluxProvider:
        def __init__(self, settings):
            self.settings = settings

        def generate(self, db, *, job, payload, model_path):
            source_path = tmp_path / "fake-flux-output.png"
            Image.new("RGB", (40, 40), color="teal").save(source_path)
            return [
                register_generated_image(
                    db,
                    self.settings,
                    session_id=payload.session_id,
                    source_path=source_path,
                    provider=payload.provider,
                    seed=payload.parameters.seed,
                )
            ]

    return FakeDiffusersFluxProvider


def test_confirm_generation_runs_provider_and_returns_safe_image(monkeypatch, tmp_path) -> None:
    from app.api import generation

    monkeypatch.setattr(generation, "CodexImageProvider", fake_success_provider(tmp_path))
    app, client = make_test_app(tmp_path)
    session_id = client.post("/sessions", json={"title": "Generation"}).json()["session_id"]

    response = client.post("/generation/confirm", json=generation_payload(session_id))

    assert response.status_code == 200
    payload = response.json()
    assert payload["job_id"].startswith("job_")
    assert payload["session_id"] == session_id
    assert payload["status"] == "succeeded"
    assert payload["images"][0]["url"].startswith(f"/files/sessions/{session_id}/generated-images/img_")
    assert "storage_path" not in response.text

    with new_session(app.state.engine) as db:
        jobs = db.exec(select(GenerationJobRecord)).all()
        images = db.exec(select(GeneratedImageRecord)).all()

    assert len(jobs) == 1
    assert len(images) == 1
    assert "cinematic city" in jobs[0].parameters_json


def test_confirm_generation_requires_optimized_prompt(tmp_path) -> None:
    _, client = make_test_app(tmp_path)
    session_id = client.post("/sessions", json={"title": "Generation"}).json()["session_id"]

    response = client.post("/generation/confirm", json=generation_payload(session_id, "  "))

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "optimized_prompt_required"


def test_cancel_generation_marks_queued_job_cancelled(tmp_path) -> None:
    app, client = make_test_app(tmp_path)
    session_id = client.post("/sessions", json={"title": "Generation"}).json()["session_id"]
    with new_session(app.state.engine) as db:
        job = GenerationJobRecord(
            job_id="job_cancel",
            session_id=session_id,
            provider="codex_cli_gpt_image",
            mode="t2i",
            status="queued",
        )
        db.add(job)
        db.commit()
    job_id = "job_cancel"

    response = client.post("/generation/cancel", json={"job_id": job_id})

    assert response.status_code == 200
    assert response.json()["status"] == "cancelled"


def test_get_generation_job_returns_status(tmp_path) -> None:
    app, client = make_test_app(tmp_path)
    session_id = client.post("/sessions", json={"title": "Generation"}).json()["session_id"]
    with new_session(app.state.engine) as db:
        job = GenerationJobRecord(
            job_id="job_status",
            session_id=session_id,
            provider="codex_cli_gpt_image",
            mode="t2i",
            status="queued",
        )
        db.add(job)
        db.commit()
    job_id = "job_status"

    response = client.get(f"/generation/{job_id}")

    assert response.status_code == 200
    assert response.json()["job_id"] == job_id


def test_confirm_generation_records_provider_failure(monkeypatch, tmp_path) -> None:
    from app.api import generation
    from app.providers.codex.codex_image_provider import CodexImageProviderError
    from app.schemas.errors import StructuredError

    class FailingCodexImageProvider:
        def __init__(self, settings):
            self.settings = settings

        def generate(self, db, *, job, payload, reference_images):
            raise CodexImageProviderError(
                StructuredError(
                    code="codex_image_failed",
                    message="fake failure",
                    suggestion="fake suggestion",
                )
            )

    monkeypatch.setattr(generation, "CodexImageProvider", FailingCodexImageProvider)
    _, client = make_test_app(tmp_path)
    session_id = client.post("/sessions", json={"title": "Generation"}).json()["session_id"]

    response = client.post("/generation/confirm", json=generation_payload(session_id))

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "failed"
    assert payload["error"]["code"] == "codex_image_failed"


def test_confirm_generation_runs_diffusers_flux_provider(monkeypatch, tmp_path) -> None:
    from app.api import generation

    monkeypatch.setattr(generation, "DiffusersFluxProvider", fake_success_flux_provider(tmp_path))
    app, client = make_test_app(tmp_path)
    session_id = client.post("/sessions", json={"title": "Flux"}).json()["session_id"]
    with new_session(app.state.engine) as db:
        db.add(
            ModelStatusRecord(
                model_status_id="diffusers_flux2_klein_9b_fp8",
                provider="diffusers_flux2_klein_9b_fp8",
                status="path_selected",
                details_json='{"model_path":"C:/safe/model/flux"}',
            )
        )
        db.commit()

    response = client.post("/generation/confirm", json=flux_generation_payload(session_id))

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "succeeded"
    assert payload["provider"] == "diffusers_flux2"
    assert payload["images"][0]["provider"] == "diffusers_flux2"
    assert payload["images"][0]["seed"] == 123
    assert "storage_path" not in response.text


def test_confirm_flux_generation_requires_model_path(tmp_path) -> None:
    _, client = make_test_app(tmp_path)
    session_id = client.post("/sessions", json={"title": "Flux"}).json()["session_id"]

    response = client.post("/generation/confirm", json=flux_generation_payload(session_id))

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "flux_model_not_configured"
