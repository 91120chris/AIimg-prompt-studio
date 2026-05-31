from fastapi.testclient import TestClient
from sqlmodel import select

from app.db.models import GeneratedImageRecord, GenerationJobRecord
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


def test_confirm_generation_creates_queued_job_without_auto_image(tmp_path) -> None:
    app, client = make_test_app(tmp_path)
    session_id = client.post("/sessions", json={"title": "Generation"}).json()["session_id"]

    response = client.post("/generation/confirm", json=generation_payload(session_id))

    assert response.status_code == 200
    payload = response.json()
    assert payload["job_id"].startswith("job_")
    assert payload["session_id"] == session_id
    assert payload["status"] == "queued"
    assert payload["images"] == []
    assert "storage_path" not in response.text

    with new_session(app.state.engine) as db:
        jobs = db.exec(select(GenerationJobRecord)).all()
        images = db.exec(select(GeneratedImageRecord)).all()

    assert len(jobs) == 1
    assert images == []
    assert "cinematic city" in jobs[0].parameters_json


def test_confirm_generation_requires_optimized_prompt(tmp_path) -> None:
    _, client = make_test_app(tmp_path)
    session_id = client.post("/sessions", json={"title": "Generation"}).json()["session_id"]

    response = client.post("/generation/confirm", json=generation_payload(session_id, "  "))

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "optimized_prompt_required"


def test_cancel_generation_marks_queued_job_cancelled(tmp_path) -> None:
    _, client = make_test_app(tmp_path)
    session_id = client.post("/sessions", json={"title": "Generation"}).json()["session_id"]
    job_id = client.post("/generation/confirm", json=generation_payload(session_id)).json()["job_id"]

    response = client.post("/generation/cancel", json={"job_id": job_id})

    assert response.status_code == 200
    assert response.json()["status"] == "cancelled"


def test_get_generation_job_returns_status(tmp_path) -> None:
    _, client = make_test_app(tmp_path)
    session_id = client.post("/sessions", json={"title": "Generation"}).json()["session_id"]
    job_id = client.post("/generation/confirm", json=generation_payload(session_id)).json()["job_id"]

    response = client.get(f"/generation/{job_id}")

    assert response.status_code == 200
    assert response.json()["job_id"] == job_id
