from fastapi.testclient import TestClient

from app.db.models import PromptVersionRecord
from app.db.session import new_session
from app.main import create_app
from app.settings import Settings


def make_client(tmp_path) -> tuple[object, TestClient]:
    app = create_app(
        Settings(
            storage_root=str(tmp_path / "storage"),
            database_url=f"sqlite:///{tmp_path / 'app.sqlite3'}",
            _env_file=None,
        )
    )
    return app, TestClient(app)


def test_prompt_versions_can_be_listed_read_and_set_current(tmp_path) -> None:
    app, client = make_client(tmp_path)
    session_id = client.post("/sessions", json={"title": "Prompt versions"}).json()["session_id"]
    with new_session(app.state.engine) as db:
        db.add(
            PromptVersionRecord(
                prompt_version_id="promptv_1",
                session_id=session_id,
                prompt_text="first optimized prompt",
                title="First",
                source="optimized_prompt",
                metadata_json='{"questionnaire_id":"q1"}',
                created_at="2026-01-01T00:00:00Z",
            )
        )
        db.add(
            PromptVersionRecord(
                prompt_version_id="promptv_2",
                session_id=session_id,
                prompt_text="feedback refined prompt",
                title="Feedback",
                source="feedback_refine",
                metadata_json='{"job_id":"job1"}',
                created_at="2026-01-02T00:00:00Z",
            )
        )
        db.commit()

    list_response = client.get(f"/sessions/{session_id}/prompt-versions")
    get_response = client.get(f"/sessions/{session_id}/prompt-versions/promptv_1")
    patch_response = client.patch(
        f"/sessions/{session_id}/current-prompt-version",
        json={"prompt_version_id": "promptv_1"},
    )
    relist_response = client.get(f"/sessions/{session_id}/prompt-versions")

    assert list_response.status_code == 200
    versions = list_response.json()
    assert versions[0]["prompt_version_id"] == "promptv_2"
    assert versions[0]["is_current"] is True
    assert get_response.status_code == 200
    assert get_response.json()["metadata"]["questionnaire_id"] == "q1"
    assert patch_response.status_code == 200
    assert patch_response.json()["prompt_version_id"] == "promptv_1"
    assert patch_response.json()["is_current"] is True
    assert relist_response.json()[-1]["prompt_version_id"] == "promptv_1"
    assert relist_response.json()[-1]["is_current"] is True
