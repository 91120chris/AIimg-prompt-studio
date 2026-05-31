from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image
from sqlmodel import Session

from app.core.file_store import generated_image_response
from app.core.image_records import register_generated_image
from app.main import create_app
from app.settings import Settings


def make_png(path, size: tuple[int, int] = (96, 64), color: str = "blue") -> None:
    Image.new("RGB", size, color=color).save(path, format="PNG")


def make_png_bytes(size: tuple[int, int] = (96, 64), color: str = "green") -> bytes:
    buffer = BytesIO()
    Image.new("RGB", size, color=color).save(buffer, format="PNG")
    return buffer.getvalue()


def make_test_app(tmp_path):
    settings = Settings(
        storage_root=str(tmp_path / "storage"),
        database_url=f"sqlite:///{tmp_path / 'app.sqlite3'}",
        _env_file=None,
    )
    app = create_app(settings)
    return settings, app, TestClient(app)


def test_thumbnail_generated_for_uploaded_reference_image(tmp_path) -> None:
    _, _, client = make_test_app(tmp_path)
    session_id = client.post("/sessions", json={"title": "Test"}).json()["session_id"]

    response = client.post(
        f"/sessions/{session_id}/reference-images",
        data={"slot": "1", "role": "primary_reference"},
        files={"file": ("reference.png", make_png_bytes(), "image/png")},
    )

    assert response.status_code == 200
    assert response.json()["thumbnail_url"] is not None
    assert "storage_path" not in response.text


def test_thumbnail_generated_for_registered_generated_image(tmp_path) -> None:
    settings, app, client = make_test_app(tmp_path)
    session_id = client.post("/sessions", json={"title": "Test"}).json()["session_id"]
    source_path = tmp_path / "generated.png"
    make_png(source_path)

    with Session(app.state.engine) as db:
        record = register_generated_image(
            db,
            settings,
            session_id=session_id,
            source_path=source_path,
            provider="codex_cli_gpt_image",
            seed=123,
        )
        payload = generated_image_response(record).model_dump()

    assert payload["thumbnail_url"] is not None
    assert payload["url"].startswith(f"/files/sessions/{session_id}/generated-images/img_")
    assert "storage_path" not in str(payload)
    assert client.get(payload["thumbnail_url"]).status_code == 200


def test_thumbnail_failure_uses_null_thumbnail_url(monkeypatch, tmp_path) -> None:
    import app.api.sessions as sessions_api

    monkeypatch.setattr(sessions_api, "generate_thumbnail", lambda source, target: None)
    _, _, client = make_test_app(tmp_path)
    session_id = client.post("/sessions", json={"title": "Test"}).json()["session_id"]

    response = client.post(
        f"/sessions/{session_id}/reference-images",
        data={"slot": "1", "role": "primary_reference"},
        files={"file": ("reference.png", make_png_bytes(), "image/png")},
    )

    assert response.status_code == 200
    assert response.json()["thumbnail_url"] is None
    assert "storage_path" not in response.text
