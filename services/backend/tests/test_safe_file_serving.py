from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image
from sqlmodel import Session

from app.db.models import GeneratedImageRecord
from app.main import create_app
from app.settings import Settings


def make_png_bytes(size: tuple[int, int] = (32, 24), color: str = "red") -> bytes:
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
    return app, TestClient(app)


def test_reference_image_upload_uses_safe_urls_and_serves_files(tmp_path) -> None:
    _, client = make_test_app(tmp_path)
    session_id = client.post("/sessions", json={"title": "Test"}).json()["session_id"]

    response = client.post(
        f"/sessions/{session_id}/reference-images",
        data={"slot": "1", "role": "primary_reference"},
        files={"file": ("product.png", make_png_bytes(), "image/png")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["session_id"] == session_id
    assert payload["reference_image_id"].startswith("ref_")
    assert payload["url"].startswith(f"/files/sessions/{session_id}/reference-images/ref_")
    assert payload["thumbnail_url"].endswith("?variant=thumbnail")
    assert payload["filename"] == "product.png"
    assert payload["width"] == 32
    assert payload["height"] == 24
    assert "storage_path" not in response.text

    original = client.get(payload["url"])
    thumbnail = client.get(payload["thumbnail_url"])

    assert original.status_code == 200
    assert thumbnail.status_code == 200


def test_invalid_reference_image_returns_structured_error_without_raw_path(tmp_path) -> None:
    _, client = make_test_app(tmp_path)
    session_id = client.post("/sessions", json={"title": "Test"}).json()["session_id"]

    response = client.post(
        f"/sessions/{session_id}/reference-images",
        data={"slot": "1", "role": "primary_reference"},
        files={"file": ("bad.png", b"not an image", "image/png")},
    )

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "invalid_image"
    assert "storage_path" not in response.text


def test_safe_file_endpoint_rejects_record_outside_session_workspace(tmp_path) -> None:
    app, client = make_test_app(tmp_path)
    session_id = client.post("/sessions", json={"title": "Test"}).json()["session_id"]
    outside_path = tmp_path / "outside.png"
    outside_path.write_bytes(make_png_bytes())

    with Session(app.state.engine) as db:
        db.add(
            GeneratedImageRecord(
                image_id="img_outside",
                session_id=session_id,
                role="optimized_prompt",
                filename="outside.png",
                storage_path=str(outside_path),
                width=32,
                height=24,
                provider="codex_cli_gpt_image",
            )
        )
        db.commit()

    response = client.get(f"/files/sessions/{session_id}/generated-images/img_outside")

    assert response.status_code == 404
    assert "outside.png" not in response.text
