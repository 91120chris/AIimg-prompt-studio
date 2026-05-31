from fastapi.testclient import TestClient

from app.main import create_app
from app.settings import Settings


def test_health_returns_expected_shape() -> None:
    client = TestClient(create_app(Settings(_env_file=None)))

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "app_name": "Prompt Optimizer Studio",
        "version": "0.1.0",
        "environment": "development",
    }


def test_health_does_not_serialize_hf_token() -> None:
    secret = "hf_secret_should_not_appear"
    client = TestClient(create_app(Settings(hf_token=secret, _env_file=None)))

    response = client.get("/health")

    assert response.status_code == 200
    assert secret not in response.text
    assert "hf_token" not in response.text
