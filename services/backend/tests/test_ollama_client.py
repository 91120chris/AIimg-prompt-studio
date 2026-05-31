from app.providers.ollama import ollama_client
from app.settings import Settings


class FakeResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return self.payload


def test_ollama_status_counts_models(monkeypatch) -> None:
    def fake_get(url: str, timeout: float) -> FakeResponse:
        assert url == "http://localhost:11434/api/tags"
        assert timeout == 2.0
        return FakeResponse({"models": [{"name": "llama3.2"}, {"name": "qwen2.5"}]})

    monkeypatch.setattr(ollama_client.httpx, "get", fake_get)

    status = ollama_client.get_ollama_status(Settings(_env_file=None))

    assert status.available is True
    assert status.model_count == 2


def test_ollama_models_returns_empty_list_on_error(monkeypatch) -> None:
    def fake_get(url: str, timeout: float) -> FakeResponse:
        raise RuntimeError("offline")

    monkeypatch.setattr(ollama_client.httpx, "get", fake_get)

    assert ollama_client.get_ollama_models(Settings(_env_file=None)) == []
