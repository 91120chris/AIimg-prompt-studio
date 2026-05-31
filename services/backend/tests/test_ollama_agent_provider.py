import json

from app.providers.ollama import ollama_agent_provider
from app.providers.ollama.ollama_agent_provider import (
    OllamaAgentProviderError,
    OllamaAgentRunner,
    default_ollama_generate,
)
from app.schemas.errors import StructuredError
from app.settings import Settings


def test_ollama_agent_runner_returns_valid_agent_response(monkeypatch) -> None:
    monkeypatch.setattr(
        ollama_agent_provider,
        "get_ollama_models",
        lambda settings: ["llama3:latest"],
    )
    calls = []

    def fake_generator(settings, model, prompt, schema):
        calls.append({"model": model, "prompt": prompt, "schema": schema})
        return json.dumps({"kind": "message", "message": "ok"})

    runner = OllamaAgentRunner(Settings(_env_file=None), generator=fake_generator)

    response = runner.run("請回覆 JSON。", model="llama3:latest")

    assert response.kind == "message"
    assert calls[0]["model"] == "llama3:latest"
    assert calls[0]["schema"]["title"] == "AgentTurnResponse"


def test_ollama_agent_runner_repairs_invalid_json(monkeypatch) -> None:
    monkeypatch.setattr(
        ollama_agent_provider,
        "get_ollama_models",
        lambda settings: ["llama3:latest"],
    )
    outputs = [
        "not json",
        json.dumps({"kind": "message", "message": "repaired"}),
    ]
    prompts = []

    def fake_generator(settings, model, prompt, schema):
        prompts.append(prompt)
        return outputs.pop(0)

    runner = OllamaAgentRunner(Settings(_env_file=None), generator=fake_generator)

    response = runner.run("請回覆 JSON。", model="llama3:latest")

    assert response.kind == "message"
    assert response.message == "repaired"
    assert len(prompts) == 2
    assert "not json" in prompts[1]


def test_ollama_agent_runner_returns_error_without_models(monkeypatch) -> None:
    monkeypatch.setattr(ollama_agent_provider, "get_ollama_models", lambda settings: [])

    runner = OllamaAgentRunner(Settings(_env_file=None), generator=lambda *args: "{}")

    response = runner.run("prompt")

    assert response.kind == "error"
    assert response.error.code == "ollama_model_unavailable"


def test_default_ollama_generate_posts_schema(monkeypatch) -> None:
    captured = {}

    class FakeResponse:
        text = ""

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {"response": json.dumps({"kind": "message", "message": "ok"})}

    def fake_post(url, json, timeout):
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(ollama_agent_provider.httpx, "post", fake_post)
    settings = Settings(ollama_agent_temperature=0.1, ollama_timeout_seconds=123, _env_file=None)

    raw_output = default_ollama_generate(
        settings,
        "llama3:latest",
        "prompt",
        {"title": "AgentTurnResponse", "type": "object"},
    )

    assert json.loads(raw_output)["kind"] == "message"
    assert captured["url"] == "http://localhost:11434/api/generate"
    assert captured["timeout"] == 123
    assert captured["json"]["model"] == "llama3:latest"
    assert captured["json"]["stream"] is False
    assert captured["json"]["format"]["title"] == "AgentTurnResponse"
    assert captured["json"]["options"]["temperature"] == 0.1


def test_default_ollama_generate_surfaces_ollama_error(monkeypatch) -> None:
    class FakeResponse:
        text = ""

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {"error": "model not found"}

    monkeypatch.setattr(
        ollama_agent_provider.httpx,
        "post",
        lambda url, json, timeout: FakeResponse(),
    )

    try:
        default_ollama_generate(
            Settings(_env_file=None),
            "missing",
            "prompt",
            {"title": "AgentTurnResponse", "type": "object"},
        )
    except OllamaAgentProviderError as error:
        structured = error.error
    else:
        structured = StructuredError(code="missing_error", message="missing")

    assert structured.code == "ollama_generate_failed"
    assert structured.suggestion == "model not found"
