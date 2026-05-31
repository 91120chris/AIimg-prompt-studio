import httpx

from app.schemas.provider import OllamaStatusResponse
from app.settings import Settings


def _tags_url(settings: Settings) -> str:
    return f"{settings.ollama_base_url.rstrip('/')}/api/tags"


def _read_ollama_model_names(payload: dict[str, object]) -> list[str]:
    models = payload.get("models")
    if not isinstance(models, list):
        return []

    names: list[str] = []
    for item in models:
        if isinstance(item, dict):
            name = item.get("name")
            if isinstance(name, str) and name:
                names.append(name)
    return names


def get_ollama_status(settings: Settings) -> OllamaStatusResponse:
    try:
        response = httpx.get(_tags_url(settings), timeout=2.0)
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        return OllamaStatusResponse(
            provider="ollama_local_llm",
            available=False,
            base_url=settings.ollama_base_url,
            model_count=0,
            error=str(exc),
        )

    if not isinstance(payload, dict):
        return OllamaStatusResponse(
            provider="ollama_local_llm",
            available=False,
            base_url=settings.ollama_base_url,
            model_count=0,
            error="Ollama returned an unexpected response shape.",
        )

    return OllamaStatusResponse(
        provider="ollama_local_llm",
        available=True,
        base_url=settings.ollama_base_url,
        model_count=len(_read_ollama_model_names(payload)),
        error=None,
    )


def get_ollama_models(settings: Settings) -> list[str]:
    try:
        response = httpx.get(_tags_url(settings), timeout=2.0)
        response.raise_for_status()
        payload = response.json()
    except Exception:
        return []

    if not isinstance(payload, dict):
        return []
    return _read_ollama_model_names(payload)
