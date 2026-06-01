from collections.abc import Callable
from dataclasses import dataclass

import httpx
from pydantic import ValidationError

from app.core.agent_fallbacks import fallback_text_questionnaire
from app.core.json_schema import codex_agent_turn_response_schema
from app.core.prompt_compiler import build_repair_prompt
from app.providers.codex.codex_agent_provider import parse_agent_turn_response
from app.providers.ollama.ollama_client import get_ollama_models
from app.schemas.agent import AgentTurnResponse, ErrorTurnResponse
from app.schemas.errors import StructuredError
from app.settings import Settings

OllamaGenerateExecutor = Callable[[Settings, str, str, dict[str, object]], str]


def _generate_url(settings: Settings) -> str:
    return f"{settings.ollama_base_url.rstrip('/')}/api/generate"


def default_ollama_generate(
    settings: Settings,
    model: str,
    prompt: str,
    schema: dict[str, object],
) -> str:
    request_payload: dict[str, object] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": schema,
        "options": {
            "temperature": settings.ollama_agent_temperature,
        },
    }

    try:
        response = httpx.post(
            _generate_url(settings),
            json=request_payload,
            timeout=settings.ollama_timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
    except httpx.HTTPStatusError as exc:
        response_text = exc.response.text.strip() if exc.response is not None else str(exc)
        raise OllamaAgentProviderError(
            StructuredError(
                code="ollama_generate_failed",
                message="Ollama 執行失敗。",
                suggestion=response_text[-700:] or "請確認 Ollama 服務與模型可正常執行。",
            )
        ) from exc
    except (httpx.HTTPError, ValueError) as exc:
        raise OllamaAgentProviderError(
            StructuredError(
                code="ollama_generate_failed",
                message="Ollama 執行失敗。",
                suggestion=str(exc)[:700],
            )
        ) from exc

    if not isinstance(payload, dict):
        raise OllamaAgentProviderError(
            StructuredError(
                code="ollama_bad_response",
                message="Ollama 回傳格式不符合預期。",
                suggestion="請確認 Ollama API 回傳 JSON 物件。",
            )
        )

    error = payload.get("error")
    if isinstance(error, str) and error.strip():
        raise OllamaAgentProviderError(
            StructuredError(
                code="ollama_generate_failed",
                message="Ollama 執行失敗。",
                suggestion=error.strip()[:700],
            )
        )

    raw_output = payload.get("response")
    if not isinstance(raw_output, str) or not raw_output.strip():
        raise OllamaAgentProviderError(
            StructuredError(
                code="ollama_empty_response",
                message="Ollama 沒有回傳可解析的內容。",
                suggestion="請換一個本機模型，或降低 prompt 複雜度後重試。",
            )
        )
    return raw_output


class OllamaAgentProviderError(RuntimeError):
    def __init__(self, error: StructuredError) -> None:
        self.error = error
        super().__init__(error.message)


@dataclass
class OllamaAgentRunner:
    settings: Settings
    generator: OllamaGenerateExecutor = default_ollama_generate

    def run(self, prompt: str, *, model: str | None = None) -> AgentTurnResponse:
        try:
            selected_model = self._select_model(model)
        except OllamaAgentProviderError as error:
            return ErrorTurnResponse(kind="error", error=error.error)

        schema = codex_agent_turn_response_schema()
        raw_output = ""
        try:
            raw_output = self.generator(self.settings, selected_model, prompt, schema)
            return parse_agent_turn_response(raw_output)
        except (ValidationError, ValueError) as first_error:
            repair_prompt = build_repair_prompt(raw_output, str(first_error))
            try:
                repair_output = self.generator(self.settings, selected_model, repair_prompt, schema)
                return parse_agent_turn_response(repair_output)
            except (
                OllamaAgentProviderError,
                ValidationError,
                ValueError,
            ) as repair_error:
                return fallback_text_questionnaire(
                    "Ollama",
                    str(repair_error),
                )
        except OllamaAgentProviderError as error:
            return ErrorTurnResponse(kind="error", error=error.error)

    def _select_model(self, requested_model: str | None) -> str:
        models = get_ollama_models(self.settings)
        selected_model = (requested_model or self.settings.ollama_selected_model or "").strip()
        if selected_model:
            if selected_model not in models:
                raise OllamaAgentProviderError(
                    StructuredError(
                        code="unknown_ollama_model",
                        message="選取的 Ollama 模型不在目前本機模型清單中。",
                        suggestion="請重新整理 Provider 狀態，或先用 ollama pull 安裝該模型。",
                    )
                )
            return selected_model

        if models:
            return models[0]

        raise OllamaAgentProviderError(
            StructuredError(
                code="ollama_model_unavailable",
                message="找不到可用的 Ollama 模型。",
                suggestion="請先啟動 Ollama，並用 ollama pull 安裝至少一個文字模型。",
            )
        )


def _error_response(code: str, message: str, suggestion: str | None = None) -> ErrorTurnResponse:
    return ErrorTurnResponse(
        kind="error",
        error=StructuredError(code=code, message=message, suggestion=suggestion),
    )
