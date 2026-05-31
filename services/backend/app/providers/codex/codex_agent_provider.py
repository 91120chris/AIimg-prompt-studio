import json
import subprocess
from collections.abc import Callable
from dataclasses import dataclass

from pydantic import TypeAdapter, ValidationError

from app.core.json_schema import schema_output_dir
from app.core.prompt_compiler import build_repair_prompt
from app.providers.codex.codex_binary_resolver import resolve_codex_binary
from app.providers.codex.codex_command_builder import build_codex_exec_command
from app.schemas.agent import AgentTurnResponse, ErrorTurnResponse
from app.schemas.errors import StructuredError
from app.settings import Settings


AgentTurnAdapter = TypeAdapter(AgentTurnResponse)
CommandExecutor = Callable[[list[str], int, str | None], str]


def default_command_executor(
    command: list[str],
    timeout_seconds: int,
    input_text: str | None = None,
) -> str:
    completed = subprocess.run(
        command,
        capture_output=True,
        check=False,
        input=input_text,
        text=True,
        timeout=timeout_seconds,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0:
        stderr_tail = completed.stderr.strip()[-700:]
        raise CodexAgentProviderError(
            StructuredError(
                code="codex_exec_failed",
                message="Codex CLI 執行失敗。",
                suggestion=stderr_tail or "請確認 Codex CLI 已登入，且目前模型可用。",
            )
        )
    return completed.stdout


class CodexAgentProviderError(RuntimeError):
    def __init__(self, error: StructuredError) -> None:
        self.error = error
        super().__init__(error.message)


@dataclass
class CodexAgentRunner:
    settings: Settings
    executor: CommandExecutor = default_command_executor

    def run(self, prompt: str, *, model: str | None = None) -> AgentTurnResponse:
        binary = resolve_codex_binary(self.settings.codex_binary_path)
        if not binary.available:
            return _error_response(
                "codex_unavailable",
                "找不到 Codex CLI。",
                binary.warning or "請確認 codex 已安裝並在 PATH 上，或設定 CODEX_BINARY_PATH。",
            )

        selected_model = model or self.settings.codex_default_model
        schema_path = schema_output_dir() / "agent_turn_response.schema.json"

        try:
            raw_output = self.executor(
                build_codex_exec_command(
                    binary,
                    "-",
                    model=selected_model,
                    sandbox="read-only",
                    output_schema_path=schema_path,
                ),
                self.settings.codex_timeout_seconds,
                prompt,
            )
            return parse_agent_turn_response(raw_output)
        except (ValidationError, ValueError) as first_error:
            repair_prompt = build_repair_prompt(raw_output, str(first_error))
            try:
                repair_output = self.executor(
                    build_codex_exec_command(
                        binary,
                        "-",
                        model=selected_model,
                        sandbox="read-only",
                        output_schema_path=schema_path,
                    ),
                    self.settings.codex_timeout_seconds,
                    repair_prompt,
                )
                return parse_agent_turn_response(repair_output)
            except (
                CodexAgentProviderError,
                ValidationError,
                ValueError,
                subprocess.TimeoutExpired,
            ) as repair_error:
                return _error_response(
                    "codex_schema_validation_failed",
                    "Codex 回覆無法通過 strict schema 驗證。",
                    str(repair_error)[:700],
                )
        except subprocess.TimeoutExpired:
            return _error_response(
                "codex_timeout",
                "Codex CLI 執行逾時。",
                "可以調高 CODEX_TIMEOUT_SECONDS，或縮短 prompt 後再試一次。",
            )
        except CodexAgentProviderError as error:
            return ErrorTurnResponse(kind="error", error=error.error)


def parse_agent_turn_response(raw_output: str) -> AgentTurnResponse:
    payload = _extract_json_object(raw_output)
    payload = normalize_codex_agent_turn_payload(payload)
    return AgentTurnAdapter.validate_python(payload)


def normalize_codex_agent_turn_payload(payload: object) -> object:
    if not isinstance(payload, dict):
        return payload

    kind = payload.get("kind")
    if kind == "message":
        return _drop_none_fields(payload, {"kind", "message", "warnings"})
    if kind == "questionnaire":
        normalized = _drop_none_fields(payload, {"kind", "message", "questionnaire", "warnings"})
        questionnaire = normalized.get("questionnaire")
        if isinstance(questionnaire, dict):
            normalized["questionnaire"] = normalize_questionnaire_payload(questionnaire)
        return normalized
    if kind == "optimized_prompt":
        return _drop_none_fields(
            payload,
            {"kind", "message", "optimized_prompt", "prompt_version_title", "warnings"},
        )
    if kind == "error":
        return _drop_none_fields(payload, {"kind", "error"})
    return payload


def normalize_questionnaire_payload(payload: dict[str, object]) -> dict[str, object]:
    normalized = dict(payload)
    questions = normalized.get("questions")
    if isinstance(questions, list):
        normalized["questions"] = [normalize_question_payload(question) for question in questions]
    return _drop_none_fields(normalized, {"questionnaire_id", "title", "description", "questions"})


def normalize_question_payload(payload: object) -> object:
    if not isinstance(payload, dict):
        return payload

    kind = payload.get("kind")
    if kind == "text":
        return _drop_none_fields(
            payload,
            {"kind", "question_id", "label", "prompt", "required", "placeholder", "max_length"},
        )
    if kind == "choice":
        normalized = _drop_none_fields(
            payload,
            {"kind", "question_id", "label", "prompt", "required", "options", "allow_multiple"},
        )
        if "allow_multiple" not in normalized:
            normalized["allow_multiple"] = False
        return normalized
    if kind == "boolean":
        normalized = _drop_none_fields(
            payload,
            {"kind", "question_id", "label", "prompt", "required", "true_label", "false_label"},
        )
        normalized.setdefault("true_label", "是")
        normalized.setdefault("false_label", "否")
        return normalized
    if kind == "scale":
        normalized = _drop_none_fields(
            payload,
            {"kind", "question_id", "label", "prompt", "required", "min_value", "max_value", "step"},
        )
        normalized.setdefault("step", 1)
        return normalized
    return payload


def _drop_none_fields(payload: dict[str, object], allowed_keys: set[str]) -> dict[str, object]:
    return {
        key: value
        for key, value in payload.items()
        if key in allowed_keys and value is not None
    }


def _extract_json_object(raw_output: str) -> object:
    stripped = raw_output.strip()
    if not stripped:
        raise ValueError("Codex returned empty output.")

    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    if stripped.startswith("```"):
        unfenced = stripped.strip("`").removeprefix("json").strip()
        try:
            return json.loads(unfenced)
        except json.JSONDecodeError:
            pass

    start = stripped.find("{")
    if start < 0:
        raise ValueError("Codex output did not contain a JSON object.")

    depth = 0
    in_string = False
    escaped = False
    for index, char in enumerate(stripped[start:], start=start):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return json.loads(stripped[start : index + 1])

    raise ValueError("Codex output JSON object was incomplete.")


def _error_response(code: str, message: str, suggestion: str | None = None) -> ErrorTurnResponse:
    return ErrorTurnResponse(
        kind="error",
        error=StructuredError(code=code, message=message, suggestion=suggestion),
    )
