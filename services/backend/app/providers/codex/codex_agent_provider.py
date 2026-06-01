import json
import re
import subprocess
from collections.abc import Callable
from dataclasses import dataclass

from pydantic import TypeAdapter, ValidationError

from app.core.agent_fallbacks import fallback_text_questionnaire
from app.core.json_schema import schema_output_dir
from app.core.prompt_compiler import build_repair_prompt
from app.core.session_workspace import new_id
from app.providers.codex.codex_binary_resolver import resolve_codex_binary
from app.providers.codex.codex_command_builder import build_codex_exec_command
from app.schemas.agent import AgentTurnResponse, ErrorTurnResponse
from app.schemas.errors import StructuredError
from app.schemas.provider import CodexReasoningEffort, CodexReasoningSummary, CodexVerbosity
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

    def run(
        self,
        prompt: str,
        *,
        model: str | None = None,
        reasoning_effort: CodexReasoningEffort | None = None,
        reasoning_summary: CodexReasoningSummary | None = None,
        verbosity: CodexVerbosity | None = None,
    ) -> AgentTurnResponse:
        binary = resolve_codex_binary(self.settings.codex_binary_path)
        if not binary.available:
            return _error_response(
                "codex_unavailable",
                "找不到 Codex CLI。",
                binary.warning or "請確認 codex 已安裝並在 PATH 上，或設定 CODEX_BINARY_PATH。",
            )

        selected_model = model or self.settings.codex_default_model
        config_overrides = codex_config_overrides(
            self.settings,
            reasoning_effort=reasoning_effort,
            reasoning_summary=reasoning_summary,
            verbosity=verbosity,
        )
        schema_path = schema_output_dir() / "agent_turn_response.schema.json"

        try:
            raw_output = self.executor(
                build_codex_exec_command(
                    binary,
                    "-",
                    model=selected_model,
                    sandbox="read-only",
                    output_schema_path=schema_path,
                    config_overrides=config_overrides,
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
                        config_overrides=config_overrides,
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
                return fallback_text_questionnaire(
                    "Codex",
                    str(repair_error),
                )
        except subprocess.TimeoutExpired:
            return _error_response(
                "codex_timeout",
                "Codex CLI 執行逾時。",
                "可以調高 CODEX_TIMEOUT_SECONDS，或縮短 prompt 後再試一次。",
            )
        except CodexAgentProviderError as error:
            return ErrorTurnResponse(kind="error", error=error.error)


def codex_config_overrides(
    settings: Settings,
    *,
    reasoning_effort: CodexReasoningEffort | None = None,
    reasoning_summary: CodexReasoningSummary | None = None,
    verbosity: CodexVerbosity | None = None,
) -> dict[str, str]:
    overrides = {
        "model_reasoning_effort": reasoning_effort or settings.codex_default_reasoning_effort,
        "model_reasoning_summary": reasoning_summary or settings.codex_default_reasoning_summary,
    }
    selected_verbosity = verbosity or settings.codex_default_verbosity
    if selected_verbosity:
        overrides["model_verbosity"] = selected_verbosity
    return overrides


def parse_agent_turn_response(raw_output: str) -> AgentTurnResponse:
    payload = _extract_json_object(raw_output)
    payload = coerce_agent_turn_payload(payload)
    return AgentTurnAdapter.validate_python(payload)


def coerce_agent_turn_payload(payload: object) -> object:
    if isinstance(payload, list):
        return _questionnaire_turn_from_payload({"questions": payload})
    if not isinstance(payload, dict):
        return payload

    if _looks_like_questionnaire(payload) and _get_value(payload, ("kind", "questionnaire")) is None:
        return _questionnaire_turn_from_payload(payload)

    kind = _canonical_turn_kind(_get_value(payload, ("kind", "type")))
    if kind is None:
        if _get_value(payload, ("questionnaire", "questions")) is not None:
            kind = "questionnaire"
        elif _first_string(
            payload,
            (
                "optimized_prompt",
                "optimizedPrompt",
                "optimized_prompt_text",
                "final_prompt",
                "prompt",
                "result",
            ),
        ):
            kind = "optimized_prompt"
        elif _get_value(payload, ("error",)) is not None:
            kind = "error"
        elif _first_string(payload, ("message", "content", "text", "response")):
            kind = "message"

    if kind == "message":
        return {
            "kind": "message",
            "message": _first_string(payload, ("message", "content", "text", "response")) or "",
            "warnings": _warnings_list(_get_value(payload, ("warnings", "warning"))),
        }
    if kind == "questionnaire":
        questionnaire = _get_value(payload, ("questionnaire",))
        if not isinstance(questionnaire, dict):
            questionnaire = payload
        return _questionnaire_turn_from_payload(
            questionnaire,
            message=_first_string(payload, ("message", "content", "text", "response")) or "",
            warnings=_warnings_list(_get_value(payload, ("warnings", "warning"))),
        )
    if kind == "optimized_prompt":
        return _drop_none_fields(
            {
                "kind": "optimized_prompt",
                "message": _first_string(payload, ("message", "content", "text", "response")) or "",
                "optimized_prompt": _first_string(
                    payload,
                    (
                        "optimized_prompt",
                        "optimizedPrompt",
                        "optimized_prompt_text",
                        "final_prompt",
                        "prompt",
                        "result",
                    ),
                )
                or "",
                "prompt_version_title": _first_string(
                    payload,
                    ("prompt_version_title", "version_title", "title", "name"),
                ),
                "warnings": _warnings_list(_get_value(payload, ("warnings", "warning"))),
            },
            {"kind", "message", "optimized_prompt", "prompt_version_title", "warnings"},
        )
    if kind == "error":
        error = _get_value(payload, ("error",))
        if isinstance(error, str):
            error = {"code": "agent_error", "message": error}
        if isinstance(error, dict):
            error = {
                "code": _first_string(error, ("code", "type")) or "agent_error",
                "message": _first_string(error, ("message", "detail", "text")) or "Agent failed.",
                "suggestion": _first_string(error, ("suggestion", "hint", "fix")),
            }
        return _drop_none_fields({"kind": "error", "error": error}, {"kind", "error"})

    return normalize_codex_agent_turn_payload(payload)


def _questionnaire_turn_from_payload(
    payload: dict[str, object],
    *,
    message: str = "",
    warnings: list[str] | None = None,
) -> dict[str, object]:
    return {
        "kind": "questionnaire",
        "message": message,
        "questionnaire": coerce_questionnaire_payload(payload),
        "warnings": warnings or [],
    }


def coerce_questionnaire_payload(payload: dict[str, object]) -> dict[str, object]:
    questions_value = _get_value(payload, ("questions", "items", "fields"))
    questions = questions_value if isinstance(questions_value, list) else []
    if not questions and _looks_like_question(payload):
        questions = [payload]

    normalized_questions = [
        coerce_question_payload(question, index=index)
        for index, question in enumerate(questions)
    ]
    if not normalized_questions:
        normalized_questions = [
            coerce_question_payload(
                {
                    "kind": "text",
                    "question_id": "manual_details",
                    "label": "Manual details",
                    "prompt": "Please describe what should be adjusted.",
                },
                index=0,
            )
        ]

    return _drop_none_fields(
        {
            "questionnaire_id": _first_string(payload, ("questionnaire_id", "id", "key"))
            or new_id("q"),
            "title": _first_string(payload, ("title", "name", "label")) or "補充問卷",
            "description": _first_string(payload, ("description", "intro", "summary")),
            "questions": normalized_questions,
        },
        {"questionnaire_id", "title", "description", "questions"},
    )


def coerce_question_payload(payload: object, *, index: int) -> dict[str, object]:
    source = payload if isinstance(payload, dict) else {"prompt": _string_value(payload)}
    prompt = _first_string(source, ("prompt", "question", "text", "description", "message"))
    label = _first_string(source, ("label", "title", "name")) or prompt
    question_id = _first_string(source, ("question_id", "id", "key", "field", "name"))
    raw_kind = _get_value(source, ("kind", "type", "input_type", "inputType", "question_type"))
    options_value = _get_value(source, ("options", "choices", "items", "values"))
    kind = _canonical_question_kind(raw_kind, has_options=options_value is not None)

    if kind is None:
        if options_value is not None:
            kind = "choice"
        elif _get_value(source, ("min_value", "min", "minimum")) is not None or _get_value(
            source,
            ("max_value", "max", "maximum"),
        ) is not None:
            kind = "scale"
        else:
            kind = "text"

    base = {
        "kind": kind,
        "question_id": question_id or _question_id_from_text(label or prompt, index),
        "label": label or f"Question {index + 1}",
        "prompt": prompt or label or f"Question {index + 1}",
        "required": _bool_value(_get_value(source, ("required", "is_required")), default=True),
    }

    if kind == "choice":
        options = _normalize_choice_options(options_value)
        if not options:
            return _text_question_from_base(source, base)
        return {
            **base,
            "options": options,
            "allow_multiple": _bool_value(
                _get_value(source, ("allow_multiple", "multiple", "multi_select", "multiSelect")),
                default=_is_multi_choice_kind(raw_kind),
            ),
        }
    if kind == "boolean":
        return {
            **base,
            "true_label": _first_string(source, ("true_label", "trueLabel", "yes_label")) or "是",
            "false_label": _first_string(source, ("false_label", "falseLabel", "no_label")) or "否",
        }
    if kind == "scale":
        min_value = _int_value(_get_value(source, ("min_value", "min", "minimum")), default=1)
        max_value = _int_value(_get_value(source, ("max_value", "max", "maximum")), default=5)
        if max_value <= min_value:
            max_value = min_value + 1
        step = max(_int_value(_get_value(source, ("step", "increment")), default=1), 1)
        return {**base, "min_value": min_value, "max_value": max_value, "step": step}
    return _text_question_from_base(source, base)


def _text_question_from_base(
    source: dict[str, object],
    base: dict[str, object],
) -> dict[str, object]:
    return _drop_none_fields(
        {
            **base,
            "kind": "text",
            "placeholder": _first_string(source, ("placeholder", "hint", "example")),
            "max_length": _int_value(_get_value(source, ("max_length", "maxLength")), default=None),
        },
        {"kind", "question_id", "label", "prompt", "required", "placeholder", "max_length"},
    )


def _normalize_choice_options(value: object) -> list[dict[str, object]]:
    if isinstance(value, str):
        raw_options: list[object] = [
            option.strip()
            for option in re.split(r"[\n,;]+", value)
            if option.strip()
        ]
    elif isinstance(value, dict):
        raw_options = [
            {"value": key, "label": option}
            for key, option in value.items()
        ]
    elif isinstance(value, list):
        raw_options = value
    else:
        raw_options = []

    normalized: list[dict[str, object]] = []
    used_values: set[str] = set()
    for index, option in enumerate(raw_options):
        if isinstance(option, dict):
            option_value = _first_string(option, ("value", "id", "key", "label", "text", "name"))
            option_label = _first_string(option, ("label", "text", "title", "name", "value"))
            description = _first_string(option, ("description", "detail", "hint"))
        else:
            option_value = _string_value(option)
            option_label = option_value
            description = None

        option_value = option_value or f"option_{index + 1}"
        option_label = option_label or option_value
        if option_value in used_values:
            option_value = f"{option_value}_{index + 1}"
        used_values.add(option_value)
        normalized.append(
            _drop_none_fields(
                {
                    "value": option_value,
                    "label": option_label,
                    "description": description,
                },
                {"value", "label", "description"},
            )
        )
    return normalized


def _canonical_turn_kind(value: object) -> str | None:
    kind = _normalized_key(_string_value(value))
    if kind in {"message", "chatmessage", "assistantmessage"}:
        return "message"
    if kind in {"questionnaire", "questions", "feedbackquestionnaire", "questionnaireturn"}:
        return "questionnaire"
    if kind in {"optimizedprompt", "optimized", "promptoptimization", "prompt"}:
        return "optimized_prompt"
    if kind in {"error", "failure"}:
        return "error"
    return None


def _canonical_question_kind(value: object, *, has_options: bool) -> str | None:
    kind = _normalized_key(_string_value(value))
    if kind in {"text", "textarea", "freetext", "shorttext", "longtext", "string", "paragraph"}:
        return "text"
    if kind in {"choice", "select", "dropdown", "radio", "singlechoice", "multichoice", "multiplechoice", "multiselect"}:
        return "choice"
    if kind in {"checkbox", "checkboxes"}:
        return "choice" if has_options else "boolean"
    if kind in {"boolean", "bool", "yesno", "toggle", "confirm"}:
        return "boolean"
    if kind in {"scale", "rating", "slider", "number", "numeric"}:
        return "scale"
    return None


def _is_multi_choice_kind(value: object) -> bool:
    return _normalized_key(_string_value(value)) in {
        "checkboxes",
        "multichoice",
        "multiplechoice",
        "multiselect",
    }


def _looks_like_questionnaire(payload: dict[str, object]) -> bool:
    return _get_value(payload, ("questions", "items", "fields")) is not None and (
        _first_string(payload, ("title", "name", "questionnaire_id", "id")) is not None
        or _get_value(payload, ("questions", "items", "fields")) is not None
    )


def _looks_like_question(payload: dict[str, object]) -> bool:
    return (
        _first_string(payload, ("prompt", "question", "text", "label", "title")) is not None
        or _get_value(payload, ("options", "choices")) is not None
    )


def _get_value(payload: dict[str, object], keys: tuple[str, ...]) -> object:
    wanted = {_normalized_key(key) for key in keys}
    for key, value in payload.items():
        if _normalized_key(key) in wanted:
            return value
    return None


def _first_string(payload: dict[str, object], keys: tuple[str, ...]) -> str | None:
    return _string_value(_get_value(payload, keys))


def _string_value(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return str(value)
    return None


def _bool_value(value: object, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    return default


def _int_value(value: object, *, default: int | None) -> int | None:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return default
    return default


def _warnings_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        warning = value.strip()
        return [warning] if warning else []
    if isinstance(value, list):
        return [warning for item in value if (warning := _string_value(item))]
    return []


def _question_id_from_text(value: str | None, index: int) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", (value or "").lower()).strip("_")
    return slug[:48] or f"question_{index + 1}"


def _normalized_key(value: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", "", (value or "").lower())


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
