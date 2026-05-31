import json
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from pydantic import TypeAdapter, ValidationError
from sqlmodel import Session

from app.core.image_records import register_generated_image
from app.core.json_schema import schema_output_dir
from app.core.session_workspace import ensure_path_inside_session, ensure_session_workspace
from app.db.models import GeneratedImageRecord, GenerationJobRecord, ReferenceImageRecord
from app.providers.codex.codex_agent_provider import _extract_json_object, codex_config_overrides
from app.providers.codex.codex_binary_resolver import resolve_codex_binary
from app.providers.codex.codex_command_builder import build_codex_image_exec_command
from app.schemas.errors import StructuredError
from app.schemas.generation import CodexImageResponse, GenerationConfirmRequest
from app.settings import Settings

CodexImageAdapter = TypeAdapter(CodexImageResponse)
ImageCommandExecutor = Callable[[list[str], int, str, Path], str]
ALLOWED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


class CodexImageProviderError(RuntimeError):
    def __init__(self, error: StructuredError) -> None:
        self.error = error
        super().__init__(error.message)


def default_image_command_executor(
    command: list[str],
    timeout_seconds: int,
    input_text: str,
    cwd: Path,
) -> str:
    completed = subprocess.run(
        command,
        capture_output=True,
        check=False,
        cwd=cwd,
        input=input_text,
        text=True,
        timeout=timeout_seconds,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0:
        stderr_tail = completed.stderr.strip()[-900:]
        raise CodexImageProviderError(
            StructuredError(
                code="codex_image_exec_failed",
                message="Codex CLI 圖像生成執行失敗。",
                suggestion=stderr_tail or "請確認 Codex CLI 可用，且目前帳號/模型支援圖像生成。",
            )
        )
    return completed.stdout


@dataclass
class CodexImageProvider:
    settings: Settings
    executor: ImageCommandExecutor = default_image_command_executor

    def generate(
        self,
        db: Session,
        *,
        job: GenerationJobRecord,
        payload: GenerationConfirmRequest,
        reference_images: list[ReferenceImageRecord],
    ) -> list[GeneratedImageRecord]:
        binary = resolve_codex_binary(self.settings.codex_binary_path)
        if not binary.available:
            raise CodexImageProviderError(
                StructuredError(
                    code="codex_unavailable",
                    message="找不到 Codex CLI。",
                    suggestion=binary.warning or "請確認 codex 已安裝並在 PATH 上。",
                )
            )

        workspace = ensure_session_workspace(self.settings, payload.session_id)
        run_dir = workspace / "generated" / "codex-runs" / job.job_id
        run_dir.mkdir(parents=True, exist_ok=True)
        schema_path = schema_output_dir() / "codex_image_response.schema.json"
        prompt = build_codex_image_prompt(payload)
        reference_paths = [Path(record.storage_path) for record in reference_images]

        try:
            raw_output = self.executor(
                build_codex_image_exec_command(
                    binary,
                    "-",
                    model=payload.codex_model or self.settings.codex_default_model,
                    sandbox="workspace-write",
                    skip_git_repo_check=True,
                    output_schema_path=schema_path,
                    reference_image_paths=reference_paths,
                    config_overrides=codex_config_overrides(
                        self.settings,
                        reasoning_effort=payload.codex_reasoning_effort,
                        reasoning_summary=payload.codex_reasoning_summary,
                        verbosity=payload.codex_verbosity,
                    ),
                ),
                self.settings.codex_timeout_seconds,
                prompt,
                run_dir,
            )
            codex_response = parse_codex_image_response(raw_output)
        except subprocess.TimeoutExpired as error:
            raise CodexImageProviderError(
                StructuredError(
                    code="codex_image_timeout",
                    message="Codex CLI 圖像生成逾時。",
                    suggestion="可以調高 CODEX_TIMEOUT_SECONDS，或先用較簡短的 prompt 測試。",
                )
            ) from error
        except (ValidationError, ValueError) as error:
            raise CodexImageProviderError(
                StructuredError(
                    code="codex_image_schema_validation_failed",
                    message="Codex 圖像生成回覆無法通過 strict schema 驗證。",
                    suggestion=str(error)[:900],
                )
            ) from error

        if codex_response.status == "failed":
            raise CodexImageProviderError(
                codex_response.error
                or StructuredError(
                    code="codex_image_failed",
                    message="Codex 圖像生成失敗。",
                    suggestion="請檢查 prompt 或稍後重試。",
                )
            )

        source_paths = validate_codex_image_files(self.settings, payload.session_id, run_dir, codex_response.image_files)
        if not source_paths:
            raise CodexImageProviderError(
                StructuredError(
                    code="codex_image_no_outputs",
                    message="Codex 沒有回傳任何生成圖片。",
                    suggestion="請重新嘗試生成，或確認 Codex CLI 圖像能力可用。",
                )
            )

        return [
            register_generated_image(
                db,
                self.settings,
                session_id=payload.session_id,
                source_path=source_path,
                provider=payload.provider,
                role="optimized_prompt",
                seed=payload.parameters.seed,
            )
            for source_path in source_paths
        ]


def build_codex_image_prompt(payload: GenerationConfirmRequest) -> str:
    reference_note = (
        "參考圖片已透過 --image 參數附加，請只把它們當作視覺參考，不要把路徑寫進 prompt。"
        if payload.reference_image_ids
        else "沒有參考圖片。"
    )
    return f"""你是 Prompt Optimizer Studio 的 Codex CLI image provider。

任務：使用可用的圖像生成能力，根據最佳化 prompt 生成 1 張圖片。

硬性規則：
- 必須等同於使用者已按下確認生成；現在可以生成圖片。
- 只能在目前工作目錄寫入輸出圖片，不要寫到其他資料夾。
- 輸出圖片檔名請使用 output.png、output.jpg 或 output.webp。
- 不要把本機檔案路徑寫進文字 prompt。
- 完成後只回傳符合 output schema 的 JSON，不要加 Markdown。
- 成功時 status="succeeded"，image_files 填相對檔名陣列，例如 ["output.png"]，error 填 null。
- 失敗時 status="failed"，image_files 填空陣列，error 填 code/message/suggestion。

模式：{payload.mode}
參考圖片：{reference_note}
參數：
{json.dumps(payload.parameters.model_dump(), ensure_ascii=False, sort_keys=True)}

原始 prompt：
{payload.original_prompt}

最佳化 prompt：
{payload.optimized_prompt}
"""


def parse_codex_image_response(raw_output: str) -> CodexImageResponse:
    return CodexImageAdapter.validate_python(_extract_json_object(raw_output))


def validate_codex_image_files(
    settings: Settings,
    session_id: str,
    run_dir: Path,
    image_files: list[str],
) -> list[Path]:
    valid_paths: list[Path] = []
    resolved_run_dir = run_dir.resolve()
    for image_file in image_files:
        candidate = (run_dir / image_file).resolve()
        if candidate != resolved_run_dir and resolved_run_dir not in candidate.parents:
            raise ValueError("Codex returned an output path outside the generation run directory.")
        session_path = ensure_path_inside_session(settings, session_id, candidate)
        if session_path.suffix.lower() not in ALLOWED_IMAGE_SUFFIXES:
            raise ValueError(f"Unsupported generated image suffix: {session_path.suffix}")
        if not session_path.exists() or not session_path.is_file():
            raise ValueError(f"Generated image file does not exist: {image_file}")
        valid_paths.append(session_path)
    return valid_paths
