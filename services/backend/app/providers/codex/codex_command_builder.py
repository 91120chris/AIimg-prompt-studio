from pathlib import Path
from typing import TypeAlias

from app.providers.codex.codex_binary_resolver import ResolvedCodexBinary

CodexConfigValue: TypeAlias = str | bool | int | float


def build_codex_exec_command(
    binary: ResolvedCodexBinary,
    prompt: str,
    *,
    model: str | None = None,
    sandbox: str | None = None,
    skip_git_repo_check: bool = False,
    output_schema_path: str | Path | None = None,
    config_overrides: dict[str, CodexConfigValue] | None = None,
    json_output: bool = False,
) -> list[str]:
    command = _base_codex_exec_command(
        binary,
        model=model,
        sandbox=sandbox,
        config_overrides=config_overrides,
    )
    command.append("exec")
    if skip_git_repo_check:
        command.append("--skip-git-repo-check")
    if output_schema_path is not None:
        command.extend(["--output-schema", str(output_schema_path)])
    if json_output:
        command.append("--json")
    command.append(prompt)
    return command


def build_codex_exec_resume_command(
    binary: ResolvedCodexBinary,
    session_id: str,
    prompt: str,
    *,
    model: str | None = None,
    sandbox: str | None = None,
    skip_git_repo_check: bool = False,
    output_schema_path: str | Path | None = None,
    config_overrides: dict[str, CodexConfigValue] | None = None,
    json_output: bool = False,
) -> list[str]:
    command = _base_codex_exec_command(
        binary,
        model=model,
        sandbox=sandbox,
        config_overrides=config_overrides,
    )
    command.extend(["exec", "resume"])
    if skip_git_repo_check:
        command.append("--skip-git-repo-check")
    if output_schema_path is not None:
        command.extend(["--output-schema", str(output_schema_path)])
    if json_output:
        command.append("--json")
    command.extend([session_id, prompt])
    return command


def _base_codex_exec_command(
    binary: ResolvedCodexBinary,
    *,
    model: str | None = None,
    sandbox: str | None = None,
    config_overrides: dict[str, CodexConfigValue] | None = None,
) -> list[str]:
    command = [
        *binary.command_prefix,
        "--ask-for-approval",
        "never",
    ]
    for key, value in (config_overrides or {}).items():
        command.extend(["--config", f"{key}={_toml_value(value)}"])
    if model and model != "auto":
        command.extend(["--model", model])
    if sandbox:
        command.extend(["--sandbox", sandbox])
    return command


def _toml_value(value: CodexConfigValue) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return str(value)
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def build_codex_image_exec_command(
    binary: ResolvedCodexBinary,
    prompt: str,
    *,
    model: str | None = None,
    sandbox: str | None = None,
    skip_git_repo_check: bool = False,
    output_schema_path: str | Path | None = None,
    reference_image_paths: list[str | Path] | None = None,
    config_overrides: dict[str, CodexConfigValue] | None = None,
) -> list[str]:
    command = build_codex_exec_command(
        binary,
        prompt,
        model=model,
        sandbox=sandbox,
        skip_git_repo_check=skip_git_repo_check,
        output_schema_path=output_schema_path,
        config_overrides=config_overrides,
    )
    prompt_arg = command.pop()
    for image_path in reference_image_paths or []:
        command.extend(["--image", str(image_path)])
    command.append(prompt_arg)
    return command
