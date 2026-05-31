from pathlib import Path

from app.providers.codex.codex_binary_resolver import ResolvedCodexBinary


def build_codex_exec_command(
    binary: ResolvedCodexBinary,
    prompt: str,
    *,
    model: str | None = None,
    sandbox: str | None = None,
    skip_git_repo_check: bool = False,
    output_schema_path: str | Path | None = None,
) -> list[str]:
    command = [
        *binary.command_prefix,
        "--ask-for-approval",
        "never",
    ]
    if model and model != "auto":
        command.extend(["--model", model])
    if sandbox:
        command.extend(["--sandbox", sandbox])
    command.append("exec")
    if skip_git_repo_check:
        command.append("--skip-git-repo-check")
    if output_schema_path is not None:
        command.extend(["--output-schema", str(output_schema_path)])
    command.append(prompt)
    return command


def build_codex_image_exec_command(
    binary: ResolvedCodexBinary,
    prompt: str,
    *,
    model: str | None = None,
    sandbox: str | None = None,
    skip_git_repo_check: bool = False,
    output_schema_path: str | Path | None = None,
    reference_image_paths: list[str | Path] | None = None,
) -> list[str]:
    command = build_codex_exec_command(
        binary,
        prompt,
        model=model,
        sandbox=sandbox,
        skip_git_repo_check=skip_git_repo_check,
        output_schema_path=output_schema_path,
    )
    prompt_arg = command.pop()
    for image_path in reference_image_paths or []:
        command.extend(["--image", str(image_path)])
    command.append(prompt_arg)
    return command
