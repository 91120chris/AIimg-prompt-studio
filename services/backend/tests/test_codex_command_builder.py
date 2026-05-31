from app.providers.codex.codex_binary_resolver import ResolvedCodexBinary
from app.providers.codex.codex_command_builder import (
    build_codex_exec_command,
    build_codex_image_exec_command,
)


def fake_binary() -> ResolvedCodexBinary:
    return ResolvedCodexBinary(
        configured_binary="codex",
        command_prefix=["codex"],
        resolved_kind="native",
        available=True,
    )


def test_codex_command_builder_places_approval_before_exec() -> None:
    command = build_codex_exec_command(
        fake_binary(),
        "Return JSON.",
        model="gpt-5.5",
        output_schema_path="schema.json",
    )

    exec_index = command.index("exec")
    approval_index = command.index("--ask-for-approval")
    assert approval_index < exec_index
    assert command[approval_index : approval_index + 2] == ["--ask-for-approval", "never"]
    assert "--output-schema" in command[exec_index:]


def test_codex_image_command_uses_separate_image_args() -> None:
    command = build_codex_image_exec_command(
        fake_binary(),
        "Generate from the attached references.",
        reference_image_paths=["ref_1.png", "ref_2.png"],
    )

    assert command.count("--image") == 2
    assert "ref_1.png" in command
    assert "ref_2.png" in command
    assert command[-1] == "Generate from the attached references."


def test_codex_exec_command_can_skip_git_check_for_scratch_workspace() -> None:
    command = build_codex_exec_command(
        fake_binary(),
        "-",
        sandbox="workspace-write",
        skip_git_repo_check=True,
    )

    exec_index = command.index("exec")
    assert command[exec_index + 1] == "--skip-git-repo-check"
    assert command[-1] == "-"
