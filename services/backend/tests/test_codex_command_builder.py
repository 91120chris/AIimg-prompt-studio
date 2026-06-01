from app.providers.codex.codex_binary_resolver import ResolvedCodexBinary
from app.providers.codex.codex_command_builder import (
    build_codex_exec_command,
    build_codex_exec_resume_command,
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


def test_codex_exec_command_adds_config_overrides_before_exec() -> None:
    command = build_codex_exec_command(
        fake_binary(),
        "-",
        config_overrides={
            "model_reasoning_effort": "xhigh",
            "model_verbosity": "high",
        },
    )

    exec_index = command.index("exec")
    config_indices = [index for index, value in enumerate(command) if value == "--config"]
    assert config_indices
    assert all(index < exec_index for index in config_indices)
    assert 'model_reasoning_effort="xhigh"' in command
    assert 'model_verbosity="high"' in command


def test_codex_exec_command_can_request_json_events() -> None:
    command = build_codex_exec_command(fake_binary(), "-", json_output=True)

    exec_index = command.index("exec")
    assert "--json" in command[exec_index:]
    assert command[-1] == "-"


def test_codex_resume_command_targets_saved_session() -> None:
    command = build_codex_exec_resume_command(
        fake_binary(),
        "019e8426-42bc-7122-bce0-608dc8f5bb00",
        "-",
        output_schema_path="schema.json",
        json_output=True,
    )

    exec_index = command.index("exec")
    assert command[exec_index + 1] == "resume"
    assert "--output-schema" in command[exec_index:]
    assert "--json" in command[exec_index:]
    assert command[-2:] == ["019e8426-42bc-7122-bce0-608dc8f5bb00", "-"]
