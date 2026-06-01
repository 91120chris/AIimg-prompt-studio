from app.providers.codex import codex_agent_provider
from app.providers.codex.codex_agent_provider import CodexAgentRunner
from app.providers.codex.codex_binary_resolver import ResolvedCodexBinary
from app.settings import Settings


def fake_binary() -> ResolvedCodexBinary:
    return ResolvedCodexBinary(
        configured_binary="codex",
        command_prefix=["codex"],
        resolved_kind="native",
        available=True,
    )


def test_codex_agent_runner_repairs_invalid_json_once(monkeypatch) -> None:
    monkeypatch.setattr(codex_agent_provider, "resolve_codex_binary", lambda _: fake_binary())
    outputs = iter(
        [
            "not json",
            '{"kind":"message","message":"已修正","warnings":[]}',
        ]
    )
    commands: list[list[str]] = []

    input_texts: list[str | None] = []

    def fake_executor(command: list[str], timeout_seconds: int, input_text: str | None) -> str:
        commands.append(command)
        input_texts.append(input_text)
        assert timeout_seconds == 300
        return next(outputs)

    runner = CodexAgentRunner(Settings(_env_file=None), executor=fake_executor)
    response = runner.run("請回覆 JSON。", model="gpt-5.5", reasoning_effort="high")

    assert response.kind == "message"
    assert response.message == "已修正"
    assert len(commands) == 2
    for command in commands:
        assert command.index("--ask-for-approval") < command.index("exec")
        assert command.index("--sandbox") < command.index("exec")
        assert command.index("--config") < command.index("exec")
        assert 'model_reasoning_effort="high"' in command
        assert command[command.index("--sandbox") + 1] == "read-only"
        assert command[-1] == "-"
    assert input_texts[0] == "請回覆 JSON。"
    assert input_texts[1] is not None


def test_codex_agent_runner_falls_back_to_text_questionnaire_after_repair_failure(
    monkeypatch,
) -> None:
    monkeypatch.setattr(codex_agent_provider, "resolve_codex_binary", lambda _: fake_binary())
    outputs = iter(["not json", "still not json"])

    def fake_executor(command: list[str], timeout_seconds: int, input_text: str | None) -> str:
        return next(outputs)

    runner = CodexAgentRunner(Settings(_env_file=None), executor=fake_executor)

    response = runner.run("請回覆 JSON。")

    assert response.kind == "questionnaire"
    assert response.questionnaire.title == "手動補充問卷"
    assert response.questionnaire.questions[0].kind == "text"
    assert response.questionnaire.questions[0].question_id == "manual_details"
    assert response.warnings
