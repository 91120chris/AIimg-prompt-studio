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
        assert "--json" in command[command.index("exec") :]
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


def test_parse_agent_turn_response_accepts_raw_questionnaire_payload() -> None:
    response = codex_agent_provider.parse_agent_turn_response(
        """
        {
          "title": "Feedback",
          "questions": [
            {
              "type": "textarea",
              "id": "changes",
              "question": "What should change?"
            }
          ]
        }
        """
    )

    assert response.kind == "questionnaire"
    assert response.questionnaire.title == "Feedback"
    assert response.questionnaire.questions[0].kind == "text"
    assert response.questionnaire.questions[0].question_id == "changes"
    assert response.questionnaire.questions[0].prompt == "What should change?"


def test_parse_agent_turn_response_normalizes_feedback_questionnaire_aliases() -> None:
    response = codex_agent_provider.parse_agent_turn_response(
        """
        {
          "kind": "feedback_questionnaire",
          "message": "Answer these before refining.",
          "questionnaire": {
            "name": "Refine feedback",
            "fields": [
              {
                "kind": "multi_choice",
                "name": "fixes",
                "question": "Which areas need work?",
                "options": ["lighting", "composition"]
              },
              {
                "type": "rating",
                "id": "quality",
                "question": "How close is it?",
                "min": "1",
                "max": "5"
              }
            ]
          }
        }
        """
    )

    assert response.kind == "questionnaire"
    assert response.message == "Answer these before refining."
    assert response.questionnaire.title == "Refine feedback"

    choice_question = response.questionnaire.questions[0]
    assert choice_question.kind == "choice"
    assert choice_question.question_id == "fixes"
    assert choice_question.allow_multiple is True
    assert [option.value for option in choice_question.options] == [
        "lighting",
        "composition",
    ]

    scale_question = response.questionnaire.questions[1]
    assert scale_question.kind == "scale"
    assert scale_question.min_value == 1
    assert scale_question.max_value == 5


def test_parse_agent_turn_response_normalizes_optimized_prompt_alias() -> None:
    response = codex_agent_provider.parse_agent_turn_response(
        """
        {
          "kind": "prompt_optimization",
          "message": "Done.",
          "final_prompt": "cinematic portrait, soft rim light",
          "title": "Cinematic portrait"
        }
        """
    )

    assert response.kind == "optimized_prompt"
    assert response.message == "Done."
    assert response.optimized_prompt == "cinematic portrait, soft rim light"
    assert response.prompt_version_title == "Cinematic portrait"


def test_parse_agent_turn_response_reads_codex_jsonl_events() -> None:
    raw_output = "\n".join(
        [
            '{"type":"session_meta","payload":{"id":"019e8426-42bc-7122-bce0-608dc8f5bb00"}}',
            '{"type":"event_msg","payload":{"type":"agent_message","message":"{\\"kind\\":\\"message\\",\\"message\\":\\"ok\\",\\"warnings\\":[]}"}}',
        ]
    )

    response = codex_agent_provider.parse_agent_turn_response(raw_output)

    assert response.kind == "message"
    assert response.message == "ok"
    assert (
        codex_agent_provider.extract_codex_session_id(raw_output)
        == "019e8426-42bc-7122-bce0-608dc8f5bb00"
    )


def test_codex_agent_runner_resumes_existing_session(monkeypatch) -> None:
    monkeypatch.setattr(codex_agent_provider, "resolve_codex_binary", lambda _: fake_binary())

    def fake_executor(command: list[str], timeout_seconds: int, input_text: str | None) -> str:
        assert command[command.index("exec") + 1] == "resume"
        assert "019e8426-42bc-7122-bce0-608dc8f5bb00" in command
        assert "--json" in command
        assert input_text == "continue"
        return "\n".join(
            [
                '{"type":"session_meta","payload":{"id":"019e8426-42bc-7122-bce0-608dc8f5bb00"}}',
                '{"type":"event_msg","payload":{"type":"agent_message","message":"{\\"kind\\":\\"message\\",\\"message\\":\\"resumed\\",\\"warnings\\":[]}"}}',
            ]
        )

    runner = CodexAgentRunner(Settings(_env_file=None), executor=fake_executor)

    response = runner.run(
        "continue",
        codex_session_id="019e8426-42bc-7122-bce0-608dc8f5bb00",
    )

    assert response.kind == "message"
    assert response.message == "resumed"
    assert runner.last_codex_session_id == "019e8426-42bc-7122-bce0-608dc8f5bb00"
