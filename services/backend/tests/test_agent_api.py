from fastapi.testclient import TestClient
from sqlmodel import select

from app.db.models import (
    GeneratedImageRecord,
    GenerationJobRecord,
    PromptRecord,
    PromptVersionRecord,
    QuestionnaireAnswerRecord,
    QuestionnaireRecord,
)
from app.db.session import new_session
from app.main import create_app
from app.schemas.agent import OptimizedPromptTurnResponse, QuestionnaireTurnResponse
from app.schemas.questionnaire import Questionnaire, TextQuestion
from app.settings import Settings


def make_test_app(tmp_path):
    settings = Settings(
        storage_root=str(tmp_path / "storage"),
        database_url=f"sqlite:///{tmp_path / 'app.sqlite3'}",
        _env_file=None,
    )
    app = create_app(settings)
    return app, TestClient(app)


class FakeCodexRunner:
    responses = []
    prompts = []
    models = []

    def __init__(self, settings):
        self.settings = settings

    def run(
        self,
        prompt: str,
        *,
        model: str | None = None,
        reasoning_effort: str | None = None,
        reasoning_summary: str | None = None,
        verbosity: str | None = None,
    ):
        self.prompts.append(prompt)
        self.models.append(model)
        return self.responses.pop(0)


def test_agent_questionnaire_loop_stores_questionnaire_answers_and_prompt_version(
    monkeypatch,
    tmp_path,
) -> None:
    from app.api import agent

    questionnaire = Questionnaire(
        questionnaire_id="q_test",
        title="影像需求",
        questions=[
            TextQuestion(
                kind="text",
                question_id="style",
                label="風格",
                prompt="想要什麼風格？",
                required=True,
            )
        ],
    )
    FakeCodexRunner.responses = [
        QuestionnaireTurnResponse(
            kind="questionnaire",
            message="請先補充需求。",
            questionnaire=questionnaire,
        ),
        OptimizedPromptTurnResponse(
            kind="optimized_prompt",
            message="已完成最佳化。",
            optimized_prompt="cinematic product photo, soft rim light",
            prompt_version_title="v1",
        ),
    ]
    FakeCodexRunner.prompts = []
    FakeCodexRunner.models = []
    monkeypatch.setattr(agent, "CodexAgentRunner", FakeCodexRunner)

    app, client = make_test_app(tmp_path)
    session_id = client.post("/sessions", json={"title": "Test"}).json()["session_id"]

    turn_response = client.post(
        "/agent/turn",
        json={
            "session_id": session_id,
            "original_prompt": "一張產品照片",
            "mode": "t2i",
            "provider": "codex_cli",
            "codex_model": "gpt-5.5",
        },
    )

    assert turn_response.status_code == 200
    assert turn_response.json()["kind"] == "questionnaire"
    assert FakeCodexRunner.models[0] == "gpt-5.5"

    answer_response = client.post(
        "/agent/answer-questionnaire",
        json={
            "session_id": session_id,
            "questionnaire_id": "q_test",
            "provider": "codex_cli",
            "codex_model": "gpt-5.5",
            "answers": [
                {
                    "kind": "text",
                    "question_id": "style",
                    "value": "乾淨、柔和、商業棚拍",
                }
            ],
        },
    )

    assert answer_response.status_code == 200
    payload = answer_response.json()
    assert payload["kind"] == "optimized_prompt"
    assert payload["optimized_prompt"] == "cinematic product photo, soft rim light"

    with new_session(app.state.engine) as db:
        questionnaires = db.exec(select(QuestionnaireRecord)).all()
        answers = db.exec(select(QuestionnaireAnswerRecord)).all()
        prompt_versions = db.exec(select(PromptVersionRecord)).all()

    assert len(questionnaires) == 1
    assert len(answers) == 1
    assert len(prompt_versions) == 1
    assert "cinematic product photo" in prompt_versions[0].prompt_text


def test_agent_answer_questionnaire_rejects_missing_required_answer(monkeypatch, tmp_path) -> None:
    from app.api import agent

    questionnaire = Questionnaire(
        questionnaire_id="q_required",
        title="影像需求",
        questions=[
            TextQuestion(
                kind="text",
                question_id="style",
                label="風格",
                prompt="想要什麼風格？",
                required=True,
            )
        ],
    )
    FakeCodexRunner.responses = [
        QuestionnaireTurnResponse(
            kind="questionnaire",
            message="請先補充需求。",
            questionnaire=questionnaire,
        )
    ]
    monkeypatch.setattr(agent, "CodexAgentRunner", FakeCodexRunner)

    _, client = make_test_app(tmp_path)
    session_id = client.post("/sessions", json={"title": "Test"}).json()["session_id"]
    client.post(
        "/agent/turn",
        json={"session_id": session_id, "original_prompt": "一張產品照片"},
    )

    response = client.post(
        "/agent/answer-questionnaire",
        json={
            "session_id": session_id,
            "questionnaire_id": "q_required",
            "answers": [
                {
                    "kind": "text",
                    "question_id": "style",
                    "value": "",
                }
            ],
        },
    )

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "invalid_answer"


def test_feedback_questionnaire_uses_safe_generation_metadata(monkeypatch, tmp_path) -> None:
    from app.api import agent

    questionnaire = Questionnaire(
        questionnaire_id="q_feedback",
        title="生成結果回饋",
        questions=[
            TextQuestion(
                kind="text",
                question_id="fix",
                label="要修正的地方",
                prompt="這張圖下一輪最需要修正什麼？",
                required=True,
            )
        ],
    )
    FakeCodexRunner.responses = [
        QuestionnaireTurnResponse(
            kind="questionnaire",
            message="請補充生成結果回饋。",
            questionnaire=questionnaire,
        )
    ]
    FakeCodexRunner.prompts = []
    FakeCodexRunner.models = []
    monkeypatch.setattr(agent, "CodexAgentRunner", FakeCodexRunner)

    app, client = make_test_app(tmp_path)
    session_id = client.post("/sessions", json={"title": "Test"}).json()["session_id"]

    with new_session(app.state.engine) as db:
        db.add(
            PromptRecord(
                prompt_id="prompt_feedback",
                session_id=session_id,
                text="一張玻璃香水瓶產品照",
            )
        )
        db.add(
            PromptVersionRecord(
                prompt_version_id="promptv_feedback",
                session_id=session_id,
                prompt_text="cinematic glass perfume bottle product photo",
            )
        )
        db.add(
            GenerationJobRecord(
                job_id="job_feedback",
                session_id=session_id,
                provider="codex_cli_gpt_image",
                mode="t2i",
                status="succeeded",
            )
        )
        db.add(
            GeneratedImageRecord(
                image_id="img_feedback",
                session_id=session_id,
                role="optimized_prompt",
                filename="image.png",
                storage_path=str(tmp_path / "private" / "image.png"),
                width=32,
                height=24,
                seed=123,
                provider="codex_cli_gpt_image",
            )
        )
        db.commit()

    response = client.post(
        "/agent/feedback-questionnaire",
        json={
            "session_id": session_id,
            "job_id": "job_feedback",
            "provider": "codex_cli",
            "codex_model": "gpt-5.5",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["kind"] == "questionnaire"
    assert FakeCodexRunner.models == ["gpt-5.5"]
    assert "img_feedback" in FakeCodexRunner.prompts[0]
    assert "/files/sessions/" in FakeCodexRunner.prompts[0]
    assert "storage_path" not in FakeCodexRunner.prompts[0]
    assert str(tmp_path) not in FakeCodexRunner.prompts[0]

    with new_session(app.state.engine) as db:
        stored_questionnaire = db.get(QuestionnaireRecord, "q_feedback")

    assert stored_questionnaire is not None


def test_refine_uses_feedback_and_creates_new_prompt_version(monkeypatch, tmp_path) -> None:
    from app.api import agent

    questionnaire = Questionnaire(
        questionnaire_id="q_refine",
        title="生成結果回饋",
        questions=[
            TextQuestion(
                kind="text",
                question_id="fix",
                label="要修正的地方",
                prompt="這張圖下一輪最需要修正什麼？",
                required=True,
            )
        ],
    )
    FakeCodexRunner.responses = [
        OptimizedPromptTurnResponse(
            kind="optimized_prompt",
            message="已根據回饋產生新版 prompt。",
            optimized_prompt="cinematic glass perfume bottle, cleaner label, softer highlights",
            prompt_version_title="feedback v2",
        )
    ]
    FakeCodexRunner.prompts = []
    FakeCodexRunner.models = []
    monkeypatch.setattr(agent, "CodexAgentRunner", FakeCodexRunner)

    app, client = make_test_app(tmp_path)
    session_id = client.post("/sessions", json={"title": "Test"}).json()["session_id"]

    with new_session(app.state.engine) as db:
        db.add(
            PromptRecord(
                prompt_id="prompt_refine",
                session_id=session_id,
                text="一張玻璃香水瓶產品照",
            )
        )
        db.add(
            PromptVersionRecord(
                prompt_version_id="promptv_refine_previous",
                session_id=session_id,
                prompt_text="cinematic glass perfume bottle product photo",
            )
        )
        db.add(
            GenerationJobRecord(
                job_id="job_refine",
                session_id=session_id,
                provider="codex_cli_gpt_image",
                mode="t2i",
                status="succeeded",
            )
        )
        db.add(
            GeneratedImageRecord(
                image_id="img_refine",
                session_id=session_id,
                role="optimized_prompt",
                filename="image.png",
                storage_path=str(tmp_path / "private" / "image.png"),
                width=32,
                height=24,
                seed=456,
                provider="codex_cli_gpt_image",
            )
        )
        db.add(
            QuestionnaireRecord(
                questionnaire_id=questionnaire.questionnaire_id,
                session_id=session_id,
                payload_json=questionnaire.model_dump_json(),
            )
        )
        db.commit()

    response = client.post(
        "/agent/refine",
        json={
            "session_id": session_id,
            "questionnaire_id": "q_refine",
            "job_id": "job_refine",
            "provider": "codex_cli",
            "codex_model": "gpt-5.5",
            "answers": [
                {
                    "kind": "text",
                    "question_id": "fix",
                    "value": "瓶身標籤要更清楚，反光減弱。",
                }
            ],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["kind"] == "optimized_prompt"
    assert "cleaner label" in payload["optimized_prompt"]
    assert FakeCodexRunner.models == ["gpt-5.5"]
    assert "job_refine" in FakeCodexRunner.prompts[0]
    assert "瓶身標籤要更清楚" in FakeCodexRunner.prompts[0]
    assert "storage_path" not in FakeCodexRunner.prompts[0]
    assert str(tmp_path) not in FakeCodexRunner.prompts[0]

    with new_session(app.state.engine) as db:
        answers = db.exec(select(QuestionnaireAnswerRecord)).all()
        prompt_versions = db.exec(
            select(PromptVersionRecord)
            .where(PromptVersionRecord.session_id == session_id)
            .order_by(PromptVersionRecord.created_at)
        ).all()

    assert len(answers) == 1
    assert len(prompt_versions) == 2
    assert "cleaner label" in prompt_versions[-1].prompt_text


def test_agent_turn_rejects_ollama_until_provider_loop_exists(tmp_path) -> None:
    _, client = make_test_app(tmp_path)
    session_id = client.post("/sessions", json={"title": "Test"}).json()["session_id"]

    response = client.post(
        "/agent/turn",
        json={
            "session_id": session_id,
            "original_prompt": "一張產品照片",
            "provider": "ollama_local_llm",
        },
    )

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "agent_provider_not_implemented"
