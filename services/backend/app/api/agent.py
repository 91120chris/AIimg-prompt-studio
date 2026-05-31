import json

from fastapi import APIRouter, HTTPException, Request
from pydantic import TypeAdapter
from sqlmodel import select

from app.core.prompt_compiler import build_optimization_prompt, build_questionnaire_prompt
from app.core.questionnaire_engine import (
    QuestionnaireValidationError,
    validate_questionnaire_answers,
)
from app.core.session_workspace import new_id
from app.db.models import (
    AgentTurnRecord,
    PromptRecord,
    PromptVersionRecord,
    QuestionnaireAnswerRecord,
    QuestionnaireRecord,
    SessionRecord,
)
from app.db.session import new_session
from app.providers.codex.codex_agent_provider import CodexAgentRunner
from app.schemas.agent import (
    AgentQuestionnaireSubmitRequest,
    AgentTurnRequest,
    AgentTurnResponse,
    OptimizedPromptTurnResponse,
    QuestionnaireTurnResponse,
)
from app.schemas.errors import StructuredError
from app.schemas.questionnaire import Questionnaire

router = APIRouter(prefix="/agent", tags=["agent"])
AgentTurnAdapter = TypeAdapter(AgentTurnResponse)


def _engine(request: Request):
    return request.app.state.engine


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _structured_http_error(status_code: int, error: StructuredError) -> HTTPException:
    return HTTPException(status_code=status_code, detail=error.model_dump())


def _ensure_codex_provider(provider: str) -> None:
    if provider != "codex_cli":
        raise _structured_http_error(
            422,
            StructuredError(
                code="agent_provider_not_implemented",
                message="Milestone 1B 目前先啟用 Codex CLI 問卷流程。",
                suggestion="請先將 Agent Provider 選為 Codex CLI；Ollama agent loop 會接在後續版本。",
            ),
        )


def _store_agent_turn(db, session_id: str, response: AgentTurnResponse) -> None:
    response_payload = AgentTurnAdapter.dump_python(response, mode="json")
    db.add(
        AgentTurnRecord(
            agent_turn_id=new_id("turn"),
            session_id=session_id,
            payload_json=_json(response_payload),
        )
    )
    if isinstance(response, QuestionnaireTurnResponse):
        db.add(
            QuestionnaireRecord(
                questionnaire_id=response.questionnaire.questionnaire_id,
                session_id=session_id,
                payload_json=response.questionnaire.model_dump_json(),
            )
        )
    if isinstance(response, OptimizedPromptTurnResponse):
        db.add(
            PromptVersionRecord(
                prompt_version_id=new_id("promptv"),
                session_id=session_id,
                prompt_text=response.optimized_prompt,
            )
        )


def _ensure_unique_questionnaire_id(db, response: AgentTurnResponse) -> AgentTurnResponse:
    if not isinstance(response, QuestionnaireTurnResponse):
        return response
    if db.get(QuestionnaireRecord, response.questionnaire.questionnaire_id) is None:
        return response

    questionnaire = response.questionnaire.model_copy(
        update={"questionnaire_id": new_id("q")},
        deep=True,
    )
    return response.model_copy(update={"questionnaire": questionnaire}, deep=True)


def _get_latest_prompt_text(db, session_id: str) -> str:
    prompt_record = db.exec(
        select(PromptRecord)
        .where(PromptRecord.session_id == session_id)
        .order_by(PromptRecord.created_at.desc())
    ).first()
    if prompt_record is None:
        raise _structured_http_error(
            404,
            StructuredError(
                code="prompt_not_found",
                message="找不到此工作階段的原始 prompt。",
                suggestion="請先送出原始 prompt 產生問卷。",
            ),
        )
    return prompt_record.text


def _get_questionnaire(db, session_id: str, questionnaire_id: str) -> Questionnaire:
    record = db.get(QuestionnaireRecord, questionnaire_id)
    if record is None or record.session_id != session_id:
        raise _structured_http_error(
            404,
            StructuredError(
                code="questionnaire_not_found",
                message="找不到指定問卷。",
                suggestion="請重新產生問卷後再送出答案。",
            ),
        )
    return Questionnaire.model_validate_json(record.payload_json)


@router.post("/turn", response_model=AgentTurnResponse)
def create_agent_turn(payload: AgentTurnRequest, request: Request) -> AgentTurnResponse:
    _ensure_codex_provider(payload.provider)
    settings = request.app.state.settings
    runner = CodexAgentRunner(settings)

    with new_session(_engine(request)) as db:
        if db.get(SessionRecord, payload.session_id) is None:
            raise _structured_http_error(
                404,
                StructuredError(
                    code="session_not_found",
                    message="找不到工作階段。",
                    suggestion="請先建立工作階段。",
                ),
            )

        db.add(
            PromptRecord(
                prompt_id=new_id("prompt"),
                session_id=payload.session_id,
                text=payload.original_prompt,
            )
        )
        response = runner.run(
            build_questionnaire_prompt(payload),
            model=payload.codex_model,
        )
        response = _ensure_unique_questionnaire_id(db, response)
        _store_agent_turn(db, payload.session_id, response)
        db.commit()
        return response


@router.post("/answer-questionnaire", response_model=AgentTurnResponse)
def answer_questionnaire(
    payload: AgentQuestionnaireSubmitRequest,
    request: Request,
) -> AgentTurnResponse:
    _ensure_codex_provider(payload.provider)
    settings = request.app.state.settings
    runner = CodexAgentRunner(settings)

    with new_session(_engine(request)) as db:
        if db.get(SessionRecord, payload.session_id) is None:
            raise _structured_http_error(
                404,
                StructuredError(
                    code="session_not_found",
                    message="找不到工作階段。",
                    suggestion="請先建立工作階段。",
                ),
            )

        questionnaire = _get_questionnaire(db, payload.session_id, payload.questionnaire_id)
        try:
            validate_questionnaire_answers(questionnaire, payload)
        except QuestionnaireValidationError as error:
            raise _structured_http_error(422, error.error) from error

        db.add(
            QuestionnaireAnswerRecord(
                questionnaire_answer_id=new_id("answer"),
                session_id=payload.session_id,
                questionnaire_id=payload.questionnaire_id,
                payload_json=payload.model_dump_json(),
            )
        )

        original_prompt = _get_latest_prompt_text(db, payload.session_id)
        response = runner.run(
            build_optimization_prompt(original_prompt, questionnaire, payload),
            model=payload.codex_model,
        )
        response = _ensure_unique_questionnaire_id(db, response)
        _store_agent_turn(db, payload.session_id, response)
        db.commit()
        return response
