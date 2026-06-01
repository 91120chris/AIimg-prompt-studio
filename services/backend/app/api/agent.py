import json
from typing import TypeAlias

from fastapi import APIRouter, HTTPException, Request
from pydantic import TypeAdapter
from sqlmodel import select

from app.core.agent_fallbacks import fallback_text_questionnaire
from app.core.file_store import generated_image_response
from app.core.prompt_compiler import (
    build_feedback_questionnaire_prompt,
    build_feedback_refinement_prompt,
    build_optimization_prompt,
    build_questionnaire_prompt,
)
from app.core.questionnaire_engine import (
    QuestionnaireValidationError,
    validate_questionnaire_answers,
)
from app.core.session_workspace import new_id
from app.db.models import (
    AgentTurnRecord,
    GeneratedImageRecord,
    GenerationJobRecord,
    PromptRecord,
    PromptVersionRecord,
    QuestionnaireAnswerRecord,
    QuestionnaireRecord,
    SessionRecord,
)
from app.db.session import new_session
from app.providers.codex.codex_agent_provider import CodexAgentRunner
from app.providers.ollama.ollama_agent_provider import OllamaAgentRunner
from app.schemas.agent import (
    AgentFeedbackQuestionnaireRequest,
    AgentQuestionnaireSubmitRequest,
    AgentRefineRequest,
    AgentTurnRequest,
    AgentTurnResponse,
    ErrorTurnResponse,
    OptimizedPromptTurnResponse,
    QuestionnaireTurnResponse,
)
from app.schemas.errors import StructuredError
from app.schemas.questionnaire import Questionnaire
from app.settings import Settings

router = APIRouter(prefix="/agent", tags=["agent"])
AgentTurnAdapter = TypeAdapter(AgentTurnResponse)
AgentRequestPayload: TypeAlias = (
    AgentTurnRequest
    | AgentQuestionnaireSubmitRequest
    | AgentFeedbackQuestionnaireRequest
    | AgentRefineRequest
)


def _engine(request: Request):
    return request.app.state.engine


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _structured_http_error(status_code: int, error: StructuredError) -> HTTPException:
    return HTTPException(status_code=status_code, detail=error.model_dump())


def _run_codex_agent(runner: CodexAgentRunner, prompt: str, payload) -> AgentTurnResponse:
    return runner.run(
        prompt,
        model=payload.codex_model,
        reasoning_effort=payload.codex_reasoning_effort,
        reasoning_summary=payload.codex_reasoning_summary,
        verbosity=payload.codex_verbosity,
    )


def _run_agent(settings: Settings, prompt: str, payload: AgentRequestPayload) -> AgentTurnResponse:
    if payload.provider == "codex_cli":
        return _run_codex_agent(CodexAgentRunner(settings), prompt, payload)
    if payload.provider == "ollama_local_llm":
        return OllamaAgentRunner(settings).run(prompt, model=payload.ollama_model)
    raise _structured_http_error(
        422,
        StructuredError(
            code="agent_provider_not_supported",
            message="不支援的 Agent Provider。",
            suggestion="請選擇 Codex CLI 或 Ollama。",
        ),
    )


def _provider_label(payload: AgentRequestPayload) -> str:
    return "Codex" if payload.provider == "codex_cli" else "Ollama"


def _run_questionnaire_agent(
    settings: Settings,
    prompt: str,
    payload: AgentTurnRequest | AgentFeedbackQuestionnaireRequest,
) -> AgentTurnResponse:
    try:
        response = _run_agent(settings, prompt, payload)
    except Exception as error:
        return fallback_text_questionnaire(_provider_label(payload), str(error))

    if isinstance(response, ErrorTurnResponse):
        reason = f"{response.error.code}: {response.error.message}"
        if response.error.suggestion:
            reason = f"{reason} {response.error.suggestion}"
        return fallback_text_questionnaire(_provider_label(payload), reason)

    return response


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
                message="找不到這個 session 的原始 prompt。",
                suggestion="請先送出一段 prompt，建立第一個 agent turn。",
            ),
        )
    return prompt_record.text


def _get_latest_optimized_prompt_text(db, session_id: str) -> str:
    prompt_version = db.exec(
        select(PromptVersionRecord)
        .where(PromptVersionRecord.session_id == session_id)
        .order_by(PromptVersionRecord.created_at.desc())
    ).first()
    if prompt_version is None:
        raise _structured_http_error(
            404,
            StructuredError(
                code="optimized_prompt_not_found",
                message="找不到這個 session 的最佳化 prompt。",
                suggestion="請先完成問卷並產生一版最佳化 prompt。",
            ),
        )
    return prompt_version.prompt_text


def _get_succeeded_generation_job(db, session_id: str, job_id: str) -> GenerationJobRecord:
    job = db.get(GenerationJobRecord, job_id)
    if job is None or job.session_id != session_id:
        raise _structured_http_error(
            404,
            StructuredError(
                code="generation_job_not_found",
                message="找不到指定的生成任務。",
                suggestion="請確認任務屬於目前 session。",
            ),
        )
    if job.status != "succeeded":
        raise _structured_http_error(
            422,
            StructuredError(
                code="generation_not_succeeded",
                message="這個生成任務尚未成功完成，不能建立回饋問卷。",
                suggestion="請先完成圖片生成，再進入回饋精修流程。",
            ),
        )
    return job


def _get_generated_images(db, session_id: str) -> list[GeneratedImageRecord]:
    images = db.exec(
        select(GeneratedImageRecord)
        .where(GeneratedImageRecord.session_id == session_id)
        .order_by(GeneratedImageRecord.created_at)
    ).all()
    if not images:
        raise _structured_http_error(
            422,
            StructuredError(
                code="generated_image_not_found",
                message="這個 session 還沒有生成圖片。",
                suggestion="請先完成圖片生成，再建立回饋問卷。",
            ),
        )
    return list(images)


def _generation_job_payload(job: GenerationJobRecord) -> dict[str, object]:
    return {
        "job_id": job.job_id,
        "session_id": job.session_id,
        "provider": job.provider,
        "mode": job.mode,
        "status": job.status,
        "created_at": job.created_at,
    }


def _generated_image_payloads(images: list[GeneratedImageRecord]) -> list[dict[str, object]]:
    return [generated_image_response(image).model_dump() for image in images]


def _get_questionnaire(db, session_id: str, questionnaire_id: str) -> Questionnaire:
    record = db.get(QuestionnaireRecord, questionnaire_id)
    if record is None or record.session_id != session_id:
        raise _structured_http_error(
            404,
            StructuredError(
                code="questionnaire_not_found",
                message="找不到指定的問卷。",
                suggestion="請重新建立問卷或確認目前 session。",
            ),
        )
    return Questionnaire.model_validate_json(record.payload_json)


@router.post("/turn", response_model=AgentTurnResponse)
def create_agent_turn(payload: AgentTurnRequest, request: Request) -> AgentTurnResponse:
    settings = request.app.state.settings

    with new_session(_engine(request)) as db:
        if db.get(SessionRecord, payload.session_id) is None:
            raise _structured_http_error(
                404,
                StructuredError(
                    code="session_not_found",
                    message="找不到指定的 session。",
                    suggestion="請先建立或選取一個 session。",
                ),
            )

        db.add(
            PromptRecord(
                prompt_id=new_id("prompt"),
                session_id=payload.session_id,
                text=payload.original_prompt,
            )
        )
        response = _run_questionnaire_agent(
            settings,
            build_questionnaire_prompt(payload),
            payload,
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
    settings = request.app.state.settings

    with new_session(_engine(request)) as db:
        if db.get(SessionRecord, payload.session_id) is None:
            raise _structured_http_error(
                404,
                StructuredError(
                    code="session_not_found",
                    message="找不到指定的 session。",
                    suggestion="請先建立或選取一個 session。",
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
        response = _run_agent(
            settings,
            build_optimization_prompt(original_prompt, questionnaire, payload),
            payload,
        )
        response = _ensure_unique_questionnaire_id(db, response)
        _store_agent_turn(db, payload.session_id, response)
        db.commit()
        return response


@router.post("/feedback-questionnaire", response_model=AgentTurnResponse)
def create_feedback_questionnaire(
    payload: AgentFeedbackQuestionnaireRequest,
    request: Request,
) -> AgentTurnResponse:
    settings = request.app.state.settings

    with new_session(_engine(request)) as db:
        if db.get(SessionRecord, payload.session_id) is None:
            raise _structured_http_error(
                404,
                StructuredError(
                    code="session_not_found",
                    message="找不到指定的 session。",
                    suggestion="請先建立或選取一個 session。",
                ),
            )

        job = _get_succeeded_generation_job(db, payload.session_id, payload.job_id)
        images = _get_generated_images(db, payload.session_id)

        original_prompt = _get_latest_prompt_text(db, payload.session_id)
        optimized_prompt = _get_latest_optimized_prompt_text(db, payload.session_id)
        response = _run_questionnaire_agent(
            settings,
            build_feedback_questionnaire_prompt(
                original_prompt=original_prompt,
                optimized_prompt=optimized_prompt,
                generation_job=_generation_job_payload(job),
                generated_images=_generated_image_payloads(images),
            ),
            payload,
        )
        response = _ensure_unique_questionnaire_id(db, response)
        _store_agent_turn(db, payload.session_id, response)
        db.commit()
        return response


@router.post("/refine", response_model=AgentTurnResponse)
def refine_prompt_from_feedback(
    payload: AgentRefineRequest,
    request: Request,
) -> AgentTurnResponse:
    settings = request.app.state.settings

    with new_session(_engine(request)) as db:
        if db.get(SessionRecord, payload.session_id) is None:
            raise _structured_http_error(
                404,
                StructuredError(
                    code="session_not_found",
                    message="找不到指定的 session。",
                    suggestion="請先建立或選取一個 session。",
                ),
            )

        job = _get_succeeded_generation_job(db, payload.session_id, payload.job_id)
        images = _get_generated_images(db, payload.session_id)
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
        previous_optimized_prompt = _get_latest_optimized_prompt_text(db, payload.session_id)
        response = _run_agent(
            settings,
            build_feedback_refinement_prompt(
                original_prompt=original_prompt,
                previous_optimized_prompt=previous_optimized_prompt,
                questionnaire=questionnaire,
                answers=payload,
                generation_job=_generation_job_payload(job),
                generated_images=_generated_image_payloads(images),
            ),
            payload,
        )
        response = _ensure_unique_questionnaire_id(db, response)
        _store_agent_turn(db, payload.session_id, response)
        db.commit()
        return response
