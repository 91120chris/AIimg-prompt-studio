import json
import re
from typing import TypeAlias

from fastapi import APIRouter, HTTPException, Request
from pydantic import TypeAdapter
from sqlmodel import select

from app.core.agent_fallbacks import fallback_text_questionnaire
from app.core.file_store import generated_image_response, reference_image_response
from app.core.registry_prompt_compiler import build_registry_proposal_prompt
from app.core.registry_proposals import (
    registry_proposal_response,
    serialize_validation,
    validate_registry_proposal,
)
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
    AppSettingRecord,
    AgentTurnRecord,
    GeneratedImageRecord,
    GenerationJobRecord,
    PromptRecord,
    PromptVersionRecord,
    QuestionnaireAnswerRecord,
    QuestionnaireRecord,
    ReferenceImageRecord,
    RegistryPatchProposalRecord,
    SessionRecord,
    SkillVersionRecord,
    TemplateVersionRecord,
)
from app.db.session import new_session
from app.providers.codex.codex_agent_provider import CodexAgentRunner
from app.providers.ollama.ollama_agent_provider import OllamaAgentRunner
from app.schemas.agent import (
    AgentFeedbackQuestionnaireRequest,
    AgentQuestionnaireSubmitRequest,
    AgentRegistryProposalRequest,
    AgentRefineRequest,
    AgentTurnRequest,
    AgentTurnResponse,
    ErrorTurnResponse,
    MessageTurnResponse,
    OptimizedPromptTurnResponse,
    QuestionnaireTurnResponse,
)
from app.schemas.errors import StructuredError
from app.schemas.questionnaire import Questionnaire
from app.schemas.registry import RegistryPatchProposalResponse
from app.settings import Settings

router = APIRouter(prefix="/agent", tags=["agent"])
AgentTurnAdapter = TypeAdapter(AgentTurnResponse)
AgentRequestPayload: TypeAlias = (
    AgentTurnRequest
    | AgentQuestionnaireSubmitRequest
    | AgentFeedbackQuestionnaireRequest
    | AgentRefineRequest
    | AgentRegistryProposalRequest
)


def _engine(request: Request):
    return request.app.state.engine


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _current_prompt_version_key(session_id: str) -> str:
    return f"current_prompt_version:{session_id}"


def _context_text(value: str, *, include: bool, label: str) -> str:
    if include:
        return value
    return f"[{label} context hidden by user.]"


def _skill_enabled_key(skill_id: str) -> str:
    return f"skill_enabled:{skill_id}"


def _is_skill_enabled(db, skill_id: str) -> bool:
    record = db.get(AppSettingRecord, _skill_enabled_key(skill_id))
    if record is None:
        return True
    return record.value != "0"


def _latest_skills(db) -> list[SkillVersionRecord]:
    records = db.exec(
        select(SkillVersionRecord).order_by(
            SkillVersionRecord.created_at,
            SkillVersionRecord.skill_version_id,
        )
    ).all()
    latest_by_id: dict[str, SkillVersionRecord] = {}
    for record in records:
        latest_by_id[record.skill_id] = record
    return list(latest_by_id.values())


def _latest_template(db, template_id: str) -> TemplateVersionRecord | None:
    return db.exec(
        select(TemplateVersionRecord)
        .where(TemplateVersionRecord.template_id == template_id)
        .order_by(
            TemplateVersionRecord.created_at.desc(),
            TemplateVersionRecord.template_version_id.desc(),
        )
    ).first()


def _template_context(db, template_id: str | None, mode: str) -> str:
    if not template_id:
        return "Selected template: auto-detect."
    record = _latest_template(db, template_id)
    if record is None:
        raise _structured_http_error(
            422,
            StructuredError(
                code="template_not_found",
                message="Selected template was not found.",
                suggestion="Please choose an existing template or use auto-detect.",
            ),
        )
    try:
        payload = json.loads(record.content)
    except json.JSONDecodeError as error:
        raise _structured_http_error(
            422,
            StructuredError(
                code="template_invalid",
                message="Selected template is not valid JSON.",
                suggestion=str(error),
            ),
        ) from error
    applies_to = payload.get("applies_to") if isinstance(payload, dict) else None
    if not isinstance(applies_to, list) or mode not in applies_to:
        raise _structured_http_error(
            422,
            StructuredError(
                code="template_mode_mismatch",
                message="Selected template does not support the current workflow mode.",
                suggestion=f"Choose a template that applies to {mode}.",
            ),
        )
    return f"Selected template JSON:\n{record.content}"


def _registry_context(
    db,
    payload: AgentRequestPayload,
    *,
    mode_override: str | None = None,
) -> str:
    enabled_skills = [
        record
        for record in _latest_skills(db)
        if _is_skill_enabled(db, record.skill_id)
    ]
    skill_text = "\n\n".join(
        f"Skill: {record.skill_id}\n{record.content}" for record in enabled_skills
    )
    mode = mode_override or getattr(payload, "mode", "t2i")
    template_text = _template_context(db, getattr(payload, "template_id", None), mode)
    return (
        "Use only the enabled skills below. Ignore disabled skills.\n\n"
        f"{skill_text or 'No enabled skills.'}\n\n"
        f"{template_text}"
    )


def _with_registry_context(prompt: str, registry_context: str) -> str:
    return f"Registry context:\n{registry_context}\n\n{prompt}"


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


def _store_agent_turn(
    db,
    session_id: str,
    response: AgentTurnResponse,
    *,
    prompt_source: str = "optimized_prompt",
    metadata: dict[str, object] | None = None,
) -> None:
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
        prompt_version_id = new_id("promptv")
        db.add(
            PromptVersionRecord(
                prompt_version_id=prompt_version_id,
                session_id=session_id,
                prompt_text=response.optimized_prompt,
                title=response.prompt_version_title,
                source=prompt_source,
                metadata_json=_json(metadata or {}),
            )
        )
        db.merge(
            AppSettingRecord(
                key=_current_prompt_version_key(session_id),
                value=prompt_version_id,
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
    current_setting = db.get(AppSettingRecord, _current_prompt_version_key(session_id))
    if current_setting is not None and current_setting.value.strip():
        current_record = db.get(PromptVersionRecord, current_setting.value)
        if current_record is not None and current_record.session_id == session_id:
            return current_record.prompt_text

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


def _get_reference_images(db, session_id: str) -> list[ReferenceImageRecord]:
    return list(
        db.exec(
            select(ReferenceImageRecord)
            .where(ReferenceImageRecord.session_id == session_id)
            .order_by(ReferenceImageRecord.slot)
        ).all()
    )


def _current_prompt_version_record(
    db,
    session_id: str,
    prompt_version_id: str | None = None,
) -> PromptVersionRecord | None:
    if prompt_version_id:
        record = db.get(PromptVersionRecord, prompt_version_id)
        if record is None or record.session_id != session_id:
            raise _structured_http_error(
                404,
                StructuredError(
                    code="prompt_version_not_found",
                    message="Prompt version not found.",
                    suggestion="Choose a prompt version from the current session.",
                ),
            )
        return record

    current_setting = db.get(AppSettingRecord, _current_prompt_version_key(session_id))
    if current_setting is not None and current_setting.value.strip():
        record = db.get(PromptVersionRecord, current_setting.value)
        if record is not None and record.session_id == session_id:
            return record

    return db.exec(
        select(PromptVersionRecord)
        .where(PromptVersionRecord.session_id == session_id)
        .order_by(PromptVersionRecord.created_at.desc(), PromptVersionRecord.prompt_version_id.desc())
    ).first()


def _latest_generation_job(db, session_id: str) -> GenerationJobRecord | None:
    return db.exec(
        select(GenerationJobRecord)
        .where(GenerationJobRecord.session_id == session_id)
        .order_by(GenerationJobRecord.created_at.desc(), GenerationJobRecord.job_id.desc())
    ).first()


def _questionnaire_answer_payload(
    db,
    session_id: str,
    questionnaire_answer_id: str | None,
) -> dict[str, object] | None:
    record = None
    if questionnaire_answer_id:
        record = db.get(QuestionnaireAnswerRecord, questionnaire_answer_id)
        if record is None or record.session_id != session_id:
            raise _structured_http_error(
                404,
                StructuredError(
                    code="questionnaire_answer_not_found",
                    message="Questionnaire answer not found.",
                    suggestion="Choose an answer from the current session.",
                ),
            )
    else:
        record = db.exec(
            select(QuestionnaireAnswerRecord)
            .where(QuestionnaireAnswerRecord.session_id == session_id)
            .order_by(
                QuestionnaireAnswerRecord.created_at.desc(),
                QuestionnaireAnswerRecord.questionnaire_answer_id.desc(),
            )
        ).first()
    if record is None:
        return None
    try:
        payload = json.loads(record.payload_json)
    except json.JSONDecodeError:
        return {"questionnaire_answer_id": record.questionnaire_answer_id}
    if isinstance(payload, dict):
        return {
            "questionnaire_answer_id": record.questionnaire_answer_id,
            "payload": payload,
        }
    return {"questionnaire_answer_id": record.questionnaire_answer_id}


def _proposal_authoring_context(
    db,
    payload: AgentRegistryProposalRequest,
) -> dict[str, object]:
    original_prompt = None
    try:
        original_prompt = _get_latest_prompt_text(db, payload.session_id)
    except HTTPException:
        original_prompt = None

    prompt_version = _current_prompt_version_record(
        db,
        payload.session_id,
        payload.current_prompt_version_id,
    )
    job = (
        _get_succeeded_generation_job(db, payload.session_id, payload.job_id)
        if payload.job_id
        else _latest_generation_job(db, payload.session_id)
    )
    images = list(
        db.exec(
            select(GeneratedImageRecord)
            .where(GeneratedImageRecord.session_id == payload.session_id)
            .order_by(GeneratedImageRecord.created_at.desc())
        ).all()
    )
    template = _latest_template(db, payload.template_id) if payload.template_id else None
    enabled_skill_ids = [
        record.skill_id
        for record in _latest_skills(db)
        if _is_skill_enabled(db, record.skill_id)
    ]
    return {
        "session_id": payload.session_id,
        "mode": payload.mode,
        "original_prompt": original_prompt,
        "current_prompt_version": (
            {
                "prompt_version_id": prompt_version.prompt_version_id,
                "title": prompt_version.title,
                "source": prompt_version.source,
                "prompt_text": prompt_version.prompt_text,
                "created_at": prompt_version.created_at,
            }
            if prompt_version is not None
            else None
        ),
        "selected_template": (
            {
                "template_id": template.template_id,
                "content": template.content,
                "created_at": template.created_at,
            }
            if template is not None
            else None
        ),
        "enabled_skill_ids": sorted(enabled_skill_ids),
        "reference_images": [
            reference_image_response(image).model_dump() for image in _get_reference_images(db, payload.session_id)
        ],
        "latest_generation_job": _generation_job_payload(job) if job is not None else None,
        "generated_images": [
            generated_image_response(image).model_dump() for image in images[:4]
        ],
        "feedback_answers": _questionnaire_answer_payload(
            db,
            payload.session_id,
            payload.questionnaire_answer_id,
        ),
    }


def _proposal_content_from_agent_response(response: AgentTurnResponse) -> tuple[str, str | None]:
    if isinstance(response, OptimizedPromptTurnResponse):
        return response.optimized_prompt, response.prompt_version_title or response.message or None
    if isinstance(response, MessageTurnResponse):
        return response.message, None
    if isinstance(response, ErrorTurnResponse):
        raise _structured_http_error(502, response.error)
    raise _structured_http_error(
        502,
        StructuredError(
            code="proposal_generation_failed",
            message="Agent did not return proposal content.",
            suggestion="Try a shorter authoring instruction or create the proposal manually.",
        ),
    )


def _infer_template_id_from_content(content: str) -> str | None:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    value = payload.get("id")
    return value.strip() if isinstance(value, str) and value.strip() else None


def _infer_skill_id_from_content(content: str) -> str | None:
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped.startswith("#"):
            continue
        title = stripped.lstrip("#").strip()
        slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
        return slug[:72] or None
    return None


def _infer_proposal_target_id(
    proposal_kind: str,
    content: str,
    explicit_target_id: str | None,
) -> str | None:
    if explicit_target_id and explicit_target_id.strip():
        return explicit_target_id.strip()
    if proposal_kind == "template":
        return _infer_template_id_from_content(content)
    return _infer_skill_id_from_content(content)


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
        registry_context = _registry_context(db, payload)
        response = _run_questionnaire_agent(
            settings,
            _with_registry_context(build_questionnaire_prompt(payload), registry_context),
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

        original_prompt = _context_text(
            _get_latest_prompt_text(db, payload.session_id),
            include=payload.include_original_prompt_context,
            label="Original prompt",
        )
        registry_context = _registry_context(db, payload)
        response = _run_agent(
            settings,
            _with_registry_context(
                build_optimization_prompt(original_prompt, questionnaire, payload),
                registry_context,
            ),
            payload,
        )
        response = _ensure_unique_questionnaire_id(db, response)
        _store_agent_turn(
            db,
            payload.session_id,
            response,
            prompt_source="optimized_prompt",
            metadata={"questionnaire_id": payload.questionnaire_id},
        )
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

        original_prompt = _context_text(
            _get_latest_prompt_text(db, payload.session_id),
            include=payload.include_original_prompt_context,
            label="Original prompt",
        )
        optimized_prompt = _context_text(
            _get_latest_optimized_prompt_text(db, payload.session_id),
            include=payload.include_optimized_prompt_context,
            label="Optimized prompt",
        )
        registry_context = _registry_context(db, payload, mode_override=job.mode)
        response = _run_questionnaire_agent(
            settings,
            _with_registry_context(
                build_feedback_questionnaire_prompt(
                    original_prompt=original_prompt,
                    optimized_prompt=optimized_prompt,
                    generation_job=_generation_job_payload(job),
                    generated_images=_generated_image_payloads(images),
                ),
                registry_context,
            ),
            payload,
        )
        response = _ensure_unique_questionnaire_id(db, response)
        _store_agent_turn(
            db,
            payload.session_id,
            response,
            metadata={"job_id": payload.job_id},
        )
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

        original_prompt = _context_text(
            _get_latest_prompt_text(db, payload.session_id),
            include=payload.include_original_prompt_context,
            label="Original prompt",
        )
        previous_optimized_prompt = _context_text(
            _get_latest_optimized_prompt_text(db, payload.session_id),
            include=payload.include_optimized_prompt_context,
            label="Optimized prompt",
        )
        registry_context = _registry_context(db, payload, mode_override=job.mode)
        response = _run_agent(
            settings,
            _with_registry_context(
                build_feedback_refinement_prompt(
                    original_prompt=original_prompt,
                    previous_optimized_prompt=previous_optimized_prompt,
                    questionnaire=questionnaire,
                    answers=payload,
                    generation_job=_generation_job_payload(job),
                    generated_images=_generated_image_payloads(images),
                ),
                registry_context,
            ),
            payload,
        )
        response = _ensure_unique_questionnaire_id(db, response)
        _store_agent_turn(
            db,
            payload.session_id,
            response,
            prompt_source="feedback_refine",
            metadata={
                "job_id": payload.job_id,
                "questionnaire_id": payload.questionnaire_id,
            },
        )
        db.commit()
        return response


@router.post("/registry-proposals", response_model=RegistryPatchProposalResponse)
def create_agent_registry_proposal(
    payload: AgentRegistryProposalRequest,
    request: Request,
) -> RegistryPatchProposalResponse:
    settings = request.app.state.settings

    with new_session(_engine(request)) as db:
        if db.get(SessionRecord, payload.session_id) is None:
            raise _structured_http_error(
                404,
                StructuredError(
                    code="session_not_found",
                    message="Session not found.",
                    suggestion="Create or select a session before authoring a proposal.",
                ),
            )

        registry_context = _registry_context(db, payload, mode_override=payload.mode)
        safe_context = _proposal_authoring_context(db, payload)
        response = _run_agent(
            settings,
            _with_registry_context(
                build_registry_proposal_prompt(
                    proposal_kind=payload.proposal_kind,
                    change_kind=payload.change_kind,
                    authoring_instruction=payload.authoring_instruction,
                    context=safe_context,
                ),
                registry_context,
            ),
            payload,
        )
        proposed_content, response_summary = _proposal_content_from_agent_response(response)
        target_id = _infer_proposal_target_id(
            payload.proposal_kind,
            proposed_content,
            payload.target_id,
        )
        if not target_id:
            target_id = f"{payload.proposal_kind}-{new_id('proposal')}"

        record = RegistryPatchProposalRecord(
            proposal_id=new_id("proposal"),
            registry_kind=payload.proposal_kind,
            change_kind=payload.change_kind,
            target_id=target_id,
            status="pending",
            summary=response_summary or payload.authoring_instruction[:180],
            diff_text=(
                f"Agent proposed {payload.change_kind} "
                f"{payload.proposal_kind} '{target_id}'."
            ),
            proposed_content=proposed_content,
            source_json=_json(
                {
                    "request": payload.model_dump(),
                    "context": safe_context,
                }
            ),
        )
        validation = validate_registry_proposal(record)
        record.validation_json = serialize_validation(validation)
        db.add(record)
        db.commit()
        db.refresh(record)
        return registry_proposal_response(record)
