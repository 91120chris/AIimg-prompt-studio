from datetime import UTC, datetime

from sqlmodel import Field, SQLModel


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


class SessionRecord(SQLModel, table=True):
    __tablename__ = "sessions"

    session_id: str = Field(primary_key=True)
    title: str | None = None
    created_at: str = Field(default_factory=utc_now_iso)


class AgentTurnRecord(SQLModel, table=True):
    __tablename__ = "agent_turns"

    agent_turn_id: str = Field(primary_key=True)
    session_id: str
    payload_json: str
    created_at: str = Field(default_factory=utc_now_iso)


class QuestionnaireRecord(SQLModel, table=True):
    __tablename__ = "questionnaires"

    questionnaire_id: str = Field(primary_key=True)
    session_id: str
    payload_json: str
    created_at: str = Field(default_factory=utc_now_iso)


class QuestionnaireAnswerRecord(SQLModel, table=True):
    __tablename__ = "questionnaire_answers"

    questionnaire_answer_id: str = Field(primary_key=True)
    session_id: str
    questionnaire_id: str
    payload_json: str
    created_at: str = Field(default_factory=utc_now_iso)


class PromptRecord(SQLModel, table=True):
    __tablename__ = "prompts"

    prompt_id: str = Field(primary_key=True)
    session_id: str
    text: str
    created_at: str = Field(default_factory=utc_now_iso)


class PromptVersionRecord(SQLModel, table=True):
    __tablename__ = "prompt_versions"

    prompt_version_id: str = Field(primary_key=True)
    session_id: str
    prompt_text: str
    title: str | None = None
    source: str = "optimized_prompt"
    metadata_json: str = "{}"
    created_at: str = Field(default_factory=utc_now_iso)


class ReferenceImageRecord(SQLModel, table=True):
    __tablename__ = "reference_images"

    reference_image_id: str = Field(primary_key=True)
    session_id: str = Field(index=True)
    slot: int = Field(index=True)
    role: str
    original_filename: str
    storage_path: str
    thumbnail_storage_path: str | None = None
    width: int
    height: int
    created_at: str = Field(default_factory=utc_now_iso)


class GenerationJobRecord(SQLModel, table=True):
    __tablename__ = "generation_jobs"

    job_id: str = Field(primary_key=True)
    session_id: str
    provider: str
    mode: str
    status: str
    parameters_json: str = "{}"
    error_json: str | None = None
    created_at: str = Field(default_factory=utc_now_iso)


class GeneratedImageRecord(SQLModel, table=True):
    __tablename__ = "generated_images"

    image_id: str = Field(primary_key=True)
    session_id: str = Field(index=True)
    role: str
    filename: str
    storage_path: str
    thumbnail_storage_path: str | None = None
    width: int
    height: int
    seed: int | None = None
    provider: str
    created_at: str = Field(default_factory=utc_now_iso)


class SkillVersionRecord(SQLModel, table=True):
    __tablename__ = "skill_versions"

    skill_version_id: str = Field(primary_key=True)
    skill_id: str
    content: str
    created_at: str = Field(default_factory=utc_now_iso)


class TemplateVersionRecord(SQLModel, table=True):
    __tablename__ = "template_versions"

    template_version_id: str = Field(primary_key=True)
    template_id: str
    content: str
    created_at: str = Field(default_factory=utc_now_iso)


class RegistryPatchProposalRecord(SQLModel, table=True):
    __tablename__ = "registry_patch_proposals"

    proposal_id: str = Field(primary_key=True)
    registry_kind: str
    change_kind: str = "update"
    target_id: str | None = None
    status: str
    summary: str | None = None
    diff_text: str
    proposed_content: str | None = None
    validation_json: str | None = None
    source_json: str | None = None
    applied_version_id: str | None = None
    created_at: str = Field(default_factory=utc_now_iso)


class ModelStatusRecord(SQLModel, table=True):
    __tablename__ = "model_status"

    model_status_id: str = Field(primary_key=True)
    provider: str
    status: str
    details_json: str = "{}"
    created_at: str = Field(default_factory=utc_now_iso)


class AppSettingRecord(SQLModel, table=True):
    __tablename__ = "app_settings"

    key: str = Field(primary_key=True)
    value: str
    updated_at: str = Field(default_factory=utc_now_iso)


class LogRecord(SQLModel, table=True):
    __tablename__ = "logs"

    log_id: str = Field(primary_key=True)
    level: str
    message: str
    created_at: str = Field(default_factory=utc_now_iso)
