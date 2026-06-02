from typing import Annotated, Literal

from pydantic import Field

from app.schemas.base import StrictBaseModel
from app.schemas.errors import StructuredError
from app.schemas.provider import CodexReasoningEffort, CodexReasoningSummary, CodexVerbosity
from app.schemas.questionnaire import Questionnaire
from app.schemas.questionnaire_answers import QuestionnaireAnswerPayload


AgentProvider = Literal["codex_cli", "ollama_local_llm"]
WorkflowMode = Literal["t2i", "i2i"]
ProposalKind = Literal["skill", "template"]
ProposalChangeKind = Literal["create", "update"]


class AgentTurnRequest(StrictBaseModel):
    session_id: str
    original_prompt: str = Field(min_length=1, max_length=12000)
    mode: WorkflowMode = "t2i"
    template_id: str | None = None
    provider: AgentProvider = "codex_cli"
    include_original_prompt_context: bool = True
    include_optimized_prompt_context: bool = True
    codex_model: str | None = None
    codex_reasoning_effort: CodexReasoningEffort | None = None
    codex_reasoning_summary: CodexReasoningSummary | None = None
    codex_verbosity: CodexVerbosity | None = None
    ollama_model: str | None = None


class MessageTurnResponse(StrictBaseModel):
    kind: Literal["message"]
    message: str
    warnings: list[str] = Field(default_factory=list)


class QuestionnaireTurnResponse(StrictBaseModel):
    kind: Literal["questionnaire"]
    message: str
    questionnaire: Questionnaire
    warnings: list[str] = Field(default_factory=list)


class OptimizedPromptTurnResponse(StrictBaseModel):
    kind: Literal["optimized_prompt"]
    message: str
    optimized_prompt: str
    prompt_version_title: str | None = None
    warnings: list[str] = Field(default_factory=list)


class ErrorTurnResponse(StrictBaseModel):
    kind: Literal["error"]
    error: StructuredError


class AgentQuestionnaireSubmitRequest(QuestionnaireAnswerPayload):
    template_id: str | None = None
    mode: WorkflowMode = "t2i"
    provider: AgentProvider = "codex_cli"
    include_original_prompt_context: bool = True
    include_optimized_prompt_context: bool = True
    codex_model: str | None = None
    codex_reasoning_effort: CodexReasoningEffort | None = None
    codex_reasoning_summary: CodexReasoningSummary | None = None
    codex_verbosity: CodexVerbosity | None = None
    ollama_model: str | None = None


class AgentFeedbackQuestionnaireRequest(StrictBaseModel):
    session_id: str
    job_id: str
    template_id: str | None = None
    provider: AgentProvider = "codex_cli"
    include_original_prompt_context: bool = True
    include_optimized_prompt_context: bool = True
    codex_model: str | None = None
    codex_reasoning_effort: CodexReasoningEffort | None = None
    codex_reasoning_summary: CodexReasoningSummary | None = None
    codex_verbosity: CodexVerbosity | None = None
    ollama_model: str | None = None


class AgentRefineRequest(QuestionnaireAnswerPayload):
    job_id: str
    template_id: str | None = None
    mode: WorkflowMode = "t2i"
    provider: AgentProvider = "codex_cli"
    include_original_prompt_context: bool = True
    include_optimized_prompt_context: bool = True
    codex_model: str | None = None
    codex_reasoning_effort: CodexReasoningEffort | None = None
    codex_reasoning_summary: CodexReasoningSummary | None = None
    codex_verbosity: CodexVerbosity | None = None
    ollama_model: str | None = None


class AgentRegistryProposalRequest(StrictBaseModel):
    session_id: str
    proposal_kind: ProposalKind
    change_kind: ProposalChangeKind
    authoring_instruction: str = Field(min_length=1, max_length=4000)
    mode: WorkflowMode
    target_id: str | None = None
    current_prompt_version_id: str | None = None
    job_id: str | None = None
    questionnaire_answer_id: str | None = None
    template_id: str | None = None
    provider: AgentProvider = "codex_cli"
    codex_model: str | None = None
    codex_reasoning_effort: CodexReasoningEffort | None = None
    codex_reasoning_summary: CodexReasoningSummary | None = None
    codex_verbosity: CodexVerbosity | None = None
    ollama_model: str | None = None


AgentTurnResponse = Annotated[
    MessageTurnResponse
    | QuestionnaireTurnResponse
    | OptimizedPromptTurnResponse
    | ErrorTurnResponse,
    Field(discriminator="kind"),
]
