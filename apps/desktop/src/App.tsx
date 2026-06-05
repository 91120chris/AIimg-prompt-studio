import { useEffect, useMemo, useRef, useState } from "react";
import {
  Copy,
  Database,
  Download,
  Eye,
  FolderOpen,
  History,
  Plus,
  RefreshCcw,
  Save,
  Settings as SettingsIcon,
  SlidersHorizontal,
  X,
} from "lucide-react";

import {
  type AgentTurnResponse,
  type CodexModelsResponse,
  type CodexStatusResponse,
  type GenerationJobResponse,
  type HealthResponse,
  type LocalFluxSettingsResponse,
  type LocalFluxStatusResponse,
  type LogResponse,
  type ModelInfoResponse,
  type OllamaModelsResponse,
  type OllamaStatusResponse,
  type PromptVersionResponse,
  type Question,
  type Questionnaire,
  type ReferenceImageResponse,
  type RegistryItemResponse,
  type RegistryPatchProposalResponse,
  type RegistryPatchProposalValidationResponse,
  type SafeSettingsResponse,
  type SecretStatusResponse,
  type SessionResponse,
  type TemplatePreviewResponse,
  type TemplateValidationResponse,
  agentTurnResponseSchema,
  codexModelsResponseSchema,
  codexStatusResponseSchema,
  generationJobResponseSchema,
  healthResponseSchema,
  localFluxSettingsResponseSchema,
  localFluxStatusResponseSchema,
  localFluxWorkflowValidationResponseSchema,
  logsResponseSchema,
  loraListResponseSchema,
  modelInfoListResponseSchema,
  ollamaModelsResponseSchema,
  ollamaStatusResponseSchema,
  promptVersionResponseSchema,
  promptVersionsResponseSchema,
  referenceImageResponseSchema,
  registryItemResponseSchema,
  registryItemsResponseSchema,
  registryPatchProposalResponseSchema,
  registryPatchProposalValidationResponseSchema,
  registryPatchProposalsResponseSchema,
  safeSettingsResponseSchema,
  secretStatusResponseSchema,
  sessionResponseSchema,
  sessionsResponseSchema,
  templatePreviewResponseSchema,
  templateValidationResponseSchema,
} from "./schemas/api.ts";

type BackendStatus = "checking" | "connected" | "disconnected";
type AgentProvider = "codex_cli" | "ollama_local_llm";
type ImageProvider = "codex_cli_gpt_image" | "local_flux";
type WorkflowMode = "t2i" | "i2i";
type LocalFluxPathField =
  | "workflow_path"
  | "i2i_one_workflow_path"
  | "i2i_two_workflow_path"
  | "model_path"
  | "vae_path"
  | "text_encoder_path";
type AnswerDraftValue = string | boolean | number | string[];
type AnswerDrafts = Record<string, AnswerDraftValue>;
type FeedbackQuestionnaireContext = { questionnaireId: string; jobId: string };
type ManagerTab = "models" | "skills" | "templates" | "prompt" | "proposals" | "logs";
type ProposalKind = "skill" | "template";
type ProposalChangeKind = "create" | "update";
type QuestionnaireAnswerRequest =
  | { kind: "text"; question_id: string; value: string }
  | { kind: "choice"; question_id: string; value: string }
  | { kind: "multi_choice"; question_id: string; values: string[] }
  | { kind: "boolean"; question_id: string; value: boolean }
  | { kind: "scale"; question_id: string; value: number };

declare global {
  interface Window {
    __TAURI_INTERNALS__?: unknown;
  }
}

const fallbackCodexModels: CodexModelsResponse = {
  default_model: "gpt-5.5",
  model_options: [
    "gpt-5.5",
    "gpt-5.4",
    "gpt-5.4-mini",
    "gpt-5.3-codex",
    "gpt-5.3-codex-spark",
    "gpt-5.2",
  ],
  default_reasoning_effort: "medium",
  reasoning_effort_options: ["low", "medium", "high", "xhigh"],
  default_reasoning_summary: "auto",
  default_verbosity: null,
};

const fallbackOllamaModels: OllamaModelsResponse = {
  selected_model: null,
  models: [],
};

const LOCAL_FLUX_PROVIDER_ID = "local_flux";

const fallbackLocalFluxSettings: LocalFluxSettingsResponse = {
  provider: "local_flux",
  base_url: "http://127.0.0.1:8188",
  workflow_path:
    "workflow/Flux 2 Klein 9B FP8 (Distilled)/Flux 2 Klein 9B FP8 (Distilled) - Image Generation.json",
  i2i_one_workflow_path:
    "workflow/Flux 2 Klein 9B FP8 (Distilled)/Flux 2 Klein 9B FP8 (Distilled) - One Image Edit.json",
  i2i_two_workflow_path:
    "workflow/Flux 2 Klein 9B FP8 (Distilled)/Flux 2 Klein 9B FP8 (Distilled) - Two Images Edit.json",
  model_path: "flux2\\flux-2-klein-9b-fp8mixed.safetensors",
  vae_path: "flux\\flux2-vae.safetensors",
  text_encoder_path: "qwen\\qwen_3_8b_fp8mixed.safetensors",
  width: 1024,
  height: 1024,
  seed: null,
  steps: 4,
  cfg: 1,
  sampler_name: "euler",
  scheduler: "simple",
  denoise: 1,
  guidance: 3.5,
  output_prefix: "aiimg",
  timeout_seconds: 600,
};

const backendBaseUrl = __FRONTEND_API_BASE_URL__.replace(/\/$/, "");

async function fetchJson<T>(
  path: string,
  schema: { parse: (value: unknown) => T },
  signal?: AbortSignal,
): Promise<T> {
  const response = await fetch(`${backendBaseUrl}${path}`, { signal });
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}`);
  }
  return schema.parse(await response.json());
}

function backendFileUrl(path: string, cacheKey?: string): string {
  const url = `${backendBaseUrl}${path}`;
  if (!cacheKey) {
    return url;
  }
  const separator = url.includes("?") ? "&" : "?";
  return `${url}${separator}v=${encodeURIComponent(cacheKey)}`;
}

async function readErrorMessage(response: Response): Promise<string> {
  try {
    const payload = await response.json();
    const detail = payload?.detail;
    if (detail?.message) {
      const errors = Array.isArray(detail.errors) ? ` ${detail.errors.join(" ")}` : "";
      return detail.suggestion
        ? `${detail.message} ${detail.suggestion}${errors}`
        : `${detail.message}${errors}`;
    }
  } catch {
    // Fall back to HTTP status below when the backend did not send JSON.
  }
  return `HTTP ${response.status}`;
}

function isTauriRuntime(): boolean {
  return typeof window !== "undefined" && window.__TAURI_INTERNALS__ !== undefined;
}

function statusText(value: boolean | undefined): string {
  if (value === true) {
    return "可用";
  }
  if (value === false) {
    return "不可用";
  }
  return "檢查中";
}

function imageProviderLabel(provider: ImageProvider): string {
  return provider === "local_flux" ? "Local Flux" : "Codex Image";
}

function templateName(template: RegistryItemResponse): string {
  try {
    const payload = JSON.parse(template.content) as { name?: unknown };
    return typeof payload.name === "string" && payload.name.trim()
      ? payload.name
      : template.item_id;
  } catch {
    return template.item_id;
  }
}

function formatSessionTime(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  return new Intl.DateTimeFormat("zh-TW", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  }).format(date);
}

function promptVersionLabel(version: PromptVersionResponse): string {
  const title = version.title?.trim() || version.source.replace(/_/g, " ");
  return `${title} / ${formatSessionTime(version.created_at)}`;
}

function proposalValidationMessage(
  proposal: RegistryPatchProposalResponse | null,
): string {
  const validation = proposal?.validation as
    | { valid?: unknown; errors?: unknown }
    | null
    | undefined;
  if (!validation) {
    return "尚未驗證";
  }
  if (validation.valid === true) {
    return "驗證通過";
  }
  const errors = Array.isArray(validation.errors)
    ? validation.errors.filter((item): item is string => typeof item === "string")
    : [];
  return errors.length > 0 ? errors.join(" ") : "驗證未通過";
}

function defaultDraftsForQuestionnaire(questionnaire: Questionnaire): AnswerDrafts {
  return Object.fromEntries(
    questionnaire.questions.map((question) => {
      if (question.kind === "choice" && question.allow_multiple) {
        return [question.question_id, []];
      }
      if (question.kind === "choice") {
        return [question.question_id, ""];
      }
      if (question.kind === "boolean") {
        return [question.question_id, false];
      }
      if (question.kind === "scale") {
        return [question.question_id, question.min_value];
      }
      return [question.question_id, ""];
    }),
  );
}

function questionnaireAnswers(
  questionnaire: Questionnaire,
  drafts: AnswerDrafts,
): QuestionnaireAnswerRequest[] {
  const answers: QuestionnaireAnswerRequest[] = [];
  questionnaire.questions.forEach((question) => {
    const draft = drafts[question.question_id];
    if (question.kind === "text") {
      const value = typeof draft === "string" ? draft.trim() : "";
      if (!value && !question.required) {
        return;
      }
      if (!value) {
        throw new Error(`請填寫「${question.label}」。`);
      }
      answers.push({ kind: "text", question_id: question.question_id, value });
      return;
    }

    if (question.kind === "choice" && question.allow_multiple) {
      const values = Array.isArray(draft) ? draft.filter(Boolean) : [];
      if (values.length === 0 && !question.required) {
        return;
      }
      if (values.length === 0) {
        throw new Error(`請選擇「${question.label}」。`);
      }
      answers.push({ kind: "multi_choice", question_id: question.question_id, values });
      return;
    }

    if (question.kind === "choice") {
      const value = typeof draft === "string" ? draft : "";
      if (!value && !question.required) {
        return;
      }
      if (!value) {
        throw new Error(`請選擇「${question.label}」。`);
      }
      answers.push({ kind: "choice", question_id: question.question_id, value });
      return;
    }

    if (question.kind === "boolean") {
      answers.push({ kind: "boolean", question_id: question.question_id, value: draft === true });
      return;
    }

    const value = typeof draft === "number" ? draft : question.min_value;
    answers.push({ kind: "scale", question_id: question.question_id, value });
  });
  return answers;
}

function App() {
  const settingsTriggerRef = useRef<HTMLButtonElement | null>(null);
  const settingsCloseRef = useRef<HTMLButtonElement | null>(null);
  const settingsDrawerRef = useRef<HTMLElement | null>(null);
  const historyTriggerRef = useRef<HTMLButtonElement | null>(null);
  const historyCloseRef = useRef<HTMLButtonElement | null>(null);
  const historyDrawerRef = useRef<HTMLElement | null>(null);
  const managerTriggerRef = useRef<HTMLButtonElement | null>(null);
  const managerCloseRef = useRef<HTMLButtonElement | null>(null);
  const localFluxTriggerRef = useRef<HTMLButtonElement | null>(null);
  const localFluxCloseRef = useRef<HTMLButtonElement | null>(null);
  const localFluxDrawerRef = useRef<HTMLElement | null>(null);
  const [workflowMode, setWorkflowMode] = useState<WorkflowMode>("t2i");
  const [agentProvider, setAgentProvider] = useState<AgentProvider>("codex_cli");
  const [imageProvider, setImageProvider] = useState<ImageProvider>("codex_cli_gpt_image");
  const [originalPrompt, setOriginalPrompt] = useState("");
  const [optimizedPrompt, setOptimizedPrompt] = useState("");
  const [includeOriginalPromptContext, setIncludeOriginalPromptContext] = useState(true);
  const [includeOptimizedPromptContext, setIncludeOptimizedPromptContext] = useState(true);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [historyOpen, setHistoryOpen] = useState(false);
  const [managerOpen, setManagerOpen] = useState(false);
  const [localFluxOpen, setLocalFluxOpen] = useState(false);
  const [settingsMessage, setSettingsMessage] = useState<string | null>(null);
  const [historyMessage, setHistoryMessage] = useState<string | null>(null);
  const [historyBusy, setHistoryBusy] = useState(false);
  const [sessionHistory, setSessionHistory] = useState<SessionResponse[]>([]);
  const [managerMessage, setManagerMessage] = useState<string | null>(null);
  const [managerBusy, setManagerBusy] = useState(false);
  const [managerTab, setManagerTab] = useState<ManagerTab>("models");
  const [managerModels, setManagerModels] = useState<ModelInfoResponse[]>([]);
  const [managerSkills, setManagerSkills] = useState<RegistryItemResponse[]>([]);
  const [managerTemplates, setManagerTemplates] = useState<RegistryItemResponse[]>([]);
  const [managerPromptVersions, setManagerPromptVersions] = useState<PromptVersionResponse[]>([]);
  const [registryProposals, setRegistryProposals] = useState<RegistryPatchProposalResponse[]>([]);
  const [managerLogs, setManagerLogs] = useState<LogResponse[]>([]);
  const [templates, setTemplates] = useState<RegistryItemResponse[]>([]);
  const [selectedTemplateId, setSelectedTemplateId] = useState("");
  const [managerSelectedTemplateId, setManagerSelectedTemplateId] = useState("");
  const [templateEditorContent, setTemplateEditorContent] = useState("");
  const [templatePreviewSamplePrompt, setTemplatePreviewSamplePrompt] = useState("");
  const [templateValidation, setTemplateValidation] = useState<TemplateValidationResponse | null>(null);
  const [templatePreview, setTemplatePreview] = useState<TemplatePreviewResponse | null>(null);
  const [selectedProposalId, setSelectedProposalId] = useState("");
  const [proposalEditorContent, setProposalEditorContent] = useState("");
  const [proposalTargetId, setProposalTargetId] = useState("");
  const [proposalAuthoringInstruction, setProposalAuthoringInstruction] =
    useState("把目前 prompt / feedback 整理成可重用規則");
  const [proposalKind, setProposalKind] = useState<ProposalKind>("template");
  const [proposalChangeKind, setProposalChangeKind] = useState<ProposalChangeKind>("create");
  const [proposalValidation, setProposalValidation] =
    useState<RegistryPatchProposalValidationResponse | null>(null);
  const [localFluxStatus, setLocalFluxStatus] = useState<LocalFluxStatusResponse | null>(null);
  const [localFluxDraft, setLocalFluxDraft] =
    useState<LocalFluxSettingsResponse>(fallbackLocalFluxSettings);
  const [localFluxMessage, setLocalFluxMessage] = useState<string | null>(null);
  const [localFluxBusy, setLocalFluxBusy] = useState(false);
  const [localFluxPickerBusy, setLocalFluxPickerBusy] = useState<LocalFluxPathField | null>(null);
  const [backendStatus, setBackendStatus] = useState<BackendStatus>("checking");
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [codexStatus, setCodexStatus] = useState<CodexStatusResponse | null>(null);
  const [ollamaStatus, setOllamaStatus] = useState<OllamaStatusResponse | null>(null);
  const [ollamaModels, setOllamaModels] = useState<OllamaModelsResponse>(fallbackOllamaModels);
  const [secretStatus, setSecretStatus] = useState<SecretStatusResponse | null>(null);
  const [codexModels, setCodexModels] = useState<CodexModelsResponse>(fallbackCodexModels);
  const [codexOptionsDraft, setCodexOptionsDraft] = useState(
    fallbackCodexModels.model_options.join(", "),
  );
  const [currentSession, setCurrentSession] = useState<SessionResponse | null>(null);
  const [referenceImages, setReferenceImages] = useState<ReferenceImageResponse[]>([]);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [agentTurn, setAgentTurn] = useState<AgentTurnResponse | null>(null);
  const [answerDrafts, setAnswerDrafts] = useState<AnswerDrafts>({});
  const [agentBusy, setAgentBusy] = useState(false);
  const [agentMessage, setAgentMessage] = useState<string | null>(null);
  const [generationJob, setGenerationJob] = useState<GenerationJobResponse | null>(null);
  const [generationBusy, setGenerationBusy] = useState(false);
  const [generationMessage, setGenerationMessage] = useState<string | null>(null);
  const [loraList, setLoraList] = useState<string[]>([]);
  const [selectedLora, setSelectedLora] = useState<string | null>(null);
  const [loraWeight, setLoraWeight] = useState(0.8);
  const [feedbackQuestionnaireContext, setFeedbackQuestionnaireContext] =
    useState<FeedbackQuestionnaireContext | null>(null);
  const isCodexAgent = agentProvider === "codex_cli";
  const activeAgentName = isCodexAgent ? "Codex" : "Ollama";
  const localFluxModel = useMemo(
    () => managerModels.find((model) => model.provider === LOCAL_FLUX_PROVIDER_ID) ?? null,
    [managerModels],
  );
  const selectedTemplate = useMemo(
    () => templates.find((template) => template.item_id === selectedTemplateId) ?? null,
    [selectedTemplateId, templates],
  );
  const selectedProposal = useMemo(
    () => registryProposals.find((proposal) => proposal.proposal_id === selectedProposalId) ?? null,
    [registryProposals, selectedProposalId],
  );

  useEffect(() => {
    setCodexOptionsDraft(codexModels.model_options.join(", "));
  }, [codexModels.model_options]);

  useEffect(() => {
    if (selectedTemplateId && !templates.some((template) => template.item_id === selectedTemplateId)) {
      setSelectedTemplateId("");
    }
  }, [selectedTemplateId, templates]);

  useEffect(() => {
    if (imageProvider !== "local_flux" || backendStatus !== "connected") {
      return;
    }
    fetchJson("/models/loras", loraListResponseSchema)
      .then((list) => setLoraList(list))
      .catch(() => setLoraList([]));
  }, [imageProvider, backendStatus]);

  useEffect(() => {
    if (!settingsOpen) {
      return;
    }

    const previousFocus =
      document.activeElement instanceof HTMLElement
        ? document.activeElement
        : settingsTriggerRef.current;

    window.setTimeout(() => {
      settingsCloseRef.current?.focus();
    }, 0);
    setHistoryOpen(false);
    setManagerOpen(false);
    setLocalFluxOpen(false);

    function handleDrawerKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") {
        event.preventDefault();
        setSettingsOpen(false);
        return;
      }

      if (event.key !== "Tab") {
        return;
      }

      const drawer = settingsDrawerRef.current;
      if (!drawer) {
        return;
      }

      const focusable = Array.from(
        drawer.querySelectorAll<HTMLElement>(
          'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])',
        ),
      ).filter((element) => element.offsetParent !== null);

      if (focusable.length === 0) {
        return;
      }

      const first = focusable[0];
      const last = focusable[focusable.length - 1];

      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    }

    window.addEventListener("keydown", handleDrawerKeyDown);
    return () => {
      window.removeEventListener("keydown", handleDrawerKeyDown);
      previousFocus?.focus();
    };
  }, [settingsOpen]);

  useEffect(() => {
    if (!historyOpen) {
      return;
    }

    const previousFocus =
      document.activeElement instanceof HTMLElement
        ? document.activeElement
        : historyTriggerRef.current;

    window.setTimeout(() => {
      historyCloseRef.current?.focus();
    }, 0);
    setSettingsOpen(false);
    setManagerOpen(false);
    setLocalFluxOpen(false);

    function handleDrawerKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") {
        event.preventDefault();
        setHistoryOpen(false);
        return;
      }

      if (event.key !== "Tab") {
        return;
      }

      const drawer = historyDrawerRef.current;
      if (!drawer) {
        return;
      }

      const focusable = Array.from(
        drawer.querySelectorAll<HTMLElement>(
          'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])',
        ),
      ).filter((element) => element.offsetParent !== null);

      if (focusable.length === 0) {
        return;
      }

      const first = focusable[0];
      const last = focusable[focusable.length - 1];

      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    }

    window.addEventListener("keydown", handleDrawerKeyDown);
    return () => {
      window.removeEventListener("keydown", handleDrawerKeyDown);
      previousFocus?.focus();
    };
  }, [historyOpen]);

  useEffect(() => {
    if (!managerOpen) {
      return;
    }

    const previousFocus =
      document.activeElement instanceof HTMLElement
        ? document.activeElement
        : managerTriggerRef.current;

    window.setTimeout(() => {
      managerCloseRef.current?.focus();
    }, 0);
    setSettingsOpen(false);
    setHistoryOpen(false);
    setLocalFluxOpen(false);

    function handleDrawerKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") {
        event.preventDefault();
        setManagerOpen(false);
      }
    }

    window.addEventListener("keydown", handleDrawerKeyDown);
    return () => {
      window.removeEventListener("keydown", handleDrawerKeyDown);
      previousFocus?.focus();
    };
  }, [managerOpen]);

  useEffect(() => {
    if (!localFluxOpen) {
      return;
    }

    const previousFocus =
      document.activeElement instanceof HTMLElement
        ? document.activeElement
        : localFluxTriggerRef.current;

    window.setTimeout(() => {
      localFluxCloseRef.current?.focus();
    }, 0);
    setSettingsOpen(false);
    setHistoryOpen(false);
    setManagerOpen(false);

    function handleDrawerKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") {
        event.preventDefault();
        setLocalFluxOpen(false);
        return;
      }

      if (event.key !== "Tab") {
        return;
      }

      const drawer = localFluxDrawerRef.current;
      if (!drawer) {
        return;
      }

      const focusable = Array.from(
        drawer.querySelectorAll<HTMLElement>(
          'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])',
        ),
      ).filter((element) => element.offsetParent !== null);

      if (focusable.length === 0) {
        return;
      }

      const first = focusable[0];
      const last = focusable[focusable.length - 1];

      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    }

    window.addEventListener("keydown", handleDrawerKeyDown);
    return () => {
      window.removeEventListener("keydown", handleDrawerKeyDown);
      previousFocus?.focus();
    };
  }, [localFluxOpen]);

  async function updateCodexDefaultModel(model: string) {
    setCodexModels((current) => ({ ...current, default_model: model }));
    try {
      const response = await fetch(`${backendBaseUrl}/providers/codex/default-model`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ default_model: model }),
      });
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      setCodexModels(codexModelsResponseSchema.parse(await response.json()));
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : "無法更新 Codex 模型");
    }
  }

  async function updateCodexRuntimeOptions(patch: Partial<CodexModelsResponse>) {
    setCodexModels((current) => ({ ...current, ...patch }));
    try {
      const response = await fetch(`${backendBaseUrl}/providers/codex/runtime-options`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          default_model: patch.default_model,
          default_reasoning_effort: patch.default_reasoning_effort,
          default_reasoning_summary: patch.default_reasoning_summary,
          default_verbosity: patch.default_verbosity,
        }),
      });
      if (!response.ok) {
        throw new Error(await readErrorMessage(response));
      }
      setCodexModels(codexModelsResponseSchema.parse(await response.json()));
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : "無法更新 Codex 設定");
    }
  }

  async function updateOllamaDefaultModel(model: string) {
    const selectedModel = model || null;
    setOllamaModels((current) => ({ ...current, selected_model: selectedModel }));
    try {
      const response = await fetch(`${backendBaseUrl}/providers/ollama/default-model`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ default_model: selectedModel }),
      });
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      setOllamaModels(ollamaModelsResponseSchema.parse(await response.json()));
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : "無法更新 Ollama 模型");
    }
  }

  function agentProviderPayload() {
    if (agentProvider === "codex_cli") {
      return {
        provider: agentProvider,
        codex_model: codexModels.default_model,
        codex_reasoning_effort: codexModels.default_reasoning_effort,
        codex_reasoning_summary: codexModels.default_reasoning_summary,
        codex_verbosity: codexModels.default_verbosity,
      };
    }

    return {
      provider: agentProvider,
      ollama_model: ollamaModels.selected_model,
    };
  }

  function agentContextPayload() {
    return {
      include_original_prompt_context: includeOriginalPromptContext,
      include_optimized_prompt_context: includeOptimizedPromptContext,
      template_id: selectedTemplateId || null,
    };
  }

  async function updateSafeSettings(
    patch: Partial<
      Pick<SafeSettingsResponse, "selected_agent_provider" | "selected_image_provider">
    >,
  ) {
    try {
      const response = await fetch(`${backendBaseUrl}/settings/safe`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(patch),
      });
      if (!response.ok) {
        throw new Error(await readErrorMessage(response));
      }
      const payload = safeSettingsResponseSchema.parse(await response.json());
      setAgentProvider(payload.selected_agent_provider);
      setImageProvider(payload.selected_image_provider);
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : "無法儲存安全設定");
    }
  }

  async function refreshProviderSettings() {
    setSettingsMessage(null);
    try {
      const [codexModelsPayload, ollamaModelsPayload, codexPayload, ollamaPayload, localFluxPayload] =
        await Promise.all([
          fetchJson("/providers/codex/models", codexModelsResponseSchema),
          fetchJson("/providers/ollama/models", ollamaModelsResponseSchema),
          fetchJson("/providers/codex/status", codexStatusResponseSchema),
          fetchJson("/providers/ollama/status", ollamaStatusResponseSchema),
          fetchJson("/providers/local-flux/status", localFluxStatusResponseSchema),
        ]);

      setCodexModels(codexModelsPayload);
      setOllamaModels(ollamaModelsPayload);
      setCodexStatus(codexPayload);
      setOllamaStatus(ollamaPayload);
      setLocalFluxStatus(localFluxPayload);
      setSettingsMessage("已重新整理");
    } catch (error) {
      setSettingsMessage(error instanceof Error ? error.message : "重新整理失敗");
    }
  }

  async function saveCodexModelOptions() {
    const modelOptions = codexOptionsDraft
      .split(",")
      .map((item) => item.trim())
      .filter(Boolean);

    if (modelOptions.length === 0) {
      setSettingsMessage("Codex 模型清單不可為空");
      return;
    }

    try {
      const response = await fetch(`${backendBaseUrl}/providers/codex/model-options`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model_options: modelOptions }),
      });
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      const payload = codexModelsResponseSchema.parse(await response.json());
      setCodexModels(payload);
      setCodexOptionsDraft(payload.model_options.join(", "));
      setSettingsMessage("Codex 模型清單已儲存");
    } catch (error) {
      setSettingsMessage(error instanceof Error ? error.message : "儲存失敗");
    }
  }

  async function refreshSessionHistory() {
    setHistoryBusy(true);
    setHistoryMessage(null);
    try {
      const sessions = await fetchJson("/sessions", sessionsResponseSchema);
      setSessionHistory(sessions);
      setHistoryMessage(`已載入 ${sessions.length} 個工作階段`);
    } catch (error) {
      setHistoryMessage(error instanceof Error ? error.message : "無法載入歷史紀錄");
    } finally {
      setHistoryBusy(false);
    }
  }

  async function openHistoryDrawer() {
    setSettingsOpen(false);
    setManagerOpen(false);
    setLocalFluxOpen(false);
    setHistoryOpen(true);
    await refreshSessionHistory();
  }

  async function refreshPromptVersions(sessionId: string | null = currentSession?.session_id ?? null) {
    if (!sessionId) {
      setManagerPromptVersions([]);
      return;
    }
    const versions = await fetchJson(
      `/sessions/${sessionId}/prompt-versions`,
      promptVersionsResponseSchema,
    );
    setManagerPromptVersions(versions);
  }

  async function refreshManagerData() {
    setManagerBusy(true);
    setManagerMessage(null);
    try {
      const [models, skills, templates, proposals, logs, promptVersions] = await Promise.all([
        fetchJson("/models", modelInfoListResponseSchema),
        fetchJson("/skills", registryItemsResponseSchema),
        fetchJson("/templates", registryItemsResponseSchema),
        fetchJson("/registry/patch-proposals", registryPatchProposalsResponseSchema),
        fetchJson("/logs", logsResponseSchema),
        currentSession
          ? fetchJson(
              `/sessions/${currentSession.session_id}/prompt-versions`,
              promptVersionsResponseSchema,
            )
          : Promise.resolve([] as PromptVersionResponse[]),
      ]);
      setManagerModels(models);
      setManagerSkills(skills);
      setManagerTemplates(templates);
      setTemplates(templates);
      setRegistryProposals(proposals);
      setManagerLogs(logs);
      setManagerPromptVersions(promptVersions);
      if (!managerSelectedTemplateId && templates.length > 0) {
        setManagerSelectedTemplateId(templates[0].item_id);
        setTemplateEditorContent(templates[0].content);
      }
      if (!selectedProposalId && proposals.length > 0) {
        setSelectedProposalId(proposals[0].proposal_id);
        setProposalEditorContent(proposals[0].proposed_content ?? "");
        setProposalTargetId(proposals[0].item_id ?? "");
      }
      setManagerMessage(
        `已載入 ${models.length} models / ${skills.length} skills / ${templates.length} templates / ${logs.length} logs`,
      );
    } catch (error) {
      setManagerMessage(error instanceof Error ? error.message : "無法載入管理資料");
    } finally {
      setManagerBusy(false);
    }
  }

  async function openManagerDrawer() {
    setSettingsOpen(false);
    setHistoryOpen(false);
    setLocalFluxOpen(false);
    setManagerOpen(true);
    await refreshManagerData();
  }

  function selectManagerTemplate(template: RegistryItemResponse) {
    setManagerSelectedTemplateId(template.item_id);
    setTemplateEditorContent(template.content);
    setTemplateValidation(null);
    setTemplatePreview(null);
  }

  function selectRegistryProposal(proposal: RegistryPatchProposalResponse) {
    setSelectedProposalId(proposal.proposal_id);
    setProposalEditorContent(proposal.proposed_content ?? "");
    setProposalTargetId(proposal.item_id ?? "");
    setProposalKind(proposal.registry_kind);
    setProposalChangeKind(proposal.change_kind);
    setProposalValidation(null);
  }

  function startNewTemplate() {
    const id = `custom-template-${Date.now()}`;
    setManagerSelectedTemplateId("");
    setTemplateEditorContent(
      JSON.stringify(
        {
          id,
          name: "Custom Template",
          applies_to: ["t2i", "i2i"],
          description: "Describe when this template should be used.",
          questions: [
            {
              id: "subject",
              type: "text",
              label: "Subject",
              required: true,
            },
          ],
          prompt_structure: {
            must_include: ["subject", "composition", "lighting"],
            avoid: [],
          },
        },
        null,
        2,
      ),
    );
    setTemplateValidation(null);
    setTemplatePreview(null);
  }

  async function toggleSkillEnabled(skill: RegistryItemResponse, enabled: boolean) {
    setManagerBusy(true);
    setManagerMessage(null);
    try {
      const response = await fetch(`${backendBaseUrl}/skills/${skill.item_id}/enabled`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ enabled }),
      });
      if (!response.ok) {
        throw new Error(await readErrorMessage(response));
      }
      const updated = await response.json();
      setManagerSkills((current) =>
        current.map((item) => (item.item_id === skill.item_id ? registryItemResponseSchema.parse(updated) : item)),
      );
      setManagerMessage(`${skill.item_id} ${enabled ? "enabled" : "disabled"}`);
    } catch (error) {
      setManagerMessage(error instanceof Error ? error.message : "Skill 狀態更新失敗");
    } finally {
      setManagerBusy(false);
    }
  }

  async function validateTemplateEditor() {
    setManagerBusy(true);
    setManagerMessage(null);
    try {
      const response = await fetch(`${backendBaseUrl}/templates/validate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          content: templateEditorContent,
          sample_prompt: templatePreviewSamplePrompt || null,
          mode: workflowMode,
        }),
      });
      if (!response.ok) {
        throw new Error(await readErrorMessage(response));
      }
      const validation = templateValidationResponseSchema.parse(await response.json());
      setTemplateValidation(validation);
      setManagerMessage(validation.valid ? "Template 驗證通過" : validation.errors.join(" "));
    } catch (error) {
      setManagerMessage(error instanceof Error ? error.message : "Template 驗證失敗");
    } finally {
      setManagerBusy(false);
    }
  }

  async function previewTemplateEditor() {
    setManagerBusy(true);
    setManagerMessage(null);
    try {
      const response = await fetch(`${backendBaseUrl}/templates/preview`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          content: templateEditorContent,
          sample_prompt: templatePreviewSamplePrompt || null,
          mode: workflowMode,
        }),
      });
      if (!response.ok) {
        throw new Error(await readErrorMessage(response));
      }
      const preview = templatePreviewResponseSchema.parse(await response.json());
      setTemplatePreview(preview);
      setManagerMessage(preview.valid ? "Template preview 已更新" : preview.errors.join(" "));
    } catch (error) {
      setManagerMessage(error instanceof Error ? error.message : "Template preview 失敗");
    } finally {
      setManagerBusy(false);
    }
  }

  async function saveTemplateEditor() {
    setManagerBusy(true);
    setManagerMessage(null);
    try {
      const method = managerSelectedTemplateId ? "PUT" : "POST";
      const path = managerSelectedTemplateId
        ? `/templates/${managerSelectedTemplateId}`
        : "/templates";
      const response = await fetch(`${backendBaseUrl}${path}`, {
        method,
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ content: templateEditorContent }),
      });
      if (!response.ok) {
        throw new Error(await readErrorMessage(response));
      }
      const updated = registryItemResponseSchema.parse(await response.json());
      setManagerSelectedTemplateId(updated.item_id);
      setTemplateEditorContent(updated.content);
      setTemplateValidation(null);
      setTemplatePreview(null);
      await refreshManagerData();
      setSelectedTemplateId(updated.item_id);
      setManagerMessage("Template 已儲存");
    } catch (error) {
      setManagerMessage(error instanceof Error ? error.message : "Template 儲存失敗");
    } finally {
      setManagerBusy(false);
    }
  }

  async function duplicateSelectedTemplate() {
    if (!managerSelectedTemplateId) {
      setManagerMessage("請先選擇要複製的 template");
      return;
    }
    setManagerBusy(true);
    setManagerMessage(null);
    try {
      const response = await fetch(`${backendBaseUrl}/templates/${managerSelectedTemplateId}/duplicate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({}),
      });
      if (!response.ok) {
        throw new Error(await readErrorMessage(response));
      }
      const duplicated = registryItemResponseSchema.parse(await response.json());
      setManagerSelectedTemplateId(duplicated.item_id);
      setTemplateEditorContent(duplicated.content);
      await refreshManagerData();
      setSelectedTemplateId(duplicated.item_id);
      setManagerMessage("Template 已複製");
    } catch (error) {
      setManagerMessage(error instanceof Error ? error.message : "Template 複製失敗");
    } finally {
      setManagerBusy(false);
    }
  }

  async function setCurrentPromptVersion(version: PromptVersionResponse) {
    if (!currentSession) {
      setManagerMessage("請先選擇或建立 session");
      return;
    }
    setManagerBusy(true);
    setManagerMessage(null);
    try {
      const response = await fetch(
        `${backendBaseUrl}/sessions/${currentSession.session_id}/current-prompt-version`,
        {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ prompt_version_id: version.prompt_version_id }),
        },
      );
      if (!response.ok) {
        throw new Error(await readErrorMessage(response));
      }
      const updated = promptVersionResponseSchema.parse(await response.json());
      setOptimizedPrompt(updated.prompt_text);
      await refreshPromptVersions(currentSession.session_id);
      setManagerMessage(`已切換 prompt version：${updated.title ?? updated.prompt_version_id}`);
    } catch (error) {
      setManagerMessage(error instanceof Error ? error.message : "Prompt version 切換失敗");
    } finally {
      setManagerBusy(false);
    }
  }

  async function refreshRegistryProposals() {
    const proposals = await fetchJson(
      "/registry/patch-proposals",
      registryPatchProposalsResponseSchema,
    );
    setRegistryProposals(proposals);
    if (selectedProposalId && !proposals.some((item) => item.proposal_id === selectedProposalId)) {
      setSelectedProposalId("");
      setProposalEditorContent("");
      setProposalTargetId("");
    }
  }

  async function saveRegistryProposalDraft(): Promise<boolean> {
    if (!selectedProposal) {
      setManagerMessage("請先選擇 proposal");
      return false;
    }
    setManagerBusy(true);
    setManagerMessage(null);
    try {
      const response = await fetch(`${backendBaseUrl}/registry/patch-proposals/${selectedProposal.proposal_id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          proposed_content: proposalEditorContent,
          item_id: proposalTargetId || null,
          summary: selectedProposal.summary ?? "",
        }),
      });
      if (!response.ok) {
        throw new Error(await readErrorMessage(response));
      }
      const updated = registryPatchProposalResponseSchema.parse(await response.json());
      setRegistryProposals((current) =>
        current.map((item) => (item.proposal_id === updated.proposal_id ? updated : item)),
      );
      selectRegistryProposal(updated);
      setManagerMessage("Proposal draft 已儲存");
      return true;
    } catch (error) {
      setManagerMessage(error instanceof Error ? error.message : "Proposal draft 儲存失敗");
      return false;
    } finally {
      setManagerBusy(false);
    }
  }

  async function validateSelectedProposal() {
    if (!selectedProposal) {
      setManagerMessage("請先選擇 proposal");
      return;
    }
    setManagerBusy(true);
    setManagerMessage(null);
    try {
      if (!(await saveRegistryProposalDraft())) {
        return;
      }
      const response = await fetch(
        `${backendBaseUrl}/registry/patch-proposals/${selectedProposal.proposal_id}/validate`,
        { method: "POST" },
      );
      if (!response.ok) {
        throw new Error(await readErrorMessage(response));
      }
      const validation = registryPatchProposalValidationResponseSchema.parse(await response.json());
      setProposalValidation(validation);
      await refreshRegistryProposals();
      setManagerMessage(validation.valid ? "Proposal 驗證通過" : validation.errors.join(" "));
    } catch (error) {
      setManagerMessage(error instanceof Error ? error.message : "Proposal 驗證失敗");
    } finally {
      setManagerBusy(false);
    }
  }

  async function approveSelectedProposal() {
    if (!selectedProposal) {
      setManagerMessage("請先選擇 proposal");
      return;
    }
    setManagerBusy(true);
    setManagerMessage(null);
    try {
      if (!(await saveRegistryProposalDraft())) {
        return;
      }
      const response = await fetch(
        `${backendBaseUrl}/registry/patch-proposals/${selectedProposal.proposal_id}/approve`,
        { method: "POST" },
      );
      if (!response.ok) {
        throw new Error(await readErrorMessage(response));
      }
      const updated = registryPatchProposalResponseSchema.parse(await response.json());
      setRegistryProposals((current) =>
        current.map((item) => (item.proposal_id === updated.proposal_id ? updated : item)),
      );
      selectRegistryProposal(updated);
      await refreshManagerData();
      setManagerMessage("Proposal 已 approve，registry 已建立新版本");
    } catch (error) {
      setManagerMessage(error instanceof Error ? error.message : "Proposal approve 失敗");
    } finally {
      setManagerBusy(false);
    }
  }

  async function rejectSelectedProposal() {
    if (!selectedProposal) {
      setManagerMessage("請先選擇 proposal");
      return;
    }
    setManagerBusy(true);
    setManagerMessage(null);
    try {
      const response = await fetch(
        `${backendBaseUrl}/registry/patch-proposals/${selectedProposal.proposal_id}/reject`,
        { method: "POST" },
      );
      if (!response.ok) {
        throw new Error(await readErrorMessage(response));
      }
      const updated = registryPatchProposalResponseSchema.parse(await response.json());
      setRegistryProposals((current) =>
        current.map((item) => (item.proposal_id === updated.proposal_id ? updated : item)),
      );
      selectRegistryProposal(updated);
      setManagerMessage("Proposal 已 reject");
    } catch (error) {
      setManagerMessage(error instanceof Error ? error.message : "Proposal reject 失敗");
    } finally {
      setManagerBusy(false);
    }
  }

  async function createAgentRegistryProposal(
    kind: ProposalKind = proposalKind,
    instruction = proposalAuthoringInstruction,
    changeKind: ProposalChangeKind = proposalChangeKind,
  ) {
    const session = await ensureSession();
    setManagerBusy(true);
    setAgentMessage(`${activeAgentName} 正在建立 ${kind} proposal...`);
    setManagerMessage(null);
    try {
      const currentPromptVersion =
        managerPromptVersions.find((version) => version.is_current) ?? managerPromptVersions[0];
      const response = await fetch(`${backendBaseUrl}/agent/registry-proposals`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: session.session_id,
          proposal_kind: kind,
          change_kind: changeKind,
          authoring_instruction: instruction,
          mode: workflowMode,
          target_id: proposalTargetId || null,
          current_prompt_version_id: currentPromptVersion?.prompt_version_id ?? null,
          job_id: generationJob?.job_id ?? null,
          template_id: selectedTemplateId || null,
          ...agentProviderPayload(),
        }),
      });
      if (!response.ok) {
        throw new Error(await readErrorMessage(response));
      }
      const proposal = registryPatchProposalResponseSchema.parse(await response.json());
      setRegistryProposals((current) => [proposal, ...current.filter((item) => item.proposal_id !== proposal.proposal_id)]);
      selectRegistryProposal(proposal);
      setManagerTab("proposals");
      setManagerOpen(true);
      setManagerMessage("Agent proposal 已建立，請在 Proposals tab 檢查後 approve 或 reject");
    } catch (error) {
      setManagerMessage(error instanceof Error ? error.message : "Agent proposal 建立失敗");
      setErrorMessage(error instanceof Error ? error.message : "Agent proposal 建立失敗");
    } finally {
      setManagerBusy(false);
      setAgentMessage(null);
    }
  }

  function updateLocalFluxDraft<K extends keyof LocalFluxSettingsResponse>(
    key: K,
    value: LocalFluxSettingsResponse[K],
  ) {
    setLocalFluxDraft((current) => ({ ...current, [key]: value }));
  }

  async function refreshLocalFluxSettings() {
    setLocalFluxBusy(true);
    setLocalFluxMessage(null);
    try {
      const [settingsPayload, statusPayload] = await Promise.all([
        fetchJson("/providers/local-flux/settings", localFluxSettingsResponseSchema),
        fetchJson("/providers/local-flux/status", localFluxStatusResponseSchema),
      ]);
      setLocalFluxDraft(settingsPayload);
      setLocalFluxStatus(statusPayload);
      setLocalFluxMessage(statusPayload.message);
    } catch (error) {
      setLocalFluxMessage(error instanceof Error ? error.message : "無法載入 Local Flux 設定");
    } finally {
      setLocalFluxBusy(false);
    }
  }

  async function openLocalFluxDrawer() {
    setSettingsOpen(false);
    setHistoryOpen(false);
    setManagerOpen(false);
    setLocalFluxOpen(true);
    await refreshLocalFluxSettings();
  }

  async function saveLocalFluxSettings() {
    setLocalFluxBusy(true);
    setLocalFluxMessage(null);
    try {
      const response = await fetch(`${backendBaseUrl}/providers/local-flux/settings`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          base_url: localFluxDraft.base_url,
          workflow_path: localFluxDraft.workflow_path,
          i2i_one_workflow_path: localFluxDraft.i2i_one_workflow_path,
          i2i_two_workflow_path: localFluxDraft.i2i_two_workflow_path,
          model_path: localFluxDraft.model_path,
          vae_path: localFluxDraft.vae_path,
          text_encoder_path: localFluxDraft.text_encoder_path,
          width: localFluxDraft.width,
          height: localFluxDraft.height,
          seed: localFluxDraft.seed,
          steps: localFluxDraft.steps,
          cfg: localFluxDraft.cfg,
          sampler_name: localFluxDraft.sampler_name,
          scheduler: localFluxDraft.scheduler,
          denoise: localFluxDraft.denoise,
          guidance: localFluxDraft.guidance,
          output_prefix: localFluxDraft.output_prefix,
          timeout_seconds: localFluxDraft.timeout_seconds,
        }),
      });
      if (!response.ok) {
        throw new Error(await readErrorMessage(response));
      }
      const payload = localFluxSettingsResponseSchema.parse(await response.json());
      setLocalFluxDraft(payload);
      setLocalFluxMessage("Local Flux 設定已儲存。");
    } catch (error) {
      setLocalFluxMessage(error instanceof Error ? error.message : "Local Flux 設定儲存失敗");
    } finally {
      setLocalFluxBusy(false);
    }
  }

  async function testLocalFluxConnection() {
    setLocalFluxBusy(true);
    setLocalFluxMessage(null);
    try {
      await saveLocalFluxSettings();
      const status = await fetchJson("/providers/local-flux/status", localFluxStatusResponseSchema);
      setLocalFluxStatus(status);
      setLocalFluxMessage(status.message);
    } catch (error) {
      setLocalFluxMessage(error instanceof Error ? error.message : "Local Flux 連線測試失敗");
    } finally {
      setLocalFluxBusy(false);
    }
  }

  async function validateLocalFluxWorkflow(mode: WorkflowMode) {
    setLocalFluxBusy(true);
    setLocalFluxMessage(null);
    const workflowPath =
      mode === "t2i"
        ? localFluxDraft.workflow_path
        : localFluxDraft.i2i_two_workflow_path || localFluxDraft.i2i_one_workflow_path;
    try {
      const response = await fetch(`${backendBaseUrl}/providers/local-flux/workflows/validate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          workflow_path: workflowPath,
          mode,
          reference_count: mode === "i2i" ? 1 : 0,
        }),
      });
      if (!response.ok) {
        throw new Error(await readErrorMessage(response));
      }
      const payload = localFluxWorkflowValidationResponseSchema.parse(await response.json());
      setLocalFluxMessage(
        payload.valid
          ? `${payload.message} (${payload.workflow_format})`
          : `${payload.message} ${payload.missing_bindings.join(", ")}`,
      );
    } catch (error) {
      setLocalFluxMessage(error instanceof Error ? error.message : "Workflow 驗證失敗");
    } finally {
      setLocalFluxBusy(false);
    }
  }

  async function chooseLocalFluxPath(field: LocalFluxPathField) {
    if (!isTauriRuntime()) {
      setLocalFluxMessage("檔案選擇器只在 Tauri 桌面版可用；目前瀏覽器預覽請手動貼上路徑。");
      return;
    }

    setLocalFluxPickerBusy(field);
    setLocalFluxMessage(null);
    try {
      const { open } = await import("@tauri-apps/plugin-dialog");
      const selected = await open({
        directory: false,
        multiple: false,
        title: "選擇 Local Flux 檔案",
        defaultPath: localFluxDraft[field] || undefined,
      });
      const value = typeof selected === "string" ? selected : Array.isArray(selected) ? selected[0] : null;
      if (typeof value === "string") {
        updateLocalFluxDraft(field, value);
        setLocalFluxMessage("已選擇路徑，請按儲存設定。");
      } else {
        setLocalFluxMessage("已取消選擇路徑。");
      }
    } catch {
      setLocalFluxMessage("無法開啟檔案選擇器；目前可手動輸入或貼上路徑。");
    } finally {
      setLocalFluxPickerBusy(null);
    }
  }

  async function loadSession(sessionId: string) {
    setHistoryBusy(true);
    setHistoryMessage(null);
    try {
      const session = await fetchJson(`/sessions/${sessionId}`, sessionResponseSchema);
      setCurrentSession(session);
      setReferenceImages(session.reference_images);
      setGenerationJob(null);
      setFeedbackQuestionnaireContext(null);
      setAgentTurn(null);
      setAnswerDrafts({});
      await refreshPromptVersions(session.session_id);
      setAgentMessage(`已載入工作階段 ${session.session_id}`);
      setHistoryOpen(false);
    } catch (error) {
      setHistoryMessage(error instanceof Error ? error.message : "無法載入工作階段");
    } finally {
      setHistoryBusy(false);
    }
  }

  async function createSession(): Promise<SessionResponse> {
    const response = await fetch(`${backendBaseUrl}/sessions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title: "本機工作階段" }),
    });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    const session = sessionResponseSchema.parse(await response.json());
    setCurrentSession(session);
    setReferenceImages(session.reference_images);
    return session;
  }

  async function ensureSession(): Promise<SessionResponse> {
    return currentSession ?? createSession();
  }

  async function uploadReferenceImage(slot: number, file: File | null) {
    if (!file) {
      return;
    }
    try {
      const session = await ensureSession();
      const formData = new FormData();
      formData.append("slot", String(slot));
      formData.append("role", slot === 1 ? "primary_reference" : "secondary_reference");
      formData.append("file", file);

      const response = await fetch(`${backendBaseUrl}/sessions/${session.session_id}/reference-images`, {
        method: "POST",
        body: formData,
      });
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      const referenceImage = referenceImageResponseSchema.parse(await response.json());
      setReferenceImages((current) =>
        [...current.filter((item) => item.slot !== referenceImage.slot), referenceImage].sort(
          (a, b) => a.slot - b.slot,
        ),
      );
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : "無法上傳參考圖片");
    }
  }

  function applyAgentTurn(turn: AgentTurnResponse) {
    setAgentTurn(turn);
    if (turn.kind === "questionnaire") {
      setAnswerDrafts((current) => {
        const defaults = defaultDraftsForQuestionnaire(turn.questionnaire);
        if (
          agentTurn?.kind === "questionnaire" &&
          agentTurn.questionnaire.questionnaire_id === turn.questionnaire.questionnaire_id
        ) {
          return { ...defaults, ...current };
        }
        return defaults;
      });
      setAgentMessage(turn.message);
      return;
    }
    if (turn.kind === "optimized_prompt") {
      setOptimizedPrompt(turn.optimized_prompt);
      setAgentMessage(turn.message);
      setFeedbackQuestionnaireContext(null);
      if (currentSession) {
        void refreshPromptVersions(currentSession.session_id);
      }
      return;
    }
    if (turn.kind === "error") {
      setAgentMessage(turn.error.message);
      setErrorMessage(turn.error.suggestion ? `${turn.error.message} ${turn.error.suggestion}` : turn.error.message);
      return;
    }
    setAgentMessage(turn.message);
  }

  async function runAgentTurn() {
    const prompt = originalPrompt.trim();
    if (!prompt) {
      setErrorMessage("請先輸入原始 Prompt。");
      return;
    }

    setAgentBusy(true);
    setAgentMessage(`${activeAgentName} 正在分析 prompt...`);
    setErrorMessage(null);
    setFeedbackQuestionnaireContext(null);
    try {
      const session = await ensureSession();
      const response = await fetch(`${backendBaseUrl}/agent/turn`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: session.session_id,
          original_prompt: prompt,
          mode: workflowMode,
          ...agentProviderPayload(),
          ...agentContextPayload(),
        }),
      });
      if (!response.ok) {
        throw new Error(await readErrorMessage(response));
      }
      applyAgentTurn(agentTurnResponseSchema.parse(await response.json()));
    } catch (error) {
      setAgentMessage("分析失敗");
      setErrorMessage(error instanceof Error ? error.message : "無法執行 agent 回合");
    } finally {
      setAgentBusy(false);
    }
  }

  async function submitQuestionnaire(questionnaire: Questionnaire) {
    if (!currentSession) {
      setErrorMessage("請先建立工作階段。");
      return;
    }

    const isFeedbackQuestionnaire =
      feedbackQuestionnaireContext?.questionnaireId === questionnaire.questionnaire_id;
    setAgentBusy(true);
    setAgentMessage(
      isFeedbackQuestionnaire
        ? `${activeAgentName} 正在根據回饋修正 prompt...`
        : `${activeAgentName} 正在整理答案並最佳化 prompt...`,
    );
    setErrorMessage(null);
    try {
      const answers = questionnaireAnswers(questionnaire, answerDrafts);
      const response = await fetch(
        `${backendBaseUrl}${isFeedbackQuestionnaire ? "/agent/refine" : "/agent/answer-questionnaire"}`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            session_id: currentSession.session_id,
            questionnaire_id: questionnaire.questionnaire_id,
            ...(isFeedbackQuestionnaire ? { job_id: feedbackQuestionnaireContext.jobId } : {}),
            mode: workflowMode,
            ...agentProviderPayload(),
            ...agentContextPayload(),
            answers,
          }),
        },
      );
      if (!response.ok) {
        throw new Error(await readErrorMessage(response));
      }
      const turn = agentTurnResponseSchema.parse(await response.json());
      applyAgentTurn(turn);
    } catch (error) {
      setAgentMessage("問卷送出失敗");
      setErrorMessage(error instanceof Error ? error.message : "無法送出問卷答案");
    } finally {
      setAgentBusy(false);
    }
  }

  async function requestFeedbackQuestionnaire(job: GenerationJobResponse) {
    if (job.status !== "succeeded" || job.images.length === 0) {
      return;
    }
    if (feedbackQuestionnaireContext && agentTurn?.kind === "questionnaire") {
      setAgentMessage("回饋問卷已保留；你可以繼續試 seed，準備好再送出目前問卷。");
      return;
    }

    setAgentBusy(true);
    setAgentMessage(`生成完成，${activeAgentName} 正在建立回饋問卷...`);
    setErrorMessage(null);
    try {
      const response = await fetch(`${backendBaseUrl}/agent/feedback-questionnaire`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: job.session_id,
          job_id: job.job_id,
          ...agentProviderPayload(),
          ...agentContextPayload(),
        }),
      });
      if (!response.ok) {
        throw new Error(await readErrorMessage(response));
      }
      const turn = agentTurnResponseSchema.parse(await response.json());
      applyAgentTurn(turn);
      if (turn.kind === "questionnaire") {
        setFeedbackQuestionnaireContext({
          questionnaireId: turn.questionnaire.questionnaire_id,
          jobId: job.job_id,
        });
      } else {
        setFeedbackQuestionnaireContext(null);
      }
    } catch (error) {
      setAgentMessage("回饋問卷建立失敗");
      setErrorMessage(error instanceof Error ? error.message : "無法建立生成後回饋問卷");
    } finally {
      setAgentBusy(false);
    }
  }

  async function confirmGeneration() {
    if (!optimizedPrompt.trim()) {
      setErrorMessage("請先取得最佳化 Prompt。");
      return;
    }

    const sortedReferenceImages = [...referenceImages].sort((a, b) => a.slot - b.slot);
    if (
      imageProvider === "local_flux" &&
      workflowMode === "i2i" &&
      sortedReferenceImages.length === 0
    ) {
      setErrorMessage("Local Flux I2I 至少需要 1 張參考圖片。");
      return;
    }

    setGenerationBusy(true);
    setGenerationMessage("正在確認並生成圖片...");
    setErrorMessage(null);
    try {
      const session = await ensureSession();
      const seedValue = localFluxDraft.seed;
      const response = await fetch(`${backendBaseUrl}/generation/confirm`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: session.session_id,
          provider: imageProvider,
          mode: workflowMode,
          original_prompt: originalPrompt,
          optimized_prompt: optimizedPrompt,
          parameters: {
            steps: localFluxDraft.steps,
            guidance: localFluxDraft.guidance,
            seed: seedValue === null ? null : Number.isFinite(seedValue) ? seedValue : null,
            lora_name: selectedLora,
            lora_weight: loraWeight,
          },
          reference_image_ids: sortedReferenceImages.map((image) => image.reference_image_id),
          codex_model: codexModels.default_model,
          codex_reasoning_effort: codexModels.default_reasoning_effort,
          codex_reasoning_summary: codexModels.default_reasoning_summary,
          codex_verbosity: codexModels.default_verbosity,
        }),
      });
      if (!response.ok) {
        throw new Error(await readErrorMessage(response));
      }
      const job = generationJobResponseSchema.parse(await response.json());
      setGenerationJob(job);
      setGenerationMessage(`生成工作狀態：${job.status}`);
      if (job.error) {
        setErrorMessage(job.error.suggestion ? `${job.error.message} ${job.error.suggestion}` : job.error.message);
      }
    } catch (error) {
      setGenerationMessage("建立生成工作失敗");
      setErrorMessage(error instanceof Error ? error.message : "無法建立生成工作");
    } finally {
      setGenerationBusy(false);
    }
  }

  async function cancelGeneration() {
    if (!generationJob) {
      return;
    }
    setGenerationBusy(true);
    setErrorMessage(null);
    try {
      const response = await fetch(`${backendBaseUrl}/generation/cancel`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ job_id: generationJob.job_id }),
      });
      if (!response.ok) {
        throw new Error(await readErrorMessage(response));
      }
      const job = generationJobResponseSchema.parse(await response.json());
      setGenerationJob(job);
      setGenerationMessage(`生成工作狀態：${job.status}`);
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : "無法取消生成工作");
    } finally {
      setGenerationBusy(false);
    }
  }

  useEffect(() => {
    const controller = new AbortController();

    async function checkBackend() {
      setBackendStatus("checking");
      setErrorMessage(null);

      try {
        const healthPayload = await fetchJson("/health", healthResponseSchema, controller.signal);
        setHealth(healthPayload);
        setBackendStatus("connected");

        const [
          settingsPayload,
          codexPayload,
          codexModelsPayload,
          ollamaPayload,
          ollamaModelsPayload,
          localFluxPayload,
          secretsPayload,
          templatesPayload,
        ] = await Promise.allSettled([
          fetchJson("/settings/safe", safeSettingsResponseSchema, controller.signal),
          fetchJson("/providers/codex/status", codexStatusResponseSchema, controller.signal),
          fetchJson("/providers/codex/models", codexModelsResponseSchema, controller.signal),
          fetchJson("/providers/ollama/status", ollamaStatusResponseSchema, controller.signal),
          fetchJson("/providers/ollama/models", ollamaModelsResponseSchema, controller.signal),
          fetchJson("/providers/local-flux/status", localFluxStatusResponseSchema, controller.signal),
          fetchJson("/security/secrets/status", secretStatusResponseSchema, controller.signal),
          fetchJson("/templates", registryItemsResponseSchema, controller.signal),
        ]);

        if (settingsPayload.status === "fulfilled") {
          setAgentProvider(settingsPayload.value.selected_agent_provider);
          setImageProvider(settingsPayload.value.selected_image_provider);
        }
        if (codexPayload.status === "fulfilled") {
          setCodexStatus(codexPayload.value);
        }
        if (codexModelsPayload.status === "fulfilled") {
          setCodexModels(codexModelsPayload.value);
        }
        if (ollamaPayload.status === "fulfilled") {
          setOllamaStatus(ollamaPayload.value);
        }
        if (ollamaModelsPayload.status === "fulfilled") {
          setOllamaModels(ollamaModelsPayload.value);
        }
        if (localFluxPayload.status === "fulfilled") {
          setLocalFluxStatus(localFluxPayload.value);
        }
        if (secretsPayload.status === "fulfilled") {
          setSecretStatus(secretsPayload.value);
        }
        if (templatesPayload.status === "fulfilled") {
          setTemplates(templatesPayload.value);
        }
      } catch (error) {
        if (controller.signal.aborted) {
          return;
        }
        setHealth(null);
        setCodexStatus(null);
        setOllamaStatus(null);
        setOllamaModels(fallbackOllamaModels);
        setLocalFluxStatus(null);
        setSecretStatus(null);
        setBackendStatus("disconnected");
        setErrorMessage(error instanceof Error ? error.message : "無法連線到後端");
      }
    }

    void checkBackend();
    const timer = window.setInterval(() => {
      void checkBackend();
    }, 10000);

    return () => {
      controller.abort();
      window.clearInterval(timer);
    };
  }, []);

  const statusLabel = useMemo(() => {
    if (backendStatus === "connected") {
      return "後端已連線";
    }
    if (backendStatus === "checking") {
      return "檢查後端";
    }
    return "後端未連線";
  }, [backendStatus]);

  const generatedImages =
    generationJob && generationJob.images.length > 0
      ? generationJob.images
      : (currentSession?.generated_images ?? []);
  const generatedPreviewImage =
    generatedImages.length > 0 ? generatedImages[generatedImages.length - 1] : null;
  const generatedPreviewStatus = generationJob?.status ?? "history";

  async function downloadGeneratedImage() {
    if (!generatedPreviewImage) {
      return;
    }

    setErrorMessage(null);
    try {
      const response = await fetch(
        backendFileUrl(generatedPreviewImage.url, generatedPreviewImage.image_id),
        { cache: "no-store" },
      );
      if (!response.ok) {
        throw new Error(await readErrorMessage(response));
      }

      const blob = await response.blob();
      const filename = generatedPreviewImage.filename || `${generatedPreviewImage.image_id}.png`;

      if (isTauriRuntime()) {
        const [{ save }, { invoke }] = await Promise.all([
          import("@tauri-apps/plugin-dialog"),
          import("@tauri-apps/api/core"),
        ]);
        const targetPath = await save({
          defaultPath: filename,
          filters: [{ name: "Images", extensions: ["png", "jpg", "jpeg", "webp"] }],
        });
        if (!targetPath) {
          setGenerationMessage("已取消下載。");
          return;
        }
        const bytes = Array.from(new Uint8Array(await blob.arrayBuffer()));
        await invoke("save_binary_file", { path: targetPath, bytes });
        setGenerationMessage(`已下載圖片：${filename}`);
        return;
      }

      const objectUrl = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = objectUrl;
      anchor.download = filename;
      document.body.append(anchor);
      anchor.click();
      anchor.remove();
      window.setTimeout(() => URL.revokeObjectURL(objectUrl), 0);
      setGenerationMessage("已開始下載圖片。");
    } catch (error) {
      setGenerationMessage("圖片下載失敗。");
      setErrorMessage(error instanceof Error ? error.message : "無法下載圖片");
    }
  }

  function setAnswerDraft(questionId: string, value: AnswerDraftValue) {
    setAnswerDrafts((current) => ({ ...current, [questionId]: value }));
  }

  function renderQuestionInput(question: Question) {
    const draft = answerDrafts[question.question_id];
    if (question.kind === "text") {
      return (
        <textarea
          className="question-textarea"
          maxLength={question.max_length ?? undefined}
          placeholder={question.placeholder ?? "輸入答案"}
          value={typeof draft === "string" ? draft : ""}
          onChange={(event) => setAnswerDraft(question.question_id, event.target.value)}
        />
      );
    }

    if (question.kind === "choice" && question.allow_multiple) {
      const values = Array.isArray(draft) ? draft : [];
      return (
        <div className="choice-list">
          {question.options.map((option) => (
            <label className="choice-row" key={option.value}>
              <input
                type="checkbox"
                checked={values.includes(option.value)}
                onChange={(event) => {
                  const nextValues = event.target.checked
                    ? [...values, option.value]
                    : values.filter((value) => value !== option.value);
                  setAnswerDraft(question.question_id, nextValues);
                }}
              />
              <span>
                <strong>{option.label}</strong>
                {option.description ? <small>{option.description}</small> : null}
              </span>
            </label>
          ))}
        </div>
      );
    }

    if (question.kind === "choice") {
      return (
        <select
          value={typeof draft === "string" ? draft : ""}
          onChange={(event) => setAnswerDraft(question.question_id, event.target.value)}
        >
          <option value="">請選擇</option>
          {question.options.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
      );
    }

    if (question.kind === "boolean") {
      return (
        <select
          value={draft === true ? "true" : "false"}
          onChange={(event) => setAnswerDraft(question.question_id, event.target.value === "true")}
        >
          <option value="true">{question.true_label}</option>
          <option value="false">{question.false_label}</option>
        </select>
      );
    }

    const value = typeof draft === "number" ? draft : question.min_value;
    return (
      <div className="scale-control">
        <input
          type="range"
          min={question.min_value}
          max={question.max_value}
          step={question.step}
          value={value}
          onChange={(event) => setAnswerDraft(question.question_id, Number(event.target.value))}
        />
        <strong>{value}</strong>
      </div>
    );
  }

  function renderLocalFluxPathField(
    field: LocalFluxPathField,
    label: string,
    placeholder: string,
  ) {
    return (
      <label>
        {label}
        <div className="manager-path-row">
          <input
            value={localFluxDraft[field]}
            placeholder={placeholder}
            onChange={(event) => updateLocalFluxDraft(field, event.target.value)}
          />
          <button
            className="command-button"
            type="button"
            disabled={localFluxBusy || localFluxPickerBusy !== null}
            onClick={() => {
              void chooseLocalFluxPath(field);
            }}
          >
            <FolderOpen size={16} aria-hidden="true" />
            {localFluxPickerBusy === field ? "開啟中" : "選擇"}
          </button>
        </div>
      </label>
    );
  }

  return (
    <main className="app-shell">
      <a className="skip-link" href="#workspace-main">
        跳到工作區
      </a>
      <header className="top-bar">
        <div className="brand-block">
          <span className="brand-kicker">Local First</span>
          <h1>Prompt Optimizer Studio</h1>
        </div>

        <div className="toolbar" aria-label="工作階段設定">
          <label>
            模式
            <select
              value={workflowMode}
              onChange={(event) => setWorkflowMode(event.target.value as WorkflowMode)}
            >
              <option value="t2i">T2I</option>
              <option value="i2i">I2I</option>
            </select>
          </label>
          <label>
            Agent
            <select
              value={agentProvider}
              onChange={(event) => {
                const nextProvider = event.target.value as AgentProvider;
                setAgentProvider(nextProvider);
                void updateSafeSettings({ selected_agent_provider: nextProvider });
              }}
            >
              <option value="codex_cli">Codex CLI</option>
              <option value="ollama_local_llm">Ollama</option>
            </select>
          </label>
          {isCodexAgent ? (
            <>
              <label>
                Codex 模型
                <select
                  value={codexModels.default_model}
                  onChange={(event) => {
                    void updateCodexDefaultModel(event.target.value);
                  }}
                >
                  {codexModels.model_options.map((model) => (
                    <option key={model} value={model}>
                      {model}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                思考強度
                <select
                  value={codexModels.default_reasoning_effort}
                  onChange={(event) => {
                    void updateCodexRuntimeOptions({
                      default_reasoning_effort: event.target
                        .value as CodexModelsResponse["default_reasoning_effort"],
                    });
                  }}
                >
                  {codexModels.reasoning_effort_options.map((effort) => (
                    <option key={effort} value={effort}>
                      {effort}
                    </option>
                  ))}
                </select>
              </label>
            </>
          ) : (
          <label>
            Ollama 模型
            <select
              value={ollamaModels.selected_model ?? ""}
              disabled={ollamaModels.models.length === 0}
              onChange={(event) => {
                void updateOllamaDefaultModel(event.target.value);
              }}
            >
              {ollamaModels.models.length === 0 ? (
                <option value="">未偵測到模型</option>
              ) : (
                ollamaModels.models.map((model) => (
                  <option key={model} value={model}>
                    {model}
                  </option>
                ))
              )}
            </select>
          </label>
          )}
          <label>
            圖像 Provider
            <select
              value={imageProvider}
              onChange={(event) => {
                const nextProvider = event.target.value as ImageProvider;
                setImageProvider(nextProvider);
                void updateSafeSettings({ selected_image_provider: nextProvider });
              }}
            >
              <option value="codex_cli_gpt_image">Codex Image</option>
              <option value="local_flux">Local Flux</option>
            </select>
          </label>
          {imageProvider === "local_flux" && (
            <>
              <label>
                LoRA
                <select
                  value={selectedLora ?? ""}
                  onChange={(event) => setSelectedLora(event.target.value || null)}
                >
                  <option value="">None</option>
                  {loraList.map((name) => (
                    <option key={name} value={name}>
                      {name}
                    </option>
                  ))}
                </select>
              </label>
              {selectedLora && (
                <label>
                  權重 {loraWeight.toFixed(2)}
                  <input
                    type="range"
                    min={-5}
                    max={5}
                    step={0.05}
                    value={loraWeight}
                    onChange={(event) => setLoraWeight(Number(event.target.value))}
                  />
                </label>
              )}
            </>
          )}
        </div>

        <div className="top-status">
          <div
            className={`connection-pill connection-pill--${backendStatus}`}
            aria-live="polite"
            aria-atomic="true"
          >
            <span aria-hidden="true" />
            {statusLabel}
          </div>
          <button
            ref={historyTriggerRef}
            className="icon-button"
            type="button"
            title="歷史"
            aria-label="開啟歷史紀錄"
            aria-haspopup="dialog"
            aria-expanded={historyOpen}
            onClick={() => {
              void openHistoryDrawer();
            }}
          >
            <History size={17} aria-hidden="true" />
          </button>
          <button
            ref={managerTriggerRef}
            className="icon-button"
            type="button"
            title="管理"
            aria-label="開啟管理抽屜"
            aria-haspopup="dialog"
            aria-expanded={managerOpen}
            onClick={() => {
              void openManagerDrawer();
            }}
          >
            <Database size={17} aria-hidden="true" />
          </button>
          <button
            ref={localFluxTriggerRef}
            className="icon-button"
            type="button"
            title="Local Flux 設定"
            aria-label="開啟 Local Flux 設定"
            aria-haspopup="dialog"
            aria-expanded={localFluxOpen}
            onClick={() => {
              void openLocalFluxDrawer();
            }}
          >
            <SlidersHorizontal size={17} aria-hidden="true" />
          </button>
          <button
            ref={settingsTriggerRef}
            className="icon-button"
            type="button"
            title="設定"
            aria-label="開啟設定"
            aria-haspopup="dialog"
            aria-expanded={settingsOpen}
            onClick={() => {
              setHistoryOpen(false);
              setManagerOpen(false);
              setLocalFluxOpen(false);
              setSettingsOpen(true);
            }}
          >
            <SettingsIcon size={17} aria-hidden="true" />
          </button>
        </div>
      </header>

      <section
        id="workspace-main"
        className="workspace-grid"
        aria-label="Prompt workflow workspace"
        tabIndex={-1}
      >
        <aside className="column column-left">
          <div className="column-heading">
            <h2>對話與問卷</h2>
            <p>Agent 將在這裡提出缺漏資訊與回饋問題。</p>
          </div>
          <div className="message-stream">
            <div className="message message-agent">
              <span>系統</span>
              <p>
                0B 已加入嚴格 schema、Codex CLI 狀態、Ollama 狀態與安全設定檢查。
              </p>
            </div>
            <div className="message message-status">
              <span>連線狀態</span>
              <p aria-live="polite" aria-atomic="true">
                {backendStatus === "connected" && health
                  ? `${health.app_name} ${health.version} / ${health.environment}`
                  : `API: ${backendBaseUrl}${errorMessage ? ` (${errorMessage})` : ""}`}
              </p>
            </div>
            <div className="provider-grid" aria-label="Provider 狀態">
              <div className="provider-card">
                <span>Codex CLI</span>
                <strong>{statusText(codexStatus?.available)}</strong>
                <p>{codexStatus?.version ?? codexStatus?.warning ?? codexStatus?.error ?? "等待檢查"}</p>
              </div>
              <div className="provider-card">
                <span>Ollama</span>
                <strong>{statusText(ollamaStatus?.available)}</strong>
                <p>
                  {ollamaStatus?.available
                    ? `${ollamaStatus.model_count} 個本機模型${
                        ollamaModels.selected_model ? ` / ${ollamaModels.selected_model}` : ""
                      }`
                    : (ollamaStatus?.error ?? "等待檢查")}
                </p>
              </div>
              <div className="provider-card">
                <span>Local Flux</span>
                <strong>{statusText(localFluxStatus?.available)}</strong>
                <p>{localFluxStatus?.message ?? "等待檢查"}</p>
              </div>
              <div className="provider-card">
                <span>HF_TOKEN</span>
                <strong>{secretStatus?.hf_token_configured ? "已設定" : "未設定"}</strong>
                <p>只顯示設定狀態，不回傳 token 值。</p>
              </div>
            </div>
            {agentMessage ? (
              <div className="message message-agent">
                <span>Agent</span>
                <p>{agentMessage}</p>
              </div>
            ) : null}
            {errorMessage ? (
              <div className="message message-error">
                <span>錯誤詳情</span>
                <p>{errorMessage}</p>
              </div>
            ) : null}
            {agentTurn?.kind === "questionnaire" ? (
              <section className="questionnaire-panel" aria-label="動態問卷">
                <div className="questionnaire-header">
                  <span>Questionnaire</span>
                  <h3>{agentTurn.questionnaire.title}</h3>
                  {agentTurn.questionnaire.description ? (
                    <p>{agentTurn.questionnaire.description}</p>
                  ) : null}
                </div>
                <div className="question-list">
                  {agentTurn.questionnaire.questions.map((question) => (
                    <label className="question-row" key={question.question_id}>
                      <span>
                        {question.label}
                        {question.required ? <b aria-label="必填">*</b> : null}
                      </span>
                      <small>{question.prompt}</small>
                      {renderQuestionInput(question)}
                    </label>
                  ))}
                </div>
                <button
                  className="command-button command-button-primary"
                  type="button"
                  disabled={agentBusy}
                  onClick={() => {
                    void submitQuestionnaire(agentTurn.questionnaire);
                  }}
                >
                  {feedbackQuestionnaireContext?.questionnaireId ===
                  agentTurn.questionnaire.questionnaire_id
                    ? "送出回饋並修正 Prompt"
                    : "送出問卷並最佳化"}
                </button>
              </section>
            ) : null}
            {agentTurn?.kind === "optimized_prompt" ? (
              <div className="message message-status">
                <span>最佳化完成</span>
                <p>{agentTurn.message}</p>
              </div>
            ) : null}
          </div>
          <div className="reference-strip">
            <div>
              <h3>參考圖片</h3>
              <p>
                {currentSession
                  ? `Session ${currentSession.session_id}`
                  : "先建立工作階段，或直接選圖自動建立。"}
              </p>
            </div>
            <div className="reference-slots" aria-label="參考圖片槽位">
              {[1, 2].map((slot) => {
                const referenceImage = referenceImages.find((item) => item.slot === slot);
                const imageUrl = referenceImage
                  ? `${backendBaseUrl}${referenceImage.thumbnail_url ?? referenceImage.url}`
                  : null;
                const inputId = `reference-upload-${slot}`;
                return (
                  <div className="reference-upload-wrap" key={slot}>
                    <input
                      id={inputId}
                      className="reference-upload-input"
                      type="file"
                      accept="image/*"
                      tabIndex={-1}
                      aria-hidden="true"
                      onChange={(event) => {
                        void uploadReferenceImage(slot, event.currentTarget.files?.[0] ?? null);
                        event.currentTarget.value = "";
                      }}
                    />
                    <label
                      className="reference-upload"
                      htmlFor={inputId}
                      role="button"
                      tabIndex={0}
                      aria-label={`上傳參考圖片 ${slot}`}
                      onKeyDown={(event) => {
                        if (event.key === "Enter" || event.key === " ") {
                          event.preventDefault();
                          document.getElementById(inputId)?.click();
                        }
                      }}
                    >
                      {imageUrl ? (
                        <img src={imageUrl} alt={`參考圖片 ${slot}`} />
                      ) : (
                        <span>Slot {slot}</span>
                      )}
                    </label>
                  </div>
                );
              })}
            </div>
            <button
              className="session-button"
              type="button"
              onClick={() => {
                void createSession().catch((error) => {
                  setErrorMessage(error instanceof Error ? error.message : "無法建立工作階段");
                });
              }}
            >
              建立新工作階段
            </button>
          </div>
        </aside>

        <section className="column column-middle">
          <div className="column-heading">
            <h2>Prompt 工作流</h2>
            <p>原始提示、最佳化結果與生成控制會集中在這裡。</p>
          </div>
          <div className="workflow-controls" aria-label="Prompt workflow controls">
            <label>
              Prompt 版本
              <select
                value={
                  managerPromptVersions.find((version) => version.is_current)?.prompt_version_id ??
                  ""
                }
                onChange={(event) => {
                  const version = managerPromptVersions.find(
                    (item) => item.prompt_version_id === event.target.value,
                  );
                  if (version) {
                    void setCurrentPromptVersion(version);
                  }
                }}
              >
                <option value="">目前草稿</option>
                {managerPromptVersions.map((version) => (
                  <option key={version.prompt_version_id} value={version.prompt_version_id}>
                    {promptVersionLabel(version)}
                  </option>
                ))}
              </select>
            </label>
            <label>
              模板
              <select
                value={selectedTemplateId}
                onChange={(event) => setSelectedTemplateId(event.target.value)}
              >
                <option value="">Auto-detect</option>
                {templates.map((template) => (
                  <option key={template.item_id} value={template.item_id}>
                    {templateName(template)}
                  </option>
                ))}
              </select>
            </label>
          </div>
          <label className="editor-block">
            原始 Prompt
            <textarea
              placeholder="輸入你想優化的影像提示..."
              value={originalPrompt}
              onChange={(event) => setOriginalPrompt(event.target.value)}
            />
          </label>
          <label className="editor-block editor-block-large">
            最佳化 Prompt
            <textarea
              placeholder="Agent 產生的最佳化提示會顯示在這裡。"
              value={optimizedPrompt}
              onChange={(event) => setOptimizedPrompt(event.target.value)}
            />
          </label>
          <div className="generation-panel">
            <div>
              <h3>生成控制</h3>
              <p>
                {workflowMode.toUpperCase()} / {imageProviderLabel(imageProvider)} / Seed{" "}
                {localFluxDraft.seed ?? "隨機"}
                {selectedTemplate ? ` / ${templateName(selectedTemplate)}` : " / Auto-detect"}
              </p>
              {generationMessage ? <p className="generation-note">{generationMessage}</p> : null}
              <div className="context-toggles" aria-label="Agent context">
                <label>
                  <input
                    type="checkbox"
                    checked={includeOriginalPromptContext}
                    onChange={(event) => setIncludeOriginalPromptContext(event.target.checked)}
                  />
                  原始 prompt 給 LLM
                </label>
                <label>
                  <input
                    type="checkbox"
                    checked={includeOptimizedPromptContext}
                    onChange={(event) => setIncludeOptimizedPromptContext(event.target.checked)}
                  />
                  上版最佳化 prompt 給 LLM
                </label>
              </div>
              {generationJob ? (
                <p className="generation-note">
                  Job {generationJob.job_id} / {generationJob.status}
                </p>
              ) : null}
            </div>
            <div className="workflow-actions">
              <button
                className="command-button command-button-primary"
                type="button"
                disabled={agentBusy || backendStatus !== "connected" || !originalPrompt.trim()}
                onClick={() => {
                  void runAgentTurn();
                }}
              >
                {agentBusy ? "Agent 執行中" : "分析提示"}
              </button>
              <button
                type="button"
                disabled={
                  generationBusy ||
                  backendStatus !== "connected" ||
                  !optimizedPrompt.trim()
                }
                onClick={() => {
                  void confirmGeneration();
                }}
              >
                {generationBusy ? "生成中" : "確認生成"}
              </button>
              <button
                type="button"
                disabled={
                  agentBusy ||
                  !generationJob ||
                  generationJob.status !== "succeeded" ||
                  generationJob.images.length === 0
                }
                onClick={() => {
                  if (generationJob) {
                    void requestFeedbackQuestionnaire(generationJob);
                  }
                }}
              >
                回饋 / 修正 Prompt
              </button>
              <button
                type="button"
                disabled={managerBusy || agentBusy || !currentSession || !optimizedPrompt.trim()}
                onClick={() => {
                  setProposalKind("template");
                  setProposalChangeKind("create");
                  void createAgentRegistryProposal(
                    "template",
                    "把目前 prompt、圖片結果與 feedback 經驗整理成可重用 template proposal",
                    "create",
                  );
                }}
              >
                轉成 Template 提案
              </button>
              <button
                type="button"
                disabled={managerBusy || agentBusy || !currentSession || !optimizedPrompt.trim()}
                onClick={() => {
                  setProposalKind("skill");
                  setProposalChangeKind("create");
                  void createAgentRegistryProposal(
                    "skill",
                    "把這次成功或失敗經驗整理成可重用 skill 規則 proposal",
                    "create",
                  );
                }}
              >
                轉成 Skill 提案
              </button>
              {generationJob && generationJob.status === "queued" ? (
                <button
                  type="button"
                  disabled={generationBusy}
                  onClick={() => {
                    void cancelGeneration();
                  }}
                >
                  取消
                </button>
              ) : null}
            </div>
          </div>
        </section>

        <aside className="column column-right">
          <div className="column-heading">
            <h2>圖像顯示</h2>
            <p>輸出圖片會透過後端安全 URL 顯示。</p>
          </div>
          <div className="image-stage">
            {generatedPreviewImage ? (
              <div className="generated-image-card">
                <img
                  src={backendFileUrl(
                    generatedPreviewImage.thumbnail_url ?? generatedPreviewImage.url,
                    `${generatedPreviewImage.image_id}-${generatedPreviewImage.created_at}`,
                  )}
                  alt="生成圖片"
                />
                <div className="generated-image-meta">
                  <div>
                    <strong>{generatedPreviewImage.filename}</strong>
                    <p>
                      {generatedPreviewImage.width} x {generatedPreviewImage.height} /{" "}
                      {generatedPreviewStatus} / Seed {generatedPreviewImage.seed ?? "n/a"}
                    </p>
                  </div>
                  <button
                    className="command-button"
                    type="button"
                    onClick={() => {
                      void downloadGeneratedImage();
                    }}
                  >
                    <Download size={16} aria-hidden="true" />
                    下載圖片
                  </button>
                </div>
              </div>
            ) : (
              <div className="image-placeholder">
                <span />
                <strong>{generationJob ? `生成工作：${generationJob.status}` : "尚未生成圖片"}</strong>
                <p>
                  {generationJob
                    ? "生成工作已建立；圖片完成後會透過安全 URL 顯示在這裡。"
                    : "完成問卷並確認生成後，結果會出現在這裡。"}
                </p>
              </div>
            )}
          </div>
        </aside>
      </section>

      {historyOpen ? (
        <div className="drawer-layer" role="presentation">
          <button
            className="drawer-scrim"
            type="button"
            aria-label="關閉歷史紀錄"
            onClick={() => setHistoryOpen(false)}
          />
          <aside
            ref={historyDrawerRef}
            className="settings-drawer"
            role="dialog"
            aria-modal="true"
            aria-labelledby="history-title"
          >
            <header className="drawer-header">
              <div>
                <span>History</span>
                <h2 id="history-title">工作階段歷史</h2>
              </div>
              <button
                ref={historyCloseRef}
                className="icon-button"
                type="button"
                title="關閉"
                aria-label="關閉歷史紀錄"
                onClick={() => setHistoryOpen(false)}
              >
                <X size={18} aria-hidden="true" />
              </button>
            </header>

            <div className="drawer-content">
              <section className="settings-section">
                <h3>Session</h3>
                <button
                  className="command-button"
                  type="button"
                  disabled={historyBusy}
                  onClick={() => {
                    void refreshSessionHistory();
                  }}
                >
                  <RefreshCcw size={16} aria-hidden="true" />
                  重新整理
                </button>
                <div className="history-list" aria-label="工作階段清單">
                  {sessionHistory.length === 0 ? (
                    <p>{historyBusy ? "正在載入..." : "尚無歷史工作階段"}</p>
                  ) : (
                    sessionHistory.map((session) => (
                      <button
                        className={`history-session${
                          currentSession?.session_id === session.session_id ? " is-active" : ""
                        }`}
                        type="button"
                        key={session.session_id}
                        disabled={historyBusy}
                        onClick={() => {
                          void loadSession(session.session_id);
                        }}
                      >
                        <strong>{session.title ?? "未命名工作階段"}</strong>
                        <span>
                          {formatSessionTime(session.created_at)} /{" "}
                          {session.reference_images.length} 參考 /{" "}
                          {session.generated_images.length} 生成
                        </span>
                        <small>{session.session_id}</small>
                      </button>
                    ))
                  )}
                </div>
              </section>
            </div>

            <footer className="drawer-footer">
              <History size={16} aria-hidden="true" />
              <span aria-live="polite" aria-atomic="true">
                {historyMessage ?? "選取一個工作階段即可載入參考圖與生成結果"}
              </span>
            </footer>
          </aside>
        </div>
      ) : null}

      {managerOpen ? (
        <div className="drawer-layer" role="presentation">
          <button
            className="drawer-scrim"
            type="button"
            aria-label="關閉管理抽屜"
            onClick={() => setManagerOpen(false)}
          />
          <aside
            className="settings-drawer"
            role="dialog"
            aria-modal="true"
            aria-labelledby="manager-title"
          >
            <header className="drawer-header">
              <div>
                <span>Manager</span>
                <h2 id="manager-title">管理中心</h2>
              </div>
              <button
                ref={managerCloseRef}
                className="icon-button"
                type="button"
                title="關閉"
                aria-label="關閉管理抽屜"
                onClick={() => setManagerOpen(false)}
              >
                <X size={18} aria-hidden="true" />
              </button>
            </header>

            <div className="drawer-content">
              <button
                className="command-button"
                type="button"
                disabled={managerBusy}
                onClick={() => {
                  void refreshManagerData();
                }}
              >
                <RefreshCcw size={16} aria-hidden="true" />
                {managerBusy ? "載入中" : "重新整理"}
              </button>

              <nav className="manager-tabs" aria-label="Manager tabs">
                {(["models", "skills", "templates", "prompt", "proposals", "logs"] as ManagerTab[]).map(
                  (tab) => (
                    <button
                      key={tab}
                      type="button"
                      className={managerTab === tab ? "is-active" : ""}
                      onClick={() => setManagerTab(tab)}
                    >
                      {tab}
                    </button>
                  ),
                )}
              </nav>

              <section className="settings-section" hidden={managerTab !== "models"}>
                <h3>Models</h3>
                <p>{managerBusy ? "正在載入模型狀態..." : `${managerModels.length} 個模型狀態`}</p>
                <div className="history-list" aria-label="模型狀態清單">
                  {managerModels.length === 0 ? (
                    <p>尚無模型狀態。</p>
                  ) : (
                    managerModels.map((model) => (
                      <div className="history-session" key={model.provider}>
                        <strong>{model.label}</strong>
                        <span>
                          {model.status}
                          {model.path_label ? ` / ${model.path_label}` : ""}
                        </span>
                        <small>{model.message ?? "等待下一步模型管理實作"}</small>
                      </div>
                    ))
                  )}
                </div>
              </section>
              <section className="settings-section" hidden={managerTab !== "skills"}>
                <h3>Skills</h3>
                <p>{managerBusy ? "正在載入技能..." : `${managerSkills.length} 個技能`}</p>
                <div className="history-list" aria-label="技能清單">
                  {managerSkills.length === 0 ? (
                    <p>尚無技能版本。</p>
                  ) : (
                    managerSkills.map((skill) => (
                      <div className="history-session" key={skill.item_id}>
                        <strong>{skill.item_id}</strong>
                        <label>
                          <input
                            type="checkbox"
                            checked={skill.enabled ?? true}
                            disabled={managerBusy}
                            onChange={(event) => {
                              void toggleSkillEnabled(skill, event.target.checked);
                            }}
                          />
                          {skill.enabled ?? true ? "Enabled" : "Disabled"}
                        </label>
                        <span>{skill.latest_version_id ?? "未建立版本"}</span>
                        <small>{skill.content}</small>
                      </div>
                    ))
                  )}
                </div>
              </section>
              <section className="settings-section" hidden={managerTab !== "templates"}>
                <h3>Templates</h3>
                <p>{managerBusy ? "正在載入模板..." : `${managerTemplates.length} 個模板`}</p>
                <div className="manager-actions" aria-label="Template 操作">
                  <button
                    className="command-button"
                    type="button"
                    disabled={managerBusy}
                    onClick={startNewTemplate}
                  >
                    <Plus size={16} aria-hidden="true" />
                    新增
                  </button>
                  <button
                    className="command-button"
                    type="button"
                    disabled={managerBusy || !managerSelectedTemplateId}
                    onClick={() => {
                      void duplicateSelectedTemplate();
                    }}
                  >
                    <Copy size={16} aria-hidden="true" />
                    複製
                  </button>
                  <button
                    className="command-button"
                    type="button"
                    disabled={managerBusy || !templateEditorContent.trim()}
                    onClick={() => {
                      void validateTemplateEditor();
                    }}
                  >
                    <RefreshCcw size={16} aria-hidden="true" />
                    驗證
                  </button>
                  <button
                    className="command-button"
                    type="button"
                    disabled={managerBusy || !templateEditorContent.trim()}
                    onClick={() => {
                      void previewTemplateEditor();
                    }}
                  >
                    <Eye size={16} aria-hidden="true" />
                    預覽
                  </button>
                  <button
                    className="command-button command-button-primary"
                    type="button"
                    disabled={managerBusy || !templateEditorContent.trim()}
                    onClick={() => {
                      void saveTemplateEditor();
                    }}
                  >
                    <Save size={16} aria-hidden="true" />
                    儲存
                  </button>
                </div>
                <div className="history-list" aria-label="模板清單">
                  {managerTemplates.length === 0 ? (
                    <p>尚無模板版本。</p>
                  ) : (
                    managerTemplates.map((template) => (
                      <button
                        className={`history-session${
                          managerSelectedTemplateId === template.item_id ? " is-active" : ""
                        }`}
                        key={template.item_id}
                        type="button"
                        onClick={() => selectManagerTemplate(template)}
                      >
                        <strong>{template.item_id}</strong>
                        <span>{template.latest_version_id ?? "未建立版本"}</span>
                        <small>{templateName(template)}</small>
                      </button>
                    ))
                  )}
                </div>
                <label>
                  Template JSON
                  <textarea
                    value={templateEditorContent}
                    rows={12}
                    placeholder="選擇 template 或新增一個 template。"
                    onChange={(event) => {
                      setTemplateEditorContent(event.target.value);
                      setTemplateValidation(null);
                      setTemplatePreview(null);
                    }}
                  />
                </label>
                <label>
                  Sample prompt
                  <input
                    value={templatePreviewSamplePrompt}
                    placeholder="用於 preview 的測試 prompt，不會呼叫 LLM"
                    onChange={(event) => setTemplatePreviewSamplePrompt(event.target.value)}
                  />
                </label>
                {templateValidation ? (
                  <p className="generation-note">
                    {templateValidation.valid
                      ? `驗證通過：${templateValidation.template_id}`
                      : `驗證失敗：${templateValidation.errors.join(" ")}`}
                  </p>
                ) : null}
                {templatePreview?.questionnaire ? (
                  <div className="history-list" aria-label="Template preview">
                    <div className="history-session">
                      <strong>{templatePreview.questionnaire.title}</strong>
                      <span>{templatePreview.questionnaire.questions.length} questions</span>
                      <small>
                        {templatePreview.questionnaire.questions
                          .map((question) => question.label)
                          .join(" / ")}
                      </small>
                    </div>
                  </div>
                ) : null}
              </section>
              <section className="settings-section" hidden={managerTab !== "prompt"}>
                <h3>Prompt</h3>
                <p>
                  {currentSession
                    ? `${managerPromptVersions.length} 個 prompt versions`
                    : "尚未選擇 session"}
                </p>
                <div className="history-list" aria-label="Prompt versions">
                  {managerPromptVersions.length === 0 ? (
                    <p>目前沒有已儲存的 optimized prompt version。</p>
                  ) : (
                    managerPromptVersions.map((version) => (
                      <button
                        className={`history-session${version.is_current ? " is-active" : ""}`}
                        key={version.prompt_version_id}
                        type="button"
                        disabled={managerBusy}
                        onClick={() => {
                          void setCurrentPromptVersion(version);
                        }}
                      >
                        <strong>{version.title ?? version.prompt_version_id}</strong>
                        <span>
                          {version.source} / {formatSessionTime(version.created_at)}
                        </span>
                        <small>{version.prompt_text}</small>
                      </button>
                    ))
                  )}
                </div>
              </section>

              <section className="settings-section" hidden={managerTab !== "proposals"}>
                <h3>Proposals</h3>
                <p>{registryProposals.length} 個待審或已處理 proposal</p>
                <div className="manager-actions" aria-label="建立 proposal">
                  <select
                    value={proposalKind}
                    onChange={(event) => setProposalKind(event.target.value as ProposalKind)}
                    aria-label="Proposal kind"
                  >
                    <option value="template">Template</option>
                    <option value="skill">Skill</option>
                  </select>
                  <select
                    value={proposalChangeKind}
                    onChange={(event) =>
                      setProposalChangeKind(event.target.value as ProposalChangeKind)
                    }
                    aria-label="Change kind"
                  >
                    <option value="create">Create</option>
                    <option value="update">Update</option>
                  </select>
                  <button
                    className="command-button command-button-primary"
                    type="button"
                    disabled={managerBusy || agentBusy || !proposalAuthoringInstruction.trim()}
                    onClick={() => {
                      void createAgentRegistryProposal();
                    }}
                  >
                    <Plus size={16} aria-hidden="true" />
                    Agent 建立提案
                  </button>
                </div>
                <label>
                  Authoring instruction
                  <textarea
                    rows={3}
                    value={proposalAuthoringInstruction}
                    onChange={(event) => setProposalAuthoringInstruction(event.target.value)}
                  />
                </label>
                <div className="history-list" aria-label="Registry proposals">
                  {registryProposals.length === 0 ? (
                    <p>目前沒有 proposal。</p>
                  ) : (
                    registryProposals.map((proposal) => (
                      <button
                        className={`history-session${
                          selectedProposalId === proposal.proposal_id ? " is-active" : ""
                        }`}
                        key={proposal.proposal_id}
                        type="button"
                        onClick={() => selectRegistryProposal(proposal)}
                      >
                        <strong>
                          {proposal.registry_kind} / {proposal.change_kind}
                        </strong>
                        <span>
                          {proposal.status} / {proposal.item_id ?? "no target"}
                        </span>
                        <small>{proposal.summary ?? proposal.diff_text}</small>
                      </button>
                    ))
                  )}
                </div>
                {selectedProposal ? (
                  <>
                    <div className="manager-actions" aria-label="Proposal 審核">
                      <button
                        className="command-button"
                        type="button"
                        disabled={managerBusy || selectedProposal.status !== "pending"}
                        onClick={() => {
                          void saveRegistryProposalDraft();
                        }}
                      >
                        <Save size={16} aria-hidden="true" />
                        儲存草稿
                      </button>
                      <button
                        className="command-button"
                        type="button"
                        disabled={managerBusy || selectedProposal.status !== "pending"}
                        onClick={() => {
                          void validateSelectedProposal();
                        }}
                      >
                        <RefreshCcw size={16} aria-hidden="true" />
                        驗證
                      </button>
                      <button
                        className="command-button command-button-primary"
                        type="button"
                        disabled={managerBusy || selectedProposal.status !== "pending"}
                        onClick={() => {
                          void approveSelectedProposal();
                        }}
                      >
                        <Save size={16} aria-hidden="true" />
                        Approve
                      </button>
                      <button
                        className="command-button"
                        type="button"
                        disabled={managerBusy || selectedProposal.status !== "pending"}
                        onClick={() => {
                          void rejectSelectedProposal();
                        }}
                      >
                        <X size={16} aria-hidden="true" />
                        Reject
                      </button>
                    </div>
                    <label>
                      Target ID
                      <input
                        value={proposalTargetId}
                        disabled={selectedProposal.status !== "pending"}
                        onChange={(event) => setProposalTargetId(event.target.value)}
                      />
                    </label>
                    <label>
                      Proposed content
                      <textarea
                        rows={14}
                        value={proposalEditorContent}
                        disabled={selectedProposal.status !== "pending"}
                        onChange={(event) => {
                          setProposalEditorContent(event.target.value);
                          setProposalValidation(null);
                        }}
                      />
                    </label>
                    <p className="generation-note">
                      {proposalValidation
                        ? proposalValidation.valid
                          ? "Proposal 驗證通過"
                          : proposalValidation.errors.join(" ")
                        : proposalValidationMessage(selectedProposal)}
                    </p>
                    <div className="history-session">
                      <strong>Diff preview</strong>
                      <small>{selectedProposal.diff_text}</small>
                    </div>
                  </>
                ) : null}
              </section>

              <section className="settings-section" hidden={managerTab !== "logs"}>
                <h3>Logs</h3>
                <p>{managerBusy ? "正在載入記錄..." : `${managerLogs.length} 筆近期記錄`}</p>
                <div className="history-list" aria-label="近期記錄清單">
                  {managerLogs.length === 0 ? (
                    <p>尚無執行記錄。</p>
                  ) : (
                    managerLogs.map((log) => (
                      <div className="history-session" key={log.log_id}>
                        <strong>{log.level}</strong>
                        <span>{formatSessionTime(log.created_at)}</span>
                        <small>{log.message}</small>
                      </div>
                    ))
                  )}
                </div>
              </section>
            </div>

            <footer className="drawer-footer">
              <Database size={16} aria-hidden="true" />
              <span>{managerMessage ?? "Models / Skills / Templates / Logs API 已接上。"}</span>
            </footer>
          </aside>
        </div>
      ) : null}

      {localFluxOpen ? (
        <div className="drawer-layer" role="presentation">
          <button
            className="drawer-scrim"
            type="button"
            aria-label="關閉 Local Flux 設定"
            onClick={() => setLocalFluxOpen(false)}
          />
          <aside
            ref={localFluxDrawerRef}
            className="settings-drawer local-flux-drawer"
            role="dialog"
            aria-modal="true"
            aria-labelledby="local-flux-title"
          >
            <header className="drawer-header">
              <div>
                <span>Local Flux</span>
                <h2 id="local-flux-title">Local Flux 設定</h2>
              </div>
              <button
                ref={localFluxCloseRef}
                className="icon-button"
                type="button"
                title="關閉"
                aria-label="關閉 Local Flux 設定"
                onClick={() => setLocalFluxOpen(false)}
              >
                <X size={18} aria-hidden="true" />
              </button>
            </header>

            <div className="drawer-content">
              <section className="settings-section">
                <h3>Server</h3>
                <label>
                  Server URL
                  <input
                    value={localFluxDraft.base_url}
                    placeholder="http://127.0.0.1:8188"
                    onChange={(event) => updateLocalFluxDraft("base_url", event.target.value)}
                  />
                </label>
                <div className="settings-status-row">
                  <span>連線</span>
                  <strong>{statusText(localFluxStatus?.available)}</strong>
                </div>
                <p>{localFluxStatus?.message ?? "尚未測試 Local Flux 連線。"}</p>
              </section>

              <section className="settings-section">
                <h3>Workflow</h3>
                {renderLocalFluxPathField(
                  "workflow_path",
                  "T2I workflow",
                  "選擇 Flux T2I workflow JSON",
                )}
                {renderLocalFluxPathField(
                  "i2i_one_workflow_path",
                  "I2I 單張 workflow",
                  "選擇 Flux one-image edit workflow JSON",
                )}
                {renderLocalFluxPathField(
                  "i2i_two_workflow_path",
                  "I2I 雙張 workflow",
                  "選擇 Flux two-image edit workflow JSON",
                )}
              </section>

              <section className="settings-section">
                <h3>Models</h3>
                {renderLocalFluxPathField(
                  "model_path",
                  "Model",
                  "flux2\\flux-2-klein-9b-fp8mixed.safetensors",
                )}
                {renderLocalFluxPathField("vae_path", "VAE", "flux\\flux2-vae.safetensors")}
                {renderLocalFluxPathField(
                  "text_encoder_path",
                  "Text encoder",
                  "qwen\\qwen_3_8b_fp8mixed.safetensors",
                )}
              </section>

              <section className="settings-section">
                <h3>Sampling</h3>
                <div className="settings-grid settings-grid-2">
                  <label>
                    Steps
                    <input
                      type="number"
                      min={1}
                      max={150}
                      value={localFluxDraft.steps}
                      onChange={(event) =>
                        updateLocalFluxDraft("steps", Number(event.target.value))
                      }
                    />
                  </label>
                  <label>
                    CFG
                    <input
                      type="number"
                      min={0}
                      max={30}
                      step={0.1}
                      value={localFluxDraft.cfg}
                      onChange={(event) => updateLocalFluxDraft("cfg", Number(event.target.value))}
                    />
                  </label>
                  <label>
                    Sampler
                    <input
                      value={localFluxDraft.sampler_name}
                      onChange={(event) =>
                        updateLocalFluxDraft("sampler_name", event.target.value)
                      }
                    />
                  </label>
                  <label>
                    Scheduler
                    <input
                      value={localFluxDraft.scheduler}
                      onChange={(event) => updateLocalFluxDraft("scheduler", event.target.value)}
                    />
                  </label>
                  <label>
                    Denoise
                    <input
                      type="number"
                      min={0}
                      max={1}
                      step={0.05}
                      value={localFluxDraft.denoise}
                      onChange={(event) =>
                        updateLocalFluxDraft("denoise", Number(event.target.value))
                      }
                    />
                  </label>
                  <label>
                    Flux guidance
                    <input
                      type="number"
                      min={0}
                      max={30}
                      step={0.1}
                      value={localFluxDraft.guidance}
                      onChange={(event) =>
                        updateLocalFluxDraft("guidance", Number(event.target.value))
                      }
                    />
                  </label>
                </div>
              </section>

              <section className="settings-section">
                <h3>Image</h3>
                <div className="settings-grid settings-grid-2">
                  <label>
                    Width
                    <input
                      type="number"
                      min={64}
                      max={4096}
                      value={localFluxDraft.width}
                      onChange={(event) =>
                        updateLocalFluxDraft("width", Number(event.target.value))
                      }
                    />
                  </label>
                  <label>
                    Height
                    <input
                      type="number"
                      min={64}
                      max={4096}
                      value={localFluxDraft.height}
                      onChange={(event) =>
                        updateLocalFluxDraft("height", Number(event.target.value))
                      }
                    />
                  </label>
                  <label>
                    Seed
                    <input
                      inputMode="numeric"
                      placeholder="空白代表隨機"
                      value={localFluxDraft.seed ?? ""}
                      onChange={(event) =>
                        updateLocalFluxDraft(
                          "seed",
                          event.target.value ? Number(event.target.value) : null,
                        )
                      }
                    />
                  </label>
                  <label>
                    Output prefix
                    <input
                      value={localFluxDraft.output_prefix}
                      onChange={(event) =>
                        updateLocalFluxDraft("output_prefix", event.target.value)
                      }
                    />
                  </label>
                  <label>
                    Timeout seconds
                    <input
                      type="number"
                      min={1}
                      max={7200}
                      value={localFluxDraft.timeout_seconds}
                      onChange={(event) =>
                        updateLocalFluxDraft("timeout_seconds", Number(event.target.value))
                      }
                    />
                  </label>
                </div>
              </section>

              <section className="settings-section">
                <h3>Actions</h3>
                <div className="manager-actions" aria-label="Local Flux 操作">
                  <button
                    className="command-button command-button-primary"
                    type="button"
                    disabled={localFluxBusy}
                    onClick={() => {
                      void saveLocalFluxSettings();
                    }}
                  >
                    <Save size={16} aria-hidden="true" />
                    儲存設定
                  </button>
                  <button
                    className="command-button"
                    type="button"
                    disabled={localFluxBusy}
                    onClick={() => {
                      void testLocalFluxConnection();
                    }}
                  >
                    <RefreshCcw size={16} aria-hidden="true" />
                    測試連線
                  </button>
                  <button
                    className="command-button"
                    type="button"
                    disabled={localFluxBusy}
                    onClick={() => {
                      void validateLocalFluxWorkflow("t2i");
                    }}
                  >
                    驗證 T2I
                  </button>
                  <button
                    className="command-button"
                    type="button"
                    disabled={localFluxBusy}
                    onClick={() => {
                      void validateLocalFluxWorkflow("i2i");
                    }}
                  >
                    驗證 I2I
                  </button>
                </div>
              </section>
            </div>

            <footer className="drawer-footer">
              <SlidersHorizontal size={16} aria-hidden="true" />
              <span aria-live="polite" aria-atomic="true">
                {localFluxBusy
                  ? "Local Flux 設定更新中..."
                  : (localFluxMessage ??
                    localFluxModel?.message ??
                    "Local Flux 會把設定好的 workflow 與參數送到本機 backend。")}
              </span>
            </footer>
          </aside>
        </div>
      ) : null}

      {settingsOpen ? (
        <div className="drawer-layer" role="presentation">
          <button
            className="drawer-scrim"
            type="button"
            aria-label="關閉設定"
            onClick={() => setSettingsOpen(false)}
          />
          <aside
            ref={settingsDrawerRef}
            className="settings-drawer"
            role="dialog"
            aria-modal="true"
            aria-labelledby="settings-title"
          >
            <header className="drawer-header">
              <div>
                <span>Settings</span>
                <h2 id="settings-title">本機設定</h2>
              </div>
              <button
                ref={settingsCloseRef}
                className="icon-button"
                type="button"
                title="關閉"
                aria-label="關閉設定"
                onClick={() => setSettingsOpen(false)}
              >
                <X size={18} aria-hidden="true" />
              </button>
            </header>

            <div className="drawer-content">
              {isCodexAgent ? (
              <section className="settings-section">
                <h3>Codex CLI</h3>
                <label>
                  預設模型
                  <select
                    value={codexModels.default_model}
                    onChange={(event) => {
                      void updateCodexDefaultModel(event.target.value);
                    }}
                  >
                    {codexModels.model_options.map((model) => (
                      <option key={model} value={model}>
                        {model}
                      </option>
                    ))}
                  </select>
                </label>
                <label>
                  思考強度
                  <select
                    value={codexModels.default_reasoning_effort}
                    onChange={(event) => {
                      void updateCodexRuntimeOptions({
                        default_reasoning_effort: event.target
                          .value as CodexModelsResponse["default_reasoning_effort"],
                      });
                    }}
                  >
                    {codexModels.reasoning_effort_options.map((effort) => (
                      <option key={effort} value={effort}>
                        {effort}
                      </option>
                    ))}
                  </select>
                </label>
                <label>
                  推理摘要
                  <select
                    value={codexModels.default_reasoning_summary}
                    onChange={(event) => {
                      void updateCodexRuntimeOptions({
                        default_reasoning_summary: event.target
                          .value as CodexModelsResponse["default_reasoning_summary"],
                      });
                    }}
                  >
                    {["auto", "concise", "detailed", "none"].map((summary) => (
                      <option key={summary} value={summary}>
                        {summary}
                      </option>
                    ))}
                  </select>
                </label>
                <label>
                  輸出詳細度
                  <select
                    value={codexModels.default_verbosity ?? ""}
                    onChange={(event) => {
                      void updateCodexRuntimeOptions({
                        default_verbosity: event.target.value
                          ? (event.target.value as NonNullable<CodexModelsResponse["default_verbosity"]>)
                          : null,
                      });
                    }}
                  >
                    <option value="">default</option>
                    {["low", "medium", "high"].map((verbosity) => (
                      <option key={verbosity} value={verbosity}>
                        {verbosity}
                      </option>
                    ))}
                  </select>
                </label>
                <label>
                  模型選項
                  <textarea
                    value={codexOptionsDraft}
                    readOnly
                  />
                </label>
                <button
                  className="command-button command-button-primary"
                  type="button"
                  hidden
                  disabled
                  onClick={() => {
                    void saveCodexModelOptions();
                  }}
                >
                  <Save size={16} aria-hidden="true" />
                  儲存
                </button>
              </section>

              ) : (
              <section className="settings-section">
                <h3>Ollama</h3>
                <label>
                  本機模型
                  <select
                    value={ollamaModels.selected_model ?? ""}
                    disabled={ollamaModels.models.length === 0}
                    onChange={(event) => {
                      void updateOllamaDefaultModel(event.target.value);
                    }}
                  >
                    {ollamaModels.models.length === 0 ? (
                      <option value="">未偵測到模型</option>
                    ) : (
                      ollamaModels.models.map((model) => (
                        <option key={model} value={model}>
                          {model}
                        </option>
                      ))
                    )}
                  </select>
                </label>
                <button
                  className="command-button"
                  type="button"
                  onClick={() => {
                    void refreshProviderSettings();
                  }}
                >
                  <RefreshCcw size={16} aria-hidden="true" />
                  重新整理
                </button>
              </section>
              )}

              <section className="settings-section">
                <h3>狀態</h3>
                <div className="settings-status-row">
                  <span>Codex</span>
                  <strong>{statusText(codexStatus?.available)}</strong>
                </div>
                <div className="settings-status-row">
                  <span>Ollama</span>
                  <strong>{statusText(ollamaStatus?.available)}</strong>
                </div>
                <div className="settings-status-row">
                  <span>HF_TOKEN</span>
                  <strong>{secretStatus?.hf_token_configured ? "已設定" : "未設定"}</strong>
                </div>
              </section>
            </div>

            <footer className="drawer-footer">
              <SlidersHorizontal size={16} aria-hidden="true" />
              <span aria-live="polite" aria-atomic="true">
                {settingsMessage ?? `API ${backendBaseUrl}`}
              </span>
            </footer>
          </aside>
        </div>
      ) : null}
    </main>
  );
}

export default App;
