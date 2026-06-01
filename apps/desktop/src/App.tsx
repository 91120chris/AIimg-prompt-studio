import { useEffect, useMemo, useRef, useState } from "react";
import {
  Database,
  FolderOpen,
  History,
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
  type FluxReadinessResponse,
  type GenerationJobResponse,
  type HealthResponse,
  type LogResponse,
  type ModelInfoResponse,
  type OllamaModelsResponse,
  type OllamaStatusResponse,
  type Question,
  type Questionnaire,
  type ReferenceImageResponse,
  type RegistryItemResponse,
  type SafeSettingsResponse,
  type SecretStatusResponse,
  type SessionResponse,
  agentTurnResponseSchema,
  codexModelsResponseSchema,
  codexStatusResponseSchema,
  fluxReadinessResponseSchema,
  generationJobResponseSchema,
  healthResponseSchema,
  logsResponseSchema,
  modelInfoResponseSchema,
  modelInfoListResponseSchema,
  ollamaModelsResponseSchema,
  ollamaStatusResponseSchema,
  referenceImageResponseSchema,
  registryItemsResponseSchema,
  safeSettingsResponseSchema,
  secretStatusResponseSchema,
  sessionResponseSchema,
  sessionsResponseSchema,
} from "./schemas/api.ts";

type BackendStatus = "checking" | "connected" | "disconnected";
type AgentProvider = "codex_cli" | "ollama_local_llm";
type ImageProvider = "codex_cli_gpt_image" | "diffusers_flux2";
type WorkflowMode = "t2i" | "i2i";
type FluxManagerAction = "set-path" | "install" | "unload";
type AnswerDraftValue = string | boolean | number | string[];
type AnswerDrafts = Record<string, AnswerDraftValue>;
type FeedbackQuestionnaireContext = { questionnaireId: string; jobId: string };
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
  default_model: "auto",
  model_options: ["auto", "gpt-5.5", "gpt-5.4"],
  default_reasoning_effort: "medium",
  reasoning_effort_options: ["low", "medium", "high", "xhigh"],
  default_reasoning_summary: "auto",
  default_verbosity: null,
};

const fallbackOllamaModels: OllamaModelsResponse = {
  selected_model: null,
  models: [],
};

const FLUX_PROVIDER_ID = "diffusers_flux2_klein_9b_fp8";

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

async function readErrorMessage(response: Response): Promise<string> {
  try {
    const payload = await response.json();
    const detail = payload?.detail;
    if (detail?.message) {
      return detail.suggestion ? `${detail.message} ${detail.suggestion}` : detail.message;
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
  const [workflowMode, setWorkflowMode] = useState<WorkflowMode>("t2i");
  const [agentProvider, setAgentProvider] = useState<AgentProvider>("codex_cli");
  const [imageProvider, setImageProvider] = useState<ImageProvider>("codex_cli_gpt_image");
  const [seed, setSeed] = useState("");
  const [steps, setSteps] = useState(28);
  const [guidance, setGuidance] = useState(3.5);
  const [originalPrompt, setOriginalPrompt] = useState("");
  const [optimizedPrompt, setOptimizedPrompt] = useState("");
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [historyOpen, setHistoryOpen] = useState(false);
  const [managerOpen, setManagerOpen] = useState(false);
  const [codexOptionsDraft, setCodexOptionsDraft] = useState(
    fallbackCodexModels.model_options.join(", "),
  );
  const [settingsMessage, setSettingsMessage] = useState<string | null>(null);
  const [historyMessage, setHistoryMessage] = useState<string | null>(null);
  const [historyBusy, setHistoryBusy] = useState(false);
  const [sessionHistory, setSessionHistory] = useState<SessionResponse[]>([]);
  const [managerMessage, setManagerMessage] = useState<string | null>(null);
  const [managerBusy, setManagerBusy] = useState(false);
  const [managerModels, setManagerModels] = useState<ModelInfoResponse[]>([]);
  const [managerSkills, setManagerSkills] = useState<RegistryItemResponse[]>([]);
  const [managerTemplates, setManagerTemplates] = useState<RegistryItemResponse[]>([]);
  const [managerLogs, setManagerLogs] = useState<LogResponse[]>([]);
  const [fluxReadiness, setFluxReadiness] = useState<FluxReadinessResponse | null>(null);
  const [fluxPathDraft, setFluxPathDraft] = useState("");
  const [managerActionBusy, setManagerActionBusy] = useState<FluxManagerAction | null>(null);
  const [managerPathPickerBusy, setManagerPathPickerBusy] = useState(false);
  const [backendStatus, setBackendStatus] = useState<BackendStatus>("checking");
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [codexStatus, setCodexStatus] = useState<CodexStatusResponse | null>(null);
  const [ollamaStatus, setOllamaStatus] = useState<OllamaStatusResponse | null>(null);
  const [ollamaModels, setOllamaModels] = useState<OllamaModelsResponse>(fallbackOllamaModels);
  const [secretStatus, setSecretStatus] = useState<SecretStatusResponse | null>(null);
  const [codexModels, setCodexModels] = useState<CodexModelsResponse>(fallbackCodexModels);
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
  const [feedbackQuestionnaireContext, setFeedbackQuestionnaireContext] =
    useState<FeedbackQuestionnaireContext | null>(null);
  const isCodexAgent = agentProvider === "codex_cli";
  const activeAgentName = isCodexAgent ? "Codex" : "Ollama";
  const fluxModel = useMemo(
    () => managerModels.find((model) => model.provider === FLUX_PROVIDER_ID) ?? null,
    [managerModels],
  );

  useEffect(() => {
    setCodexOptionsDraft(codexModels.model_options.join(", "));
  }, [codexModels.model_options]);

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
      const [codexModelsPayload, ollamaModelsPayload, codexPayload, ollamaPayload] =
        await Promise.all([
          fetchJson("/providers/codex/models", codexModelsResponseSchema),
          fetchJson("/providers/ollama/models", ollamaModelsResponseSchema),
          fetchJson("/providers/codex/status", codexStatusResponseSchema),
          fetchJson("/providers/ollama/status", ollamaStatusResponseSchema),
        ]);

      setCodexModels(codexModelsPayload);
      setOllamaModels(ollamaModelsPayload);
      setCodexStatus(codexPayload);
      setOllamaStatus(ollamaPayload);
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
    setHistoryOpen(true);
    await refreshSessionHistory();
  }

  async function refreshManagerData() {
    setManagerBusy(true);
    setManagerMessage(null);
    try {
      const [models, readiness, skills, templates, logs] = await Promise.all([
        fetchJson("/models", modelInfoListResponseSchema),
        fetchJson("/models/flux/readiness", fluxReadinessResponseSchema),
        fetchJson("/skills", registryItemsResponseSchema),
        fetchJson("/templates", registryItemsResponseSchema),
        fetchJson("/logs", logsResponseSchema),
      ]);
      setManagerModels(models);
      setFluxReadiness(readiness);
      setManagerSkills(skills);
      setManagerTemplates(templates);
      setManagerLogs(logs);
      setManagerMessage(
        `已載入 ${models.length} models / ${skills.length} skills / ${templates.length} templates / ${logs.length} logs`,
      );
    } catch (error) {
      setManagerMessage(error instanceof Error ? error.message : "無法載入管理資料");
    } finally {
      setManagerBusy(false);
    }
  }

  function replaceManagerModel(nextModel: ModelInfoResponse) {
    setManagerModels((current) => {
      const hasModel = current.some((model) => model.provider === nextModel.provider);
      if (!hasModel) {
        return [nextModel, ...current];
      }
      return current.map((model) => (model.provider === nextModel.provider ? nextModel : model));
    });
  }

  async function chooseFluxModelPath() {
    if (!isTauriRuntime()) {
      setManagerMessage("資料夾選擇器只在 Tauri 桌面版可用；目前瀏覽器預覽請手動貼上路徑。");
      return;
    }

    setManagerPathPickerBusy(true);
    setManagerMessage(null);
    try {
      const { open } = await import("@tauri-apps/plugin-dialog");
      const selected = await open({
        directory: true,
        multiple: false,
        title: "選擇 FLUX 模型資料夾",
        defaultPath: fluxPathDraft.trim() || undefined,
      });

      if (typeof selected === "string") {
        setFluxPathDraft(selected);
        setManagerMessage("已選擇 FLUX 模型路徑，請按儲存路徑。");
        return;
      }
      if (Array.isArray(selected) && typeof selected[0] === "string") {
        setFluxPathDraft(selected[0]);
        setManagerMessage("已選擇 FLUX 模型路徑，請按儲存路徑。");
        return;
      }
      setManagerMessage("已取消選擇路徑。");
    } catch {
      setManagerMessage("無法開啟資料夾選擇器；目前可手動輸入或貼上路徑。");
    } finally {
      setManagerPathPickerBusy(false);
    }
  }

  async function runFluxManagerAction(action: FluxManagerAction) {
    const modelPath = fluxPathDraft.trim();
    if (action === "set-path" && !modelPath) {
      setManagerMessage("請先輸入 FLUX 本機模型路徑。");
      return;
    }

    const actionPath =
      action === "set-path"
        ? "/models/flux/set-path"
        : action === "install"
          ? "/models/flux/install"
          : "/models/flux/unload";
    const actionMessages: Record<FluxManagerAction, string> = {
      "set-path": "FLUX 路徑已儲存。",
      install: "FLUX 已標記為待安裝。",
      unload: "FLUX 已卸載。",
    };

    setManagerActionBusy(action);
    setManagerMessage(null);
    try {
      const response = await fetch(`${backendBaseUrl}${actionPath}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: action === "set-path" ? JSON.stringify({ model_path: modelPath }) : undefined,
      });
      if (!response.ok) {
        throw new Error(await readErrorMessage(response));
      }
      const model = modelInfoResponseSchema.parse(await response.json());
      replaceManagerModel(model);
      if (action === "set-path") {
        setFluxPathDraft("");
      }
      try {
        const [logs, readiness] = await Promise.all([
          fetchJson("/logs", logsResponseSchema),
          fetchJson("/models/flux/readiness", fluxReadinessResponseSchema),
        ]);
        setManagerLogs(logs);
        setFluxReadiness(readiness);
      } catch {
        // The model action succeeded; log refresh can recover on the next manual refresh.
      }
      setManagerMessage(actionMessages[action]);
    } catch (error) {
      setManagerMessage(error instanceof Error ? error.message : "FLUX 模型操作失敗");
    } finally {
      setManagerActionBusy(null);
    }
  }

  async function openManagerDrawer() {
    setSettingsOpen(false);
    setHistoryOpen(false);
    setManagerOpen(true);
    await refreshManagerData();
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
      setAnswerDrafts(defaultDraftsForQuestionnaire(turn.questionnaire));
      setAgentMessage(turn.message);
      return;
    }
    if (turn.kind === "optimized_prompt") {
      setOptimizedPrompt(turn.optimized_prompt);
      setAgentMessage(turn.message);
      setFeedbackQuestionnaireContext(null);
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
            ...agentProviderPayload(),
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

    setGenerationBusy(true);
    setGenerationMessage("正在確認並生成圖片...");
    setErrorMessage(null);
    try {
      const session = await ensureSession();
      const seedValue = seed.trim() ? Number(seed.trim()) : null;
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
            steps,
            guidance,
            seed: seedValue === null ? null : Number.isFinite(seedValue) ? seedValue : null,
          },
          reference_image_ids: referenceImages.map((image) => image.reference_image_id),
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
      await requestFeedbackQuestionnaire(job);
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
          secretsPayload,
        ] = await Promise.allSettled([
          fetchJson("/settings/safe", safeSettingsResponseSchema, controller.signal),
          fetchJson("/providers/codex/status", codexStatusResponseSchema, controller.signal),
          fetchJson("/providers/codex/models", codexModelsResponseSchema, controller.signal),
          fetchJson("/providers/ollama/status", ollamaStatusResponseSchema, controller.signal),
          fetchJson("/providers/ollama/models", ollamaModelsResponseSchema, controller.signal),
          fetchJson("/security/secrets/status", secretStatusResponseSchema, controller.signal),
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
        if (secretsPayload.status === "fulfilled") {
          setSecretStatus(secretsPayload.value);
        }
      } catch (error) {
        if (controller.signal.aborted) {
          return;
        }
        setHealth(null);
        setCodexStatus(null);
        setOllamaStatus(null);
        setOllamaModels(fallbackOllamaModels);
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

  const generatedPreviewImage = generationJob?.images[0] ?? currentSession?.generated_images[0] ?? null;

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
              <option value="codex_cli_gpt_image">Codex GPT Image</option>
              <option value="diffusers_flux2" disabled>
                FLUX Phase 2
              </option>
            </select>
          </label>
          <label className="seed-field">
            Seed
            <input
              inputMode="numeric"
              placeholder="隨機"
              value={seed}
              onChange={(event) => setSeed(event.target.value)}
            />
          </label>
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
              <select defaultValue="draft">
                <option value="draft">草稿 v0</option>
              </select>
            </label>
            <label>
              模板
              <select defaultValue="general">
                <option value="general">通用影像</option>
                <option value="product">產品視覺</option>
                <option value="character">角色設計</option>
              </select>
            </label>
            <label>
              Steps
              <input
                min={1}
                max={80}
                type="number"
                value={steps}
                onChange={(event) => setSteps(Number(event.target.value))}
              />
            </label>
            <label>
              Guidance
              <input
                min={0}
                max={20}
                step={0.5}
                type="number"
                value={guidance}
                onChange={(event) => setGuidance(Number(event.target.value))}
              />
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
                {workflowMode.toUpperCase()} / {imageProvider} / Seed {seed || "隨機"}
              </p>
              {generationMessage ? <p className="generation-note">{generationMessage}</p> : null}
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
                  !optimizedPrompt.trim() ||
                  imageProvider !== "codex_cli_gpt_image"
                }
                onClick={() => {
                  void confirmGeneration();
                }}
              >
                {generationBusy ? "生成中" : "確認生成"}
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
                  src={`${backendBaseUrl}${
                    generatedPreviewImage.thumbnail_url ?? generatedPreviewImage.url
                  }`}
                  alt="生成圖片"
                />
                <div>
                  <strong>{generatedPreviewImage.filename}</strong>
                  <p>
                    {generatedPreviewImage.width} x {generatedPreviewImage.height} /{" "}
                    {generationJob?.status ?? "history"}
                  </p>
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

              <section className="settings-section">
                <h3>Models</h3>
                <p>{managerBusy ? "正在載入模型狀態..." : `${managerModels.length} 個模型狀態`}</p>
                <div className="manager-model-controls" aria-label="FLUX 模型管理">
                  <div className="manager-state-line">
                    <span>FLUX.2 Klein 9B FP8</span>
                    <strong>
                      {fluxModel
                        ? `${fluxModel.status}${fluxModel.path_label ? ` / ${fluxModel.path_label}` : ""}`
                        : "尚未載入"}
                    </strong>
                  </div>
                  <div className="manager-state-line">
                    <span>Hugging Face Token</span>
                    <strong>{fluxReadiness?.hf_token_configured ? "已設定" : "未設定"}</strong>
                  </div>
                  <div className="manager-state-line">
                    <span>HF Cache</span>
                    <strong>{fluxReadiness?.hf_cache_configured ? "已設定" : "使用預設"}</strong>
                  </div>
                  <label>
                    本機模型路徑
                    <div className="manager-path-row">
                      <input
                        value={fluxPathDraft}
                        placeholder="例如 C:\\models\\flux2-klein-fp8"
                        onChange={(event) => setFluxPathDraft(event.target.value)}
                      />
                      <button
                        className="command-button"
                        type="button"
                        disabled={managerBusy || managerActionBusy !== null || managerPathPickerBusy}
                        onClick={() => {
                          void chooseFluxModelPath();
                        }}
                      >
                        <FolderOpen size={16} aria-hidden="true" />
                        {managerPathPickerBusy ? "開啟中" : "選擇"}
                      </button>
                    </div>
                  </label>
                  <div className="manager-actions" aria-label="FLUX 模型操作">
                    <button
                      className="command-button command-button-primary"
                      type="button"
                      disabled={managerBusy || managerActionBusy !== null}
                      onClick={() => {
                        void runFluxManagerAction("set-path");
                      }}
                    >
                      <Save size={16} aria-hidden="true" />
                      儲存路徑
                    </button>
                    <button
                      className="command-button"
                      type="button"
                      disabled={
                        managerBusy ||
                        managerActionBusy !== null ||
                        fluxReadiness?.can_queue_install !== true
                      }
                      onClick={() => {
                        void runFluxManagerAction("install");
                      }}
                    >
                      <RefreshCcw size={16} aria-hidden="true" />
                      排入安裝
                    </button>
                    <button
                      className="command-button"
                      type="button"
                      disabled={managerBusy || managerActionBusy !== null}
                      onClick={() => {
                        void runFluxManagerAction("unload");
                      }}
                    >
                      <X size={16} aria-hidden="true" />
                      卸載
                    </button>
                  </div>
                  <p aria-live="polite">
                    {managerActionBusy
                      ? "正在更新 FLUX 模型狀態..."
                      : (fluxReadiness?.message ??
                        "完整 token 與本機路徑只保存在後端，介面只顯示安全狀態。")}
                  </p>
                </div>
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
              <section className="settings-section">
                <h3>Skills</h3>
                <p>{managerBusy ? "正在載入技能..." : `${managerSkills.length} 個技能`}</p>
                <div className="history-list" aria-label="技能清單">
                  {managerSkills.length === 0 ? (
                    <p>尚無技能版本。</p>
                  ) : (
                    managerSkills.map((skill) => (
                      <div className="history-session" key={skill.item_id}>
                        <strong>{skill.item_id}</strong>
                        <span>{skill.latest_version_id ?? "未建立版本"}</span>
                        <small>{skill.content}</small>
                      </div>
                    ))
                  )}
                </div>
              </section>
              <section className="settings-section">
                <h3>Templates</h3>
                <p>{managerBusy ? "正在載入模板..." : `${managerTemplates.length} 個模板`}</p>
                <div className="history-list" aria-label="模板清單">
                  {managerTemplates.length === 0 ? (
                    <p>尚無模板版本。</p>
                  ) : (
                    managerTemplates.map((template) => (
                      <div className="history-session" key={template.item_id}>
                        <strong>{template.item_id}</strong>
                        <span>{template.latest_version_id ?? "未建立版本"}</span>
                        <small>{template.content}</small>
                      </div>
                    ))
                  )}
                </div>
              </section>
              <section className="settings-section">
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
                    onChange={(event) => setCodexOptionsDraft(event.target.value)}
                  />
                </label>
                <button
                  className="command-button command-button-primary"
                  type="button"
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
