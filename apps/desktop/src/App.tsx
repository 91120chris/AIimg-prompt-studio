import { useEffect, useMemo, useRef, useState } from "react";
import {
  Database,
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
  type GenerationJobResponse,
  type HealthResponse,
  type OllamaModelsResponse,
  type OllamaStatusResponse,
  type Question,
  type Questionnaire,
  type ReferenceImageResponse,
  type SecretStatusResponse,
  type SessionResponse,
  agentTurnResponseSchema,
  codexModelsResponseSchema,
  codexStatusResponseSchema,
  generationJobResponseSchema,
  healthResponseSchema,
  ollamaModelsResponseSchema,
  ollamaStatusResponseSchema,
  referenceImageResponseSchema,
  secretStatusResponseSchema,
  sessionResponseSchema,
} from "./schemas/api.ts";

type BackendStatus = "checking" | "connected" | "disconnected";
type AgentProvider = "codex_cli" | "ollama_local_llm";
type ImageProvider = "codex_cli_gpt_image" | "diffusers_flux2";
type WorkflowMode = "t2i" | "i2i";
type AnswerDraftValue = string | boolean | number | string[];
type AnswerDrafts = Record<string, AnswerDraftValue>;
type QuestionnaireAnswerRequest =
  | { kind: "text"; question_id: string; value: string }
  | { kind: "choice"; question_id: string; value: string }
  | { kind: "multi_choice"; question_id: string; values: string[] }
  | { kind: "boolean"; question_id: string; value: boolean }
  | { kind: "scale"; question_id: string; value: number };

const fallbackCodexModels: CodexModelsResponse = {
  default_model: "auto",
  model_options: ["auto", "gpt-5.5", "gpt-5.4"],
};

const fallbackOllamaModels: OllamaModelsResponse = {
  selected_model: null,
  models: [],
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

function statusText(value: boolean | undefined): string {
  if (value === true) {
    return "可用";
  }
  if (value === false) {
    return "不可用";
  }
  return "檢查中";
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
  const [workflowMode, setWorkflowMode] = useState<WorkflowMode>("t2i");
  const [agentProvider, setAgentProvider] = useState<AgentProvider>("codex_cli");
  const [imageProvider, setImageProvider] = useState<ImageProvider>("codex_cli_gpt_image");
  const [seed, setSeed] = useState("");
  const [steps, setSteps] = useState(28);
  const [guidance, setGuidance] = useState(3.5);
  const [originalPrompt, setOriginalPrompt] = useState("");
  const [optimizedPrompt, setOptimizedPrompt] = useState("");
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [codexOptionsDraft, setCodexOptionsDraft] = useState(
    fallbackCodexModels.model_options.join(", "),
  );
  const [settingsMessage, setSettingsMessage] = useState<string | null>(null);
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
    setAgentMessage("Codex 正在分析 prompt...");
    setErrorMessage(null);
    try {
      const session = await ensureSession();
      const response = await fetch(`${backendBaseUrl}/agent/turn`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: session.session_id,
          original_prompt: prompt,
          mode: workflowMode,
          provider: agentProvider,
          codex_model: codexModels.default_model,
          ollama_model: ollamaModels.selected_model,
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

    setAgentBusy(true);
    setAgentMessage("Codex 正在整理答案並最佳化 prompt...");
    setErrorMessage(null);
    try {
      const answers = questionnaireAnswers(questionnaire, answerDrafts);
      const response = await fetch(`${backendBaseUrl}/agent/answer-questionnaire`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: currentSession.session_id,
          questionnaire_id: questionnaire.questionnaire_id,
          provider: agentProvider,
          codex_model: codexModels.default_model,
          ollama_model: ollamaModels.selected_model,
          answers,
        }),
      });
      if (!response.ok) {
        throw new Error(await readErrorMessage(response));
      }
      applyAgentTurn(agentTurnResponseSchema.parse(await response.json()));
    } catch (error) {
      setAgentMessage("問卷送出失敗");
      setErrorMessage(error instanceof Error ? error.message : "無法送出問卷答案");
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

        const [codexPayload, codexModelsPayload, ollamaPayload, ollamaModelsPayload, secretsPayload] =
          await Promise.allSettled([
            fetchJson("/providers/codex/status", codexStatusResponseSchema, controller.signal),
            fetchJson("/providers/codex/models", codexModelsResponseSchema, controller.signal),
            fetchJson("/providers/ollama/status", ollamaStatusResponseSchema, controller.signal),
            fetchJson("/providers/ollama/models", ollamaModelsResponseSchema, controller.signal),
            fetchJson("/security/secrets/status", secretStatusResponseSchema, controller.signal),
          ]);

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

  const generatedPreviewImage = generationJob?.images[0] ?? null;

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
              onChange={(event) => setAgentProvider(event.target.value as AgentProvider)}
            >
              <option value="codex_cli">Codex CLI</option>
              <option value="ollama_local_llm">Ollama</option>
            </select>
          </label>
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
          <label>
            圖像 Provider
            <select
              value={imageProvider}
              onChange={(event) => setImageProvider(event.target.value as ImageProvider)}
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
          <button className="icon-button" type="button" title="歷史" aria-label="歷史紀錄">
            <History size={17} aria-hidden="true" />
          </button>
          <button className="icon-button" type="button" title="模型" aria-label="模型狀態">
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
            onClick={() => setSettingsOpen(true)}
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
                  送出問卷並最佳化
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
                    {generationJob?.status}
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
