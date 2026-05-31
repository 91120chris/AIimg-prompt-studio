import { useEffect, useMemo, useState } from "react";

import {
  type CodexModelsResponse,
  type CodexStatusResponse,
  type HealthResponse,
  type OllamaModelsResponse,
  type OllamaStatusResponse,
  type ReferenceImageResponse,
  type SecretStatusResponse,
  type SessionResponse,
  codexModelsResponseSchema,
  codexStatusResponseSchema,
  healthResponseSchema,
  ollamaModelsResponseSchema,
  ollamaStatusResponseSchema,
  referenceImageResponseSchema,
  secretStatusResponseSchema,
  sessionResponseSchema,
} from "./schemas/api.ts";

type BackendStatus = "checking" | "connected" | "disconnected";

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

function statusText(value: boolean | undefined): string {
  if (value === true) {
    return "可用";
  }
  if (value === false) {
    return "不可用";
  }
  return "檢查中";
}

function App() {
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

  return (
    <main className="app-shell">
      <header className="top-bar">
        <div className="brand-block">
          <span className="brand-kicker">Local First</span>
          <h1>Prompt Optimizer Studio</h1>
        </div>

        <div className="toolbar" aria-label="工作階段設定">
          <label>
            模式
            <select defaultValue="t2i">
              <option value="t2i">T2I</option>
              <option value="i2i">I2I</option>
            </select>
          </label>
          <label>
            Agent
            <select defaultValue="codex_cli">
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
            <select defaultValue="codex_cli_gpt_image">
              <option value="codex_cli_gpt_image">Codex GPT Image</option>
              <option value="diffusers_flux2" disabled>
                FLUX Phase 2
              </option>
            </select>
          </label>
          <label className="seed-field">
            Seed
            <input inputMode="numeric" placeholder="隨機" />
          </label>
        </div>

        <div className={`connection-pill connection-pill--${backendStatus}`}>
          <span aria-hidden="true" />
          {statusLabel}
        </div>
      </header>

      <section className="workspace-grid" aria-label="Prompt workflow workspace">
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
              <p>
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
                return (
                  <label className="reference-upload" key={slot}>
                    {imageUrl ? (
                      <img src={imageUrl} alt={`參考圖片 ${slot}`} />
                    ) : (
                      <span>Slot {slot}</span>
                    )}
                    <input
                      type="file"
                      accept="image/*"
                      onChange={(event) => {
                        void uploadReferenceImage(slot, event.currentTarget.files?.[0] ?? null);
                        event.currentTarget.value = "";
                      }}
                    />
                  </label>
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
          <label className="editor-block">
            原始 Prompt
            <textarea placeholder="輸入你想優化的影像提示..." />
          </label>
          <label className="editor-block editor-block-large">
            最佳化 Prompt
            <textarea placeholder="Agent 產生的最佳化提示會顯示在這裡。" />
          </label>
          <div className="generation-panel">
            <div>
              <h3>生成控制</h3>
              <p>Agent 不會自動生成圖片，必須等待使用者確認。</p>
            </div>
            <button type="button" disabled>
              等待 1C 啟用
            </button>
          </div>
        </section>

        <aside className="column column-right">
          <div className="column-heading">
            <h2>圖像顯示</h2>
            <p>輸出圖片會透過後端安全 URL 顯示。</p>
          </div>
          <div className="image-stage">
            <div className="image-placeholder">
              <span />
              <strong>尚未生成圖片</strong>
              <p>完成問卷並確認生成後，結果會出現在這裡。</p>
            </div>
          </div>
        </aside>
      </section>
    </main>
  );
}

export default App;
