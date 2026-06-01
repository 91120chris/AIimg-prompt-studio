# Prompt Optimizer Studio

Local-first T2I / I2I prompt optimization desktop app.

Current implementation covers the runnable Tauri + React shell, FastAPI backend, provider status, SQLite session storage, safe file URLs, reference image thumbnails, the Codex and Ollama questionnaire loops, the Codex CLI image-generation path after explicit user confirmation, post-generation feedback questionnaires, and prompt refinement from feedback.
Safe runtime settings are persisted in SQLite through `app_settings`, so provider/model selections survive backend restarts.

## Backend

```bash
cd services/backend
cp .env.example .env
# Edit .env and set HF_TOKEN only if you want future Hugging Face downloads.
uv sync
uv run uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Run tests:

```bash
cd services/backend
uv run pytest
```

## Frontend

```bash
cd apps/desktop
npm install
npm run build
npm run tauri:dev
```

On this Windows PowerShell setup, script execution policy may block `npm.ps1`. Use:

```powershell
cd apps/desktop
cmd /c npm install
cmd /c npm run build
cmd /c npm run tauri:dev
```

The default `npm run dev` builds and serves `dist/` with Vite preview so this repository works even when the local path contains `#`. On paths without special URL characters, `npm run dev:vite` starts the normal Vite dev server with hot reload.

## Local Tool Checks

```bash
codex --version
codex login status
ollama list
curl http://localhost:11434/api/tags
```

PowerShell fallback:

```powershell
cmd /c codex --version
cmd /c codex login status
ollama list
curl http://localhost:11434/api/tags
```

If `codex --version` fails in PowerShell because of execution policy, use `cmd /c codex --version` and later configure `CODEX_BINARY_PATH=codex.cmd` or an absolute executable path.

## Codex Runtime Options

Codex model labels are configured in `.env` with `CODEX_MODEL_OPTIONS`, while the active default is `CODEX_DEFAULT_MODEL`. The desktop settings drawer can update those in memory during a running backend session.

Reasoning controls are separate from model labels:

```env
CODEX_DEFAULT_REASONING_EFFORT=medium
CODEX_REASONING_EFFORT_OPTIONS=low,medium,high,xhigh
CODEX_DEFAULT_REASONING_SUMMARY=auto
CODEX_DEFAULT_VERBOSITY=
```

When the app invokes Codex CLI, it keeps `--model` for the selected model and sends runtime behavior through Codex config overrides, for example:

```powershell
cmd /c codex --config model_reasoning_effort=high exec "..."
```

This matches Codex CLI's `--config key=value` mechanism and keeps the app flexible if future Codex releases add more runtime knobs.

## Ollama Runtime Options

Ollama models are discovered live from `/api/tags`, so the desktop model picker reflects whichever models are currently installed locally. The agent loop uses `/api/generate` with `stream=false` and the same strict JSON schema used by the Codex loop.

```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_SELECTED_MODEL=
OLLAMA_TIMEOUT_SECONDS=300
OLLAMA_AGENT_TEMPERATURE=0.2
```

## Optional Codex Smoke Test

Codex smoke tests are opt-in because they may call a real local Codex model:

```bash
cd services/backend
RUN_CODEX_SMOKE=1 uv run pytest -m codex_smoke
```

PowerShell:

```powershell
cd services/backend
$env:RUN_CODEX_SMOKE="1"; uv run pytest -m codex_smoke
```

## Notes

- Do not commit `.env`.
- Do not paste Hugging Face tokens into source files.
- If a token was exposed, rotate it in Hugging Face settings.
- This app does not require an OpenAI API key.
- Codex CLI is the primary agent provider. The current loop can ask Codex for a schema-validated questionnaire, produce an optimized prompt, generate an image after explicit confirmation, ask for post-generation feedback, and refine the next prompt version from that feedback.
- Ollama is the local fallback agent provider. The desktop shell reads installed Ollama models live from `/api/tags`, lets you choose the current local model, and uses `/api/generate` with structured output schema for questionnaire, prompt optimization, feedback questionnaire, and refinement turns.
- If an agent response fails strict schema validation even after one repair attempt, the backend returns a safe single-question text questionnaire so the user can continue manually instead of losing the flow.
- The Manager drawer now reads local model status, skill versions, template versions, and recent logs from backend APIs. Patch proposal endpoints currently record and approve/reject proposals only; applying content diffs is a later milestone.
- Diffusers FLUX support comes after Milestone 1C / Phase 1.
- `CORS_ALLOW_ORIGINS` is comma-separated and parsed with `NoDecode`.
- `CODEX_MODEL_OPTIONS` is comma-separated and parsed with `NoDecode`.
- Tauri development requires Rust/Cargo. Install the Rust MSVC toolchain before running `npm run tauri:dev`.

## Implemented Local APIs

- `GET /health`
- `GET /providers/codex/status`
- `GET /providers/codex/models`
- `PATCH /providers/codex/model-options`
- `PATCH /providers/codex/default-model`
- `PATCH /providers/codex/runtime-options`
- `GET /providers/ollama/status`
- `GET /providers/ollama/models`
- `PATCH /providers/ollama/default-model`
- `GET /models`
- `GET /models/flux/status`
- `POST /models/flux/set-path`
- `POST /models/flux/install`
- `POST /models/flux/unload`
- `GET /settings/safe`
- `PATCH /settings/safe`
- `GET /security/secrets/status`
- `GET /skills`
- `GET /skills/{skill_id}`
- `POST /skills/patch-proposals`
- `POST /skills/patch-proposals/{proposal_id}/approve`
- `POST /skills/patch-proposals/{proposal_id}/reject`
- `GET /templates`
- `GET /templates/{template_id}`
- `POST /templates/patch-proposals`
- `POST /templates/patch-proposals/{proposal_id}/approve`
- `POST /templates/patch-proposals/{proposal_id}/reject`
- `GET /logs`
- `POST /sessions`
- `GET /sessions`
- `GET /sessions/{session_id}`
- `DELETE /sessions/{session_id}`
- `POST /sessions/{session_id}/reference-images`
- `DELETE /sessions/{session_id}/reference-images/{slot}`
- `GET /sessions/{session_id}/reference-images`
- `GET /sessions/{session_id}/generated-images`
- `POST /agent/turn`
- `POST /agent/answer-questionnaire`
- `POST /agent/feedback-questionnaire`
- `POST /agent/refine`
- `POST /generation/confirm`
- `POST /generation/cancel`
- `GET /generation/{job_id}`
- `GET /files/sessions/{session_id}/generated-images/{image_id}`
- `GET /files/sessions/{session_id}/reference-images/{reference_image_id}`
