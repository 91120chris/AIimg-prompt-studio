# Prompt Optimizer Studio

Local-first T2I / I2I prompt optimization desktop app.

Current implementation covers the runnable Tauri + React shell, FastAPI backend, provider status, SQLite session storage, safe file URLs, reference image thumbnails, the first Codex questionnaire loop, and the Codex CLI image-generation path after explicit user confirmation.

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
- Codex CLI is the primary agent provider. Milestone 1B can ask Codex for a schema-validated questionnaire and then produce an optimized prompt from answers.
- Ollama is the local fallback future provider. The desktop shell reads installed Ollama models live from `/api/tags` and lets you choose the current local model; the full Ollama agent loop is still pending.
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
- `GET /providers/ollama/status`
- `GET /providers/ollama/models`
- `PATCH /providers/ollama/default-model`
- `GET /settings/safe`
- `PATCH /settings/safe`
- `GET /security/secrets/status`
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
- `GET /files/sessions/{session_id}/generated-images/{image_id}`
- `GET /files/sessions/{session_id}/reference-images/{reference_image_id}`
