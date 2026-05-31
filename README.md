# Prompt Optimizer Studio

Local-first T2I / I2I prompt optimization desktop app scaffold.

Milestone 0A provides a runnable FastAPI backend, React/Vite frontend shell, and Tauri v2 desktop shell. Codex, Ollama, database, safe files, and generation workflows come in later milestones.

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
- Codex CLI is the primary future provider.
- Ollama is the local fallback future provider. The desktop shell reads installed Ollama models live from `/api/tags` and lets you choose the current local model.
- Diffusers FLUX support comes after Milestone 1C / Phase 1.
- `CORS_ALLOW_ORIGINS` is comma-separated and parsed with `NoDecode`.
- `CODEX_MODEL_OPTIONS` is comma-separated and parsed with `NoDecode`.
- Tauri development requires Rust/Cargo. Install the Rust MSVC toolchain before running `npm run tauri:dev`.
