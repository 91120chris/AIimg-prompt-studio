# Prompt Optimizer Studio

Local-first T2I / I2I prompt optimization desktop app.

Current implementation covers the runnable Tauri + React shell, FastAPI backend, provider status, SQLite session storage, safe file URLs, reference image thumbnails, the Codex and Ollama questionnaire loops, the Codex CLI image-generation path after explicit user confirmation, post-generation feedback questionnaires, and prompt refinement from feedback.
Safe runtime settings are persisted in SQLite through `app_settings`, so provider/model selections survive backend restarts.
The Phase 1 registry layer is also in place: project-local seed files define the initial skills/templates, FastAPI seeds them into SQLite on startup, and approved patch proposals create new skill/template versions without allowing the agent to directly edit registry files.
Phase 2 has started: the FLUX manager can read `HF_TOKEN`, report safe Hugging Face readiness, install the configured FP8 checkpoint snapshot with `huggingface_hub`, classify gated/private/token/download errors, and register only a safe path label for the frontend. Diffusers T2I can now run through the `diffusers_flux2` image provider when the optional FLUX dependencies and model files are available.

## Backend

```bash
cd services/backend
cp .env.example .env
# Edit .env and set HF_TOKEN only if you want future Hugging Face downloads.
uv sync
uv run uvicorn app.main:app --reload --host 127.0.0.1 --port 17321
```

Optional FLUX dependencies:

```bash
cd services/backend
uv sync --extra flux
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

Codex model labels are configured in `.env` with `CODEX_MODEL_OPTIONS`, while the active default is `CODEX_DEFAULT_MODEL`. The default picker follows the visible Codex CLI catalog from `cmd /c codex debug models`; current choices are `gpt-5.5`, `gpt-5.4`, `gpt-5.4-mini`, `gpt-5.3-codex`, `gpt-5.3-codex-spark`, and `gpt-5.2`.

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
- If a questionnaire creation turn fails because the provider returns an error, or if an agent response fails strict schema validation even after one repair attempt, the backend returns a safe single-question text questionnaire so the user can continue manually instead of losing the flow.
- The Manager drawer now reads local model status, skill versions, template versions, and recent logs from backend APIs. It can show FLUX/Hugging Face readiness, set a FLUX model path through the Tauri folder picker or manual browser fallback, mark FLUX install pending when `HF_TOKEN` is configured, and unload the FLUX placeholder without exposing the token or full local model path. Patch proposal APIs now support reviewable diff text plus optional proposed content; approving a proposal with an item id and proposed content creates a new SQLite-backed skill/template version.
- The FLUX install endpoint downloads `FLUX_MODEL_REPO_ID` into `FLUX_MODEL_LOCAL_DIR` when `HF_TOKEN` has access. The default is `black-forest-labs/FLUX.2-klein-9b-fp8`, which provides a single-file FP8 transformer checkpoint. The `diffusers_flux2` image provider loads that checkpoint with `Flux2Transformer2DModel.from_single_file`, then uses `FLUX_PIPELINE_REPO_ID=black-forest-labs/FLUX.2-klein-9b` for the remaining Diffusers pipeline components/config. The core diffusion weights stay FP8.
- BFL's FP8 checkpoint includes scalar `input_scale` / `weight_scale` metadata tensors. The provider filters those metadata keys before passing the checkpoint into Diffusers so the converter does not treat them as qkv weights.
- FLUX.2 Klein 9B FP8 is large but substantially lighter than the BF16 transformer. `FLUX_DEVICE_MAP=balanced` is the recommended CUDA setup on a 4090; the provider avoids Diffusers `device_map=balanced` for the custom FP8 transformer and instead enables `enable_model_cpu_offload()` so components move to CUDA when needed without CPU/CUDA tensor mismatches. `FLUX_DEVICE_MAP=cuda` tries to move the loaded pipeline onto CUDA directly and may run out of VRAM.
- `CORS_ALLOW_ORIGINS` is comma-separated and parsed with `NoDecode`.
- `CODEX_MODEL_OPTIONS` is comma-separated and parsed with `NoDecode`.
- Tauri development requires Rust/Cargo. Install the Rust MSVC toolchain before running `npm run tauri:dev`. The FLUX path picker uses Tauri's official dialog plugin; normal browser preview keeps a manual path fallback.

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
- `GET /models/flux/readiness`
- `POST /models/flux/set-path`
- `POST /models/flux/install`
- `POST /models/flux/unload`
- `GET /settings/safe`
- `PATCH /settings/safe`
- `GET /security/secrets/status`
- `GET /skills`
- `GET /skills/{skill_id}`
- `POST /skills/patch-proposals`
- `GET /skills/patch-proposals`
- `POST /skills/patch-proposals/{proposal_id}/approve`
- `POST /skills/patch-proposals/{proposal_id}/reject`
- `GET /templates`
- `GET /templates/{template_id}`
- `POST /templates/patch-proposals`
- `GET /templates/patch-proposals`
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
