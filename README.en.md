# AIimg Prompt Studio

[中文版](README.md)

AIimg Prompt Studio is a local-first desktop prompt optimization studio. It combines a Tauri + React desktop UI, a FastAPI backend, Codex/Ollama agent loops, SQLite session storage, and Local Flux image generation workflows in one monorepo. The goal is to let users optimize T2I / I2I prompts, generate images locally, refine prompts from feedback, and turn successful or failed attempts into reusable Template / Skill proposals.

Current status:

- Desktop: Tauri v2 + React/Vite with a fixed three-column workspace for prompts, questionnaires, generated images, and Manager.
- Backend: FastAPI + SQLite APIs for providers, sessions, safe files, generation, agents, and registry management.
- Agent: supports Codex CLI and Ollama. It can create questionnaires, optimize prompts, and refine prompts from generated-image feedback.
- Local Flux: uses local ComfyUI as the hidden execution backend. The app-facing provider label is `Local Flux`.
- Manager: manages Models, Skills, Templates, Prompt Versions, Registry Proposals, and Logs.
- Registry proposal loop: the agent can only create pending proposals. The user must validate / approve / reject them in Manager before a template or skill version becomes official.

## Quick Start

Recommended startup order:

1. Start ComfyUI and confirm `http://127.0.0.1:8188` is reachable.
2. Start the FastAPI backend.
3. Start the Tauri desktop app.

```powershell
# 1. Backend
cd "C:\#My\coding\python\AIimg prompt studio\services\backend"
uv sync
uv run uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

```powershell
# 2. Desktop
cd "C:\#My\coding\python\AIimg prompt studio\apps\desktop"
cmd /c npm install
cmd /c npm run tauri:dev
```

If PowerShell blocks `npm.ps1` or `codex.ps1`, prefer:

```powershell
cmd /c npm run build
cmd /c npm run tauri:dev
cmd /c codex --version
cmd /c codex login status
```

## Requirements

Base requirements:

- Windows 10/11, preferably with an NVIDIA GPU.
- Node.js + npm.
- `uv`; the backend uses Python `>=3.11,<3.14`.
- Rust MSVC toolchain for Tauri desktop dev/build.
- Codex CLI as the primary agent provider.
- Ollama optional, as a local LLM agent provider.
- ComfyUI as the Local Flux image backend.

This project does not require an OpenAI API key. The Codex provider uses the local Codex CLI; the Local Flux provider uses the local ComfyUI API.

## Backend Setup

```powershell
cd services/backend
copy .env.example .env
uv sync
uv run uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Important environment variables:

```env
FRONTEND_API_BASE_URL=http://127.0.0.1:8000
LOCAL_FLUX_BASE_URL=http://127.0.0.1:8188
CODEX_BINARY_PATH=codex
CODEX_DEFAULT_MODEL=gpt-5.5
CODEX_MODEL_OPTIONS=gpt-5.5,gpt-5.4,gpt-5.4-mini,gpt-5.3-codex,gpt-5.3-codex-spark,gpt-5.2
OLLAMA_BASE_URL=http://localhost:11434
HF_TOKEN=
```

Do not commit `.env`. If you ever paste a Hugging Face token into a public place, rotate it in Hugging Face settings.

## Frontend / Desktop Setup

```powershell
cd apps/desktop
cmd /c npm install
cmd /c npm run build
cmd /c npm run tauri:dev
```

Notes:

- `npm run dev` currently builds first and then serves `dist/` through Vite preview at `http://127.0.0.1:1420`; this avoids Vite dev-server issues when the local path contains `#`.
- If your project path has no special characters, you can use `cmd /c npm run dev:vite` for the normal Vite dev server.
- Tauri file pickers work only inside the Tauri runtime; browser preview still supports manual Local Flux path inputs.

## Installing ComfyUI

For Windows, the recommended option is [Tavris1/ComfyUI-Easy-Install](https://github.com/Tavris1/ComfyUI-Easy-Install). It is a portable ComfyUI installer with an EZi Desktop app, designed to avoid manual Python/Git dependency setup and to support NVIDIA GPU use cases.

Suggested flow:

1. Download the latest release from [ComfyUI-Easy-Install releases](https://github.com/Tavris1/ComfyUI-Easy-Install/releases).
2. Extract it into a clean new folder.
3. Run `ComfyUI-Easy-Install.bat`.
4. Start ComfyUI after installation.
5. Open `http://127.0.0.1:8188` and confirm the ComfyUI web UI is available.

Important notes:

- Do not run the installer as Administrator.
- Avoid `Program Files`, `Windows`, and the `C:\` root.
- Avoid folder names with spaces or special characters.
- Keep NVIDIA drivers up to date.
- Multiple portable installs are supported; if useful, keep a dedicated ComfyUI install for AIimg Prompt Studio.

## Downloading Flux Models

Local Flux currently uses the Flux 2 Klein 9B FP8 distilled workflow. Please follow the Hugging Face model pages and their licenses, especially Flux dev / non-commercial license notices; verify the license yourself before commercial use.

Files and target locations:

| Type | File | Download | Suggested location |
| --- | --- | --- | --- |
| Diffusion model | `flux-2-klein-9b-fp8mixed.safetensors` | [silveroxides/FLUX.2-dev-fp8_scaled](https://huggingface.co/silveroxides/FLUX.2-dev-fp8_scaled/resolve/main/flux-2-klein-9b-fp8mixed.safetensors) | `ComfyUI/models/diffusion_models/flux2` |
| Text encoder | `qwen_3_8b_fp8mixed.safetensors` | [Comfy-Org/flux2-klein-9B](https://huggingface.co/Comfy-Org/flux2-klein-9B/resolve/main/split_files/text_encoders/qwen_3_8b_fp8mixed.safetensors) | `ComfyUI/models/text_encoders` |
| VAE | `flux2-vae.safetensors` | [Comfy-Org/flux2-dev](https://huggingface.co/Comfy-Org/flux2-dev/resolve/main/split_files/vae/flux2-vae.safetensors) | `ComfyUI/models/vae` |

Public information checked:

- `silveroxides/FLUX.2-dev-fp8_scaled` is tagged for image-to-image, image generation, ComfyUI, and fp8_scaled, and its model page points users to workflow assets / workflow instructions.
- `qwen_3_8b_fp8mixed.safetensors` is visible on the Comfy-Org text encoder file page, about 8.66 GB.
- `flux2-vae.safetensors` is visible on the Comfy-Org VAE file page, about 336 MB.

This repo includes Local Flux workflow JSON:

```text
workflow/Flux 2 Klein 9B FP8 (Distilled)/
  Flux 2 Klein 9B FP8 (Distilled) - Image Generation.json
  Flux 2 Klein 9B FP8 (Distilled) - One Image Edit.json
  Flux 2 Klein 9B FP8 (Distilled) - Two Images Edit.json
```

In the app, click the `Local Flux settings` icon in the top toolbar to configure:

- ComfyUI server URL.
- T2I workflow path.
- I2I one-image / two-image workflow paths.
- Model / VAE / text encoder paths.
- Width, height, seed, steps, cfg, sampler, scheduler, denoise, guidance, and output prefix.

## Startup and Health Checks

Backend:

```powershell
cd services/backend
uv run uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Check backend:

```powershell
curl http://127.0.0.1:8000/health
```

ComfyUI:

```powershell
curl http://127.0.0.1:8188/system_stats
```

Desktop:

```powershell
cd apps/desktop
cmd /c npm run tauri:dev
```

Browser preview only:

```powershell
cd apps/desktop
cmd /c npm run dev
```

## Usage Guide

### T2I prompt optimization

1. Select `T2I`.
2. Select an agent provider: `Codex CLI` or `Ollama`.
3. Select a Template, or keep `Auto-detect`.
4. Enter the original prompt.
5. Click `Analyze prompt`.
6. The agent creates a questionnaire.
7. Fill in and submit the questionnaire.
8. The agent creates an optimized prompt.
9. Select an image provider: `Codex Image` or `Local Flux`.
10. Click `Confirm generation`.

### I2I reference image workflow

1. Select `I2I`.
2. Upload 1 to 2 reference images.
3. Select `Local Flux`.
4. Confirm that Local Flux settings point to the correct I2I workflows.
5. Follow the T2I flow to create an optimized prompt.
6. Click `Confirm generation`.

For Local Flux I2I, the frontend blocks generation if there are no reference images. The backend chooses the one-image or two-image workflow based on the reference count and uploads reference images to ComfyUI.

### Seed behavior

- If the Local Flux seed is blank or `null`, the backend generates a new random seed for each generation.
- If a seed is explicitly provided, that seed is used for reproducibility.
- The UI shows the actual seed for generated images, but it does not write a random seed back into the input box, so the next run is not accidentally locked.

### Feedback / prompt refinement

Image generation does not automatically restart a feedback questionnaire. Click `Feedback / Fix Prompt` manually:

1. The agent uses the original prompt, current optimized prompt, generated image metadata, selected template, and enabled skills to create a feedback questionnaire.
2. After the questionnaire is submitted, the agent creates a new optimized prompt version.
3. Older versions are preserved and can be selected in Manager's `Prompt` tab.

## Agent Flow

Full loop:

```text
original prompt
  -> questionnaire
  -> questionnaire answers
  -> optimized prompt
  -> prompt version
  -> confirm generation
  -> generated image
  -> optional feedback questionnaire
  -> refined prompt version
```

Agent context sources:

- Original prompt.
- Current optimized prompt version.
- Workflow mode: T2I or I2I.
- Selected template JSON.
- Enabled skills.
- Reference image metadata.
- Generated image metadata.
- Feedback answers.
- User-selected context checkboxes.

Agent providers:

- Codex CLI: uses the selected Codex model and thinking-level settings.
- Ollama: discovers locally installed models live from `/api/tags`.

If agent output fails strict schema validation, the backend attempts one repair pass. If repair still fails, it falls back to a safe single text-question questionnaire so the user can continue the flow.

## Template / Skill / Proposal Manager

Manager drawer tabs:

- `Models`: provider and Local Flux status.
- `Skills`: inspect skill content and toggle enabled / disabled.
- `Templates`: create, duplicate, edit, validate, and preview template JSON.
- `Prompt`: inspect prompt versions and switch the current optimized prompt.
- `Proposals`: review template / skill proposals created by the agent.
- `Logs`: recent backend logs.

Rules:

- Only enabled skills are injected into the agent prompt.
- Disabled skills are completely excluded from agent context.
- A template must support the current mode; for example, a T2I-only template cannot be used in an I2I workflow.
- The agent cannot directly change official templates or skills.
- The agent can only create pending proposals.
- The user must edit / validate / approve / reject proposals in Manager.
- Approving a template proposal creates a new template version.
- Approving a new skill proposal creates a new skill version, disabled by default.

## Project Structure

```text
apps/desktop
  Tauri v2 + React/Vite desktop app
  src/App.tsx
  src-tauri/

services/backend
  FastAPI backend
  app/api/
  app/core/
  app/db/
  app/providers/
  app/schemas/
  tests/

registries
  skills/
  templates/

workflow
  Flux 2 Klein 9B FP8 (Distilled) workflow JSON
```

Data and safety policies:

- SQLite stores sessions, prompt versions, generated image metadata, settings, registry versions, and proposals.
- Image files are exposed through safe file URLs; private storage paths are not returned.
- `/health` and safe settings never return secrets.
- `.env` should never be committed.

## Tests and Verification

Backend tests:

```powershell
cd services/backend
uv run pytest
```

Frontend build:

```powershell
cd apps/desktop
cmd /c npm run build
```

Codex CLI checks:

```powershell
cmd /c codex --version
cmd /c codex login status
```

Ollama checks:

```powershell
ollama list
curl http://localhost:11434/api/tags
```

Local Flux check:

```powershell
curl http://127.0.0.1:8188/system_stats
```

The app's `Local Flux settings` drawer also has connection test and workflow validation controls.

## Troubleshooting

### `npm.ps1` or `codex.ps1` is blocked by PowerShell

Use `cmd /c`:

```powershell
cmd /c npm run tauri:dev
cmd /c codex --version
```

### Vite warns that the path contains `#`

The current repo path contains `C:\#My\...`. Vite warns that special characters may cause issues. This project's `npm run dev` builds first and then serves a preview to reduce the impact. To avoid the warning entirely, move the repo to a path without `#` or spaces.

### Port already in use

Common ports:

- frontend: `1420`
- backend: `8000`
- ComfyUI: `8188`
- Ollama: `11434`

If `1420` is occupied, close the old Vite preview or Tauri dev process.

### Local Flux generation returns 400 Bad Request

This usually means the ComfyUI workflow or model paths do not match:

- Confirm ComfyUI is running at `http://127.0.0.1:8188`.
- Confirm the workflow path points to an API-format workflow JSON.
- Confirm the diffusion model, text encoder, and VAE are in the correct folders.
- Confirm Local Flux settings match the model names or paths expected by the workflow nodes.
- Check the ComfyUI terminal log; it usually points to the missing node, model, or input.

### Questionnaire schema validation fails

The backend attempts to repair the agent output. If it still fails, it falls back to a single text-question questionnaire so the user can manually provide the missing requirements and continue.

### HF token

`HF_TOKEN` belongs only in `.env`. Do not put it in README, source files, commits, issues, or screenshots.

## Main Local APIs

Common APIs:

- `GET /health`
- `GET /settings/safe`
- `GET /providers/codex/status`
- `GET /providers/codex/models`
- `GET /providers/ollama/status`
- `GET /providers/ollama/models`
- `GET /providers/local-flux/status`
- `GET /providers/local-flux/settings`
- `PATCH /providers/local-flux/settings`
- `POST /providers/local-flux/workflows/validate`
- `POST /sessions`
- `GET /sessions`
- `GET /sessions/{session_id}/prompt-versions`
- `PATCH /sessions/{session_id}/current-prompt-version`
- `POST /agent/turn`
- `POST /agent/answer-questionnaire`
- `POST /agent/feedback-questionnaire`
- `POST /agent/refine`
- `POST /agent/registry-proposals`
- `GET /registry/patch-proposals`
- `PATCH /registry/patch-proposals/{proposal_id}`
- `POST /registry/patch-proposals/{proposal_id}/validate`
- `POST /registry/patch-proposals/{proposal_id}/approve`
- `POST /registry/patch-proposals/{proposal_id}/reject`
- `POST /generation/confirm`
- `GET /generation/{job_id}`

## Roadmap

- More complete Local Flux workflow binding/import.
- Local Flux progress, cancel, and logs.
- Template / skill version diff and rollback history.
- More complete agent authoring UX.
- More complete manager search/filter.
- Packaging and release installer.
- More stable workflow/model discovery.

## License / model notice

Use this repository's code according to the repository license. Flux, ComfyUI, Hugging Face models, and workflows may have their own licenses or usage limits. Before downloading or using any model, verify the corresponding model page's license, usage policy, and commercial-use restrictions.
