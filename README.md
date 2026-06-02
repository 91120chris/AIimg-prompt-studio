# AIimg Prompt Studio

[English version](README.en.md)

AIimg Prompt Studio 是一個 local-first 的桌面 prompt optimization studio。它把 Tauri + React 桌面介面、FastAPI 後端、Codex/Ollama agent loop、SQLite session storage、Local Flux 圖像生成工作流整合在同一個 monorepo 裡。目標是讓使用者能在本機完成 T2I / I2I prompt 優化、圖片生成、回饋修正，以及把成功或失敗經驗整理成可重用的 Template / Skill 提案。

目前狀態：

- 桌面端：Tauri v2 + React/Vite，可用固定三欄工作區操作 prompt、問卷、生成結果與 Manager。
- 後端：FastAPI + SQLite，提供 provider 狀態、session、檔案、generation、agent、registry APIs。
- Agent：支援 Codex CLI 與 Ollama。可產生問卷、優化 prompt、根據圖片回饋產生下一版 prompt。
- Local Flux：使用本機 ComfyUI 作為隱藏執行後端，app UI 中顯示為 `Local Flux`。
- Manager：管理 Models、Skills、Templates、Prompt Versions、Registry Proposals、Logs。
- Registry proposal loop：Agent 只能建立 pending proposal，使用者必須在 Manager 中 validate / approve / reject，才會正式寫入 template 或 skill version。

## 快速開始

建議啟動順序：

1. 啟動 ComfyUI，確認 `http://127.0.0.1:8188` 可連線。
2. 啟動 FastAPI backend。
3. 啟動 Tauri desktop app。

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

PowerShell 若阻擋 `npm.ps1` 或 `codex.ps1`，請優先使用：

```powershell
cmd /c npm run build
cmd /c npm run tauri:dev
cmd /c codex --version
cmd /c codex login status
```

## 系統需求

基本需求：

- Windows 10/11，建議 NVIDIA GPU。
- Node.js + npm。
- `uv`，後端會使用 Python `>=3.11,<3.14`。
- Rust MSVC toolchain，用於 Tauri desktop dev/build。
- Codex CLI，作為主要 agent provider。
- Ollama optional，作為 local LLM agent provider。
- ComfyUI，作為 Local Flux 圖像生成後端。

本專案不需要 OpenAI API key。Codex provider 使用本機 Codex CLI；Local Flux provider 使用本機 ComfyUI API。

## 安裝後端

```powershell
cd services/backend
copy .env.example .env
uv sync
uv run uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

重要環境變數：

```env
FRONTEND_API_BASE_URL=http://127.0.0.1:8000
LOCAL_FLUX_BASE_URL=http://127.0.0.1:8188
CODEX_BINARY_PATH=codex
CODEX_DEFAULT_MODEL=gpt-5.5
CODEX_MODEL_OPTIONS=gpt-5.5,gpt-5.4,gpt-5.4-mini,gpt-5.3-codex,gpt-5.3-codex-spark,gpt-5.2
OLLAMA_BASE_URL=http://localhost:11434
HF_TOKEN=
```

不要 commit `.env`。如果你曾經把 Hugging Face token 貼到公開位置，請到 Hugging Face settings 重新產生 token。

## 安裝前端 / 桌面端

```powershell
cd apps/desktop
cmd /c npm install
cmd /c npm run build
cmd /c npm run tauri:dev
```

備註：

- `npm run dev` 目前會先 build，再用 Vite preview 跑 `http://127.0.0.1:1420`，這是為了避開本機路徑含 `#` 時 Vite dev server 可能出現的問題。
- 如果你的專案路徑沒有特殊字元，也可以用 `cmd /c npm run dev:vite` 啟動一般 Vite dev server。
- Tauri file picker 只在 Tauri runtime 內可用；純 browser preview 仍可手動輸入 Local Flux 路徑。

## 安裝 ComfyUI

推薦 Windows 使用 [Tavris1/ComfyUI-Easy-Install](https://github.com/Tavris1/ComfyUI-Easy-Install)。它是 portable ComfyUI installer，包含 EZi Desktop app，主打免手動安裝 Python/Git dependencies，並支援 NVIDIA GPU 使用情境。

建議流程：

1. 到 [ComfyUI-Easy-Install releases](https://github.com/Tavris1/ComfyUI-Easy-Install/releases) 下載最新版本。
2. 解壓縮到一個乾淨的新資料夾。
3. 執行 `ComfyUI-Easy-Install.bat`。
4. 安裝完成後啟動 ComfyUI。
5. 打開 `http://127.0.0.1:8188`，確認 ComfyUI web UI 可用。

注意事項：

- 不要用 Administrator 執行 installer。
- 避免安裝在 `Program Files`、`Windows`、`C:\` root。
- 避免資料夾名稱含空白或特殊字元。
- 確認 NVIDIA driver 已更新。
- ComfyUI-Easy-Install 支援多份 portable install；必要時可獨立放一份給 AIimg Prompt Studio 使用。

## 下載 Flux 模型

Local Flux 目前使用 Flux 2 Klein 9B FP8 distilled workflow。請遵守 Hugging Face 模型頁與相關授權條款，尤其是 Flux dev / non-commercial license 提示；商業使用前請自行確認授權。

下載檔案與放置位置：

| 類型 | 檔案 | 下載 | 建議放置位置 |
| --- | --- | --- | --- |
| Diffusion model | `flux-2-klein-9b-fp8mixed.safetensors` | [silveroxides/FLUX.2-dev-fp8_scaled](https://huggingface.co/silveroxides/FLUX.2-dev-fp8_scaled/resolve/main/flux-2-klein-9b-fp8mixed.safetensors) | `ComfyUI/models/diffusion_models/flux2` |
| Text encoder | `qwen_3_8b_fp8mixed.safetensors` | [Comfy-Org/flux2-klein-9B](https://huggingface.co/Comfy-Org/flux2-klein-9B/resolve/main/split_files/text_encoders/qwen_3_8b_fp8mixed.safetensors) | `ComfyUI/models/text_encoders` |
| VAE | `flux2-vae.safetensors` | [Comfy-Org/flux2-dev](https://huggingface.co/Comfy-Org/flux2-dev/resolve/main/split_files/vae/flux2-vae.safetensors) | `ComfyUI/models/vae` |

已確認的公開資訊：

- `silveroxides/FLUX.2-dev-fp8_scaled` 模型頁標示 image-to-image、image-generation、ComfyUI、fp8_scaled，並提示 workflow assets / workflow 內含使用資訊。
- `qwen_3_8b_fp8mixed.safetensors` 在 Comfy-Org text encoder 檔案頁可見，大小約 8.66 GB。
- `flux2-vae.safetensors` 在 Comfy-Org VAE 檔案頁可見，大小約 336 MB。

本 repo 內已包含 Local Flux workflow JSON：

```text
workflow/Flux 2 Klein 9B FP8 (Distilled)/
  Flux 2 Klein 9B FP8 (Distilled) - Image Generation.json
  Flux 2 Klein 9B FP8 (Distilled) - One Image Edit.json
  Flux 2 Klein 9B FP8 (Distilled) - Two Images Edit.json
```

在 app 內點右上方 `Local Flux 設定` icon，可設定：

- ComfyUI server URL。
- T2I workflow path。
- I2I one-image / two-image workflow path。
- model / VAE / text encoder 路徑。
- width、height、seed、steps、cfg、sampler、scheduler、denoise、guidance、output prefix。

## 啟動與健康檢查

Backend：

```powershell
cd services/backend
uv run uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

確認 backend：

```powershell
curl http://127.0.0.1:8000/health
```

ComfyUI：

```powershell
curl http://127.0.0.1:8188/system_stats
```

Desktop：

```powershell
cd apps/desktop
cmd /c npm run tauri:dev
```

如果只想看 browser preview：

```powershell
cd apps/desktop
cmd /c npm run dev
```

## 使用教學

### T2I prompt optimization

1. 選擇 `T2I`。
2. 選擇 Agent provider：`Codex CLI` 或 `Ollama`。
3. 選擇 Template，或保留 `Auto-detect`。
4. 輸入 original prompt。
5. 按 `分析提示`。
6. Agent 產生 questionnaire。
7. 填寫問卷並送出。
8. Agent 產生 optimized prompt。
9. 選擇 image provider：`Codex Image` 或 `Local Flux`。
10. 按 `確認生成`。

### I2I reference image workflow

1. 選擇 `I2I`。
2. 上傳 1 到 2 張 reference images。
3. 選擇 `Local Flux`。
4. 確認 Local Flux settings 中已設定 I2I workflow。
5. 依照 T2I 流程產生 optimized prompt。
6. 按 `確認生成`。

Local Flux I2I 若沒有 reference image，前端會阻擋送出。後端會依 reference 數量使用 one-image 或 two-image workflow，並將 reference image upload 到 ComfyUI。

### Seed 行為

- Local Flux seed 空白或 `null` 時，每次生成都由後端產生新的 random seed。
- 明確填入 seed 時，會固定使用該 seed，方便重現。
- UI 會顯示圖片實際 seed，但不會自動把 random seed 寫回輸入框，避免下一次不小心被鎖定。

### Feedback / prompt refinement

圖片生成後不會自動重新建立 feedback 問卷。你可以手動按 `回饋 / 修正 Prompt`：

1. Agent 會根據 original prompt、current optimized prompt、生成圖片 metadata、selected template、enabled skills 建立 feedback questionnaire。
2. 填完問卷後，Agent 會建立新的 optimized prompt version。
3. 舊版本不會被覆蓋，可在 Manager 的 `Prompt` tab 切換。

## Agent 流程

完整 loop：

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

Agent context 來源：

- original prompt。
- current optimized prompt version。
- workflow mode：T2I 或 I2I。
- selected template JSON。
- enabled skills。
- reference image metadata。
- generated image metadata。
- feedback answers。
- 使用者勾選的 context checkbox。

Agent provider：

- Codex CLI：使用 Codex model 與 thinking level 設定。
- Ollama：從本機 `/api/tags` 即時讀取 installed models。

若 Agent 輸出無法通過 strict schema，後端會嘗試 repair；仍失敗時，會回傳安全的單題文字問卷，讓流程可以繼續。

## Template / Skill / Proposal Manager

Manager 抽屜包含：

- `Models`：顯示 Local Flux / provider 狀態。
- `Skills`：查看 skill content，切換 enabled / disabled。
- `Templates`：建立、複製、編輯、validate、preview template JSON。
- `Prompt`：查看 prompt versions，切換 current optimized prompt。
- `Proposals`：審核 Agent 建立的 template / skill proposals。
- `Logs`：查看近期 backend logs。

規則：

- Enabled skills 才會被注入 agent prompt。
- Disabled skills 完全不進 agent context。
- Template 必須支援目前 mode，例如 T2I template 不能送進 I2I workflow。
- Agent 不會直接改正式 template 或 skill。
- Agent 只能建立 pending proposal。
- 使用者必須在 Manager 中 edit / validate / approve / reject。
- Approve template proposal 會建立新的 template version。
- Approve new skill proposal 會建立新的 skill version，且預設 disabled。

## 專案架構

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

資料與安全策略：

- SQLite 儲存 session、prompt versions、generated images metadata、settings、registry versions、proposals。
- 圖片檔案透過 safe file URLs 提供，不回傳私人 storage path。
- `/health` 與 safe settings 不回傳 secret。
- `.env` 不應 commit。

## 測試與驗證

後端測試：

```powershell
cd services/backend
uv run pytest
```

前端 build：

```powershell
cd apps/desktop
cmd /c npm run build
```

Codex CLI 檢查：

```powershell
cmd /c codex --version
cmd /c codex login status
```

Ollama 檢查：

```powershell
ollama list
curl http://localhost:11434/api/tags
```

Local Flux 檢查：

```powershell
curl http://127.0.0.1:8188/system_stats
```

在 app 中也可以使用 `Local Flux 設定` 裡的 connection test 與 workflow validation。

## 疑難排解

### `npm.ps1` 或 `codex.ps1` 被 PowerShell 阻擋

使用 `cmd /c`：

```powershell
cmd /c npm run tauri:dev
cmd /c codex --version
```

### Vite 提示路徑含 `#`

目前 repo 路徑含 `C:\#My\...`。Vite 會警告特殊字元可能造成問題。本專案的 `npm run dev` 會先 build 再 preview，以降低這個問題。若要完全避免，請把 repo 移到不含 `#` 與空白的路徑。

### Port 被占用

常用 port：

- frontend：`1420`
- backend：`8000`
- ComfyUI：`8188`
- Ollama：`11434`

若 `1420` 被占用，先關掉舊的 Vite preview 或 Tauri dev process。

### Local Flux 生成出現 400 Bad Request

通常是 ComfyUI workflow 或模型路徑不匹配：

- 確認 ComfyUI 正在 `http://127.0.0.1:8188`。
- 確認 workflow path 指向 API-format workflow JSON。
- 確認 diffusion model、text encoder、VAE 已放到正確資料夾。
- 確認 Local Flux settings 裡的 model path/name 與 ComfyUI workflow node 可對應。
- 查看 ComfyUI terminal log，通常會指出缺少哪個 node、model 或 input。

### 問卷格式驗證失敗

後端會嘗試 repair agent output。若仍失敗，會降級成單題文字問卷，讓使用者手動補需求，流程不會中斷。

### HF token

`HF_TOKEN` 只放在 `.env`。不要寫入 README、source code、commit、issue 或 screenshot。

## 主要 Local APIs

常用 API：

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

## 未來計畫

- 更完整的 Local Flux workflow binding/import。
- Local Flux progress、cancel、logs。
- Template / skill version diff 與 rollback history。
- 更完整的 agent authoring UX。
- 更完整的 manager search/filter。
- Packaging 與 release installer。
- 更穩定的 workflow/model discovery。

## License / model notice

本 repo 的程式碼授權請以 repository license 為準。Flux / ComfyUI / Hugging Face 模型與 workflow 可能有各自授權或使用限制。下載與使用模型前，請自行確認對應模型頁的 license、usage policy 與商業使用限制。
