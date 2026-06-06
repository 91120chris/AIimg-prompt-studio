# 專案任務：新增 LoRA 模型掛載功能

## 任務目標
為目前的 AIimg-prompt-studio 專案新增 LoRA 模型支援，讓使用者能在前端介面選擇本機端的 LoRA 模型並調整權重，進而影響 Local Flux 的圖像生成結果。

## 實作步驟清單

### 1. 後端實作：本機檔案掃描 API
- **目標**：讓前端能取得可用的 LoRA 模型列表。
- **任務**：
  - 確認或建立一個專屬資料夾路徑用於存放 LoRA 模型（例如 `models/loras/`）。
  - 新增一隻 API 路由（例如 `GET /api/models/loras`）。
  - 該 API 需掃描指定資料夾，過濾出附檔名為 `.safetensors` 或 `.pt` 的檔案。
  - 將檔名整理成格式化的資料（如 Array of Strings）回傳給前端。

### 2. 前端實作：UI 介面與狀態管理
- **目標**：在現有的「生成控制」面板中加入 LoRA 設定選項。
- **任務**：
  - 新增「LoRA 選擇器」元件（Dropdown），選項需在元件掛載時（On Mount）呼叫步驟 1 的 API 動態生成。
  - 新增「LoRA 權重控制」元件（Slider），數值範圍設定為 `0.0` 到 `1.0`（允許小數點後兩位），預設值可設為 `0.8`。
  - 確保使用者可以選擇「不使用 LoRA」（例如加入一個 None 的選項）。
  - 更新前端的發送邏輯：當使用者點擊「確認生成」時，確保傳送給後端的 JSON Payload 中，正確包含 `lora_name` 與 `lora_weight` 參數。

### 3. 後端實作：生成引擎整合與 VRAM 管理
- **目標**：修改現有的 Local Flux 生成邏輯，套用前端傳來的 LoRA 參數。
- **任務**：
  - 攔截並解析前端傳來的生成請求，提取 `lora_name` 與 `lora_weight`。
  - 根據專案目前對接 Local Flux 的方式（Python Diffusers、ComfyUI API 或其他框架），將掛載 LoRA 的語法加入生成 Pipeline 中。
  - **🚨 關鍵效能與穩定性要求（VRAM 記憶體管理）**：
    - 必須實作卸載（Unload）機制。當使用者切換不同的 LoRA，或是選擇不使用 LoRA 時，務必確保舊的 LoRA 權重已從顯示卡記憶體（VRAM）中完全釋放，避免連續生成導致 Out of Memory (OOM) 錯誤。