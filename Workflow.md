# AIimg Prompt Studio - Agent 協作紀錄 (Workflow Log)

本文件紀錄開發 **AIimg Prompt Studio** 期間，開發者與 AI Agent 進行結對程式設計（AI Pair Programming）的完整協作歷程。文件依循專題執行的四個階段，詳細記錄了使用的工具組合、關鍵 Prompt、互動過程，以及 Agent 協助解決的核心技術問題。

---

## 一、 使用的工具組合

本專案採用 Local-First 架構，在開發過程中結合了以下工具鏈與 Agent 進行協作：
* **核心開發 Agent**：
  * **Claude Code (CLI)**：負責讀取本機檔案、執行終端機指令、理解專案架構並直接進行原始碼重構與環境建置。
  * **Codex CLI / Ollama**：不僅作為專案系統內的提示詞優化引擎，在開發期間也輔助生成複雜的後端推論腳本與資料結構。
* **技術堆疊**：Tauri v2 + React/Vite (前端)、FastAPI + SQLite (後端)、本機 ComfyUI API / Local Flux (圖像推論後端)。

---

## 二、 專題執行步驟與關鍵 Prompt 紀錄 (Agent Workflow)

### 階段一：發想與企劃
* **目標**：確立專題目標、核心功能與預期整合之技術堆疊。
* **關鍵 Prompt (Context 賦予)**：
  > 「我需要建立一個 Local-First 的桌面應用，旨在解決 AI 繪圖中提示詞難以追蹤、迭代與重用的問題。系統需要整合提示詞優化、本機圖像生成與知識庫管理。請幫我生成一份專題提案，確立核心功能與建議的技術堆疊。」
* **Agent 互動與生成過程**：
  Agent 協助梳理出核心價值為「將 Prompt Engineering 視為專案工作流」，並提議加入問卷系統、提示詞版本控制與 Skills/Templates 註冊表。技術堆疊上，Agent 推薦採用 Monorepo 架構（Tauri + FastAPI）並介接 ComfyUI 作為推論後端。

### 階段二：架構設計與任務拆解
* **目標**：將題目拆解為實作任務，定義系統架構、前後端介接方式與 API 資料交換格式。
* **關鍵 Prompt**：
  > 「請將第一階段的企劃拆解為具體的實作任務，為 `apps/desktop` 與 `services/backend` 規劃資料夾結構。請定義前端與 FastAPI 後端介接的 RESTful API 藍圖，特別是 `/agent/turn` (處理問卷/提示詞) 以及 `/generation/confirm` (觸發算圖) 的資料交換 Schema。」
* **Agent 互動與生成過程**：
  Agent 生成了完整的 API 規格與 Pydantic schemas。針對動態模型需求，Agent 規劃了 `GET /providers/local-flux/loras` API，並制定了讀取 `.env`（`LOCAL_FLUX_LORA_DIR`）的安全規範，確保大型 `.safetensors` 檔案不會被錯誤加入 Git 版本控制。

### 階段三：程式碼生成與實作
* **目標**：擔任系統規劃者，引導 Agent 撰寫推論腳本並處理除錯。
* **關鍵 Prompt**：
  > 「請利用 Python (>=3.11) 與 FastAPI 實作 Local Flux 的推論腳本。需攔截前端傳來的 `prompt`, `seed`, `lora_name`, `lora_weight` 參數，將其動態 Patch 寫入 ComfyUI 的 JSON 工作流中並發送。🚨 注意：實作切換 LoRA 邏輯時，務必呼叫 ComfyUI 的 `/free` API 釋放 VRAM 以避免 OOM。」
* **Agent 互動與生成過程**：
  Agent 成功實作了將前端設定轉化為 ComfyUI 節點格式的腳本。在實作過程中，Agent 獨立生成了「兩階段技能路由器 (Two-Stage Skill Router)」，透過演算法初步評分篩選，確保傳遞給 LLM 的 Context 保持精簡。

### 階段四：介面封裝與總結
* **目標**：整合後端為可互動之 App，並利用 Agent 輔助撰寫技術文件。
* **關鍵 Prompt**：
  > 「請不使用 Gradio/Streamlit，而是直接生成符合 Tauri 架構的 React 前端介面。需要實作固定的三欄式工作區，並在頂部全域導覽列加上 LoRA 選擇器與權重滑桿 (-5.0 到 +5.0)。完成後，請根據目前專案架構，生成一份完整的 `README.md`，包含安裝依賴與 ComfyUI 的設定教學。」
* **Agent 互動與生成過程**：
  Agent 成功將後端推論邏輯封裝為具備 Manager 抽屜與即時預覽的高互動性桌面 App。最終，Agent 匯整了前後端啟動指令、環境變數 (`.env`) 規範與 API 列表，產出了符合開源標準的技術文件。

---

## 三、 Agent 協助解決之技術問題

在上述開發流程中，透過與 Agent 密集互動，成功克服了以下三大核心技術瓶頸：

1. **顯示卡記憶體 (VRAM) 溢位管理 (OOM 防禦)**
   * **問題**：在階段三實作 LoRA 下拉選單切換時，舊模型權重殘留導致生成崩潰。
   * **Agent 協助**：在後端 API 中引入了強制卸載機制，攔截切換事件並呼叫 ComfyUI 的清理端點，實作了資源的安全調度。

2. **Agent 結構化輸出的穩定性與容錯**
   * **問題**：依賴 LLM (Codex/Ollama) 輸出 JSON 時，常遇到格式跑版導致前端解析失敗。
   * **Agent 協助**：在 FastAPI 實作了「Strict Schema 驗證與自動修復 (Repair)」機制。當修復失敗時，Agent 寫出了優雅降級的邏輯，將流程退回單題文字問卷，確保系統不崩潰。
   
3. **Context Window 管理與 Token 爆量**
   * **問題**：專案累積過多 Skills 時，全文注入 Prompt 會嚴重消耗 Token 並引發幻覺。
   * **Agent 協助**：Agent 設計並實作了「兩階段技能路由器」，先透過後端進行啟發式評分摘要，再讓 Agent 挑選需要全文讀取的技能，完美解決了上下文過載的問題。