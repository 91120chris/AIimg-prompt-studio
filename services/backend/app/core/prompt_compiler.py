import json

from app.schemas.agent import AgentTurnRequest
from app.schemas.questionnaire import Questionnaire
from app.schemas.questionnaire_answers import QuestionnaireAnswerPayload


def _json_dump(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def build_questionnaire_prompt(payload: AgentTurnRequest) -> str:
    mode_label = "image-to-image" if payload.mode == "i2i" else "text-to-image"
    return f"""你是 Prompt Optimizer Studio 的本機 prompt agent。

任務：分析使用者的原始影像 prompt，產生下一步需要的互動問卷。

硬性規則：
- 只回傳符合 output schema 的 JSON，不要加 Markdown。
- output schema 會要求所有欄位都出現；不適用的欄位請填 null，warnings 請填陣列。
- 問題物件也要填滿所有 schema 欄位；不適用的欄位請填 null。
- 原始 prompt 已經在下方提供；不要要求使用者再貼一次 prompt。
- 這是問卷產生回合，kind 必須是 "questionnaire"。
- 不要要求 OpenAI API key。
- 不要產生圖片，也不要說你已經產生圖片。
- 不要修改技能、模板或模型 registry；如果需要，只能把它當成後續建議。
- 參考圖片會由系統另外處理，不要把本機檔案路徑寫進 prompt。
- 問卷請使用繁體中文，問題數量以 3 到 5 題為主。
- 問卷至少 3 題，最多 5 題，優先詢問構圖、風格、主體細節、色彩/光線、限制條件。

工作模式：{mode_label}
原始 prompt：
{payload.original_prompt}
"""


def build_optimization_prompt(
    original_prompt: str,
    questionnaire: Questionnaire,
    answers: QuestionnaireAnswerPayload,
) -> str:
    return f"""你是 Prompt Optimizer Studio 的本機 prompt agent。

任務：根據原始 prompt、問卷與使用者答案，輸出最佳化後的影像生成 prompt。

硬性規則：
- 只回傳符合 output schema 的 JSON，不要加 Markdown。
- output schema 會要求所有欄位都出現；不適用的欄位請填 null，warnings 請填陣列。
- 如果回傳問卷或問題物件，也要填滿所有 schema 欄位；不適用的欄位請填 null。
- 回傳 kind 必須是 "optimized_prompt"，除非輸入不足到無法繼續。
- 不要產生圖片，也不要說你已經產生圖片。
- 不要把參考圖片路徑寫進 prompt。
- 最佳化 prompt 要可直接交給後續 image provider 使用。
- 使用繁體中文說明 message；optimized_prompt 可混合英文關鍵詞，但要清楚可用。

原始 prompt：
{original_prompt}

問卷：
{_json_dump(questionnaire.model_dump())}

使用者答案：
{_json_dump(answers.model_dump())}
"""


def build_feedback_questionnaire_prompt(
    *,
    original_prompt: str,
    optimized_prompt: str,
    generation_job: dict[str, object],
    generated_images: list[dict[str, object]],
) -> str:
    return f"""你是 Prompt Optimizer Studio 的本機 prompt agent。

任務：圖片生成已完成。請建立一份給使用者填寫的回饋問卷，用來改善下一版 prompt。

硬性規則：
- 只回傳符合 output schema 的 JSON，不要加 Markdown。
- output schema 會要求所有欄位都出現；不適用的欄位請填 null，warnings 請填陣列。
- 問題物件也要填滿所有 schema 欄位；不適用的欄位請填 null。
- 這是生成後回饋問卷回合，kind 必須是 "questionnaire"。
- 不要產生圖片，也不要說你已經產生圖片。
- 不要修改技能、模板或模型 registry。
- 不要要求 OpenAI API key。
- 不要要求使用者貼本機檔案路徑。
- 問卷請使用繁體中文，3 到 5 題。
- 問題應協助使用者檢查：是否符合需求、要保留什麼、要修正什麼、畫面/風格/文字/構圖問題、下一輪限制。

原始 prompt：
{original_prompt}

最佳化 prompt：
{optimized_prompt}

生成工作 safe metadata：
{_json_dump(generation_job)}

生成圖片 safe metadata：
{_json_dump(generated_images)}
"""


def build_feedback_refinement_prompt(
    *,
    original_prompt: str,
    previous_optimized_prompt: str,
    questionnaire: Questionnaire,
    answers: QuestionnaireAnswerPayload,
    generation_job: dict[str, object],
    generated_images: list[dict[str, object]],
) -> str:
    return f"""你是 Prompt Optimizer Studio 的本機 prompt agent。

任務：根據使用者對上一輪生成圖片的回饋，產生下一版最佳化 prompt。

硬性規則：
- 只回傳符合 output schema 的 JSON，不要加 Markdown。
- output schema 會要求所有欄位都出現；不適用的欄位請填 null，warnings 請填陣列。
- 這是回饋修正回合，kind 必須是 "optimized_prompt"。
- 不要產生圖片，也不要說你已經產生圖片。
- 不要修改技能、模板或模型 registry。
- 不要要求 OpenAI API key。
- 不要要求使用者貼本機檔案路徑。
- 不要覆寫上一版 prompt；請輸出一個可直接用於下一輪生成的新 prompt。
- 保留使用者明確滿意的元素，修正使用者指出的問題，並避免放大未提到的風格偏差。
- optimized_prompt 應完整、具體、可供 image provider 直接使用。

原始 prompt：
{original_prompt}

上一版最佳化 prompt：
{previous_optimized_prompt}

生成工作 safe metadata：
{_json_dump(generation_job)}

生成圖片 safe metadata：
{_json_dump(generated_images)}

回饋問卷：
{_json_dump(questionnaire.model_dump())}

使用者回饋答案：
{_json_dump(answers.model_dump())}
"""


def build_repair_prompt(raw_output: str, validation_error: str) -> str:
    return f"""上一個回覆沒有通過 Prompt Optimizer Studio 的 strict JSON schema 驗證。

請修正為單一 JSON 物件，並且只輸出 JSON，不要加 Markdown 或解釋。

驗證錯誤：
{validation_error}

上一個回覆：
{raw_output}
"""
