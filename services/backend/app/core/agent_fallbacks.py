from app.core.session_workspace import new_id
from app.schemas.agent import QuestionnaireTurnResponse
from app.schemas.questionnaire import Questionnaire, TextQuestion


def fallback_text_questionnaire(provider_label: str, reason: str) -> QuestionnaireTurnResponse:
    return QuestionnaireTurnResponse(
        kind="questionnaire",
        message=(
            f"{provider_label} 回覆格式未通過驗證，已切換成手動補充問卷。"
            "請直接用文字補上需求，流程可以繼續。"
        ),
        questionnaire=Questionnaire(
            questionnaire_id=new_id("q_fallback"),
            title="手動補充問卷",
            description="模型未能產生可驗證的結構化問卷，因此改用單題文字問卷。",
            questions=[
                TextQuestion(
                    kind="text",
                    question_id="manual_details",
                    label="補充需求",
                    prompt=(
                        "請描述你想補充或修正的重點，例如主體、風格、構圖、色彩、"
                        "光線、材質、比例、負面限制，或任何必須保留/避免的細節。"
                    ),
                    required=True,
                    placeholder="例如：更寫實、柔和逆光、主體置中、避免文字錯誤...",
                    max_length=4000,
                )
            ],
        ),
        warnings=[f"Fallback reason: {reason[:300]}"],
    )
