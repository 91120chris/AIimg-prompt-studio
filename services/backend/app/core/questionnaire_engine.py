from app.schemas.errors import StructuredError
from app.schemas.questionnaire import ChoiceQuestion, Question, Questionnaire, ScaleQuestion
from app.schemas.questionnaire_answers import (
    BooleanQuestionnaireAnswer,
    ChoiceQuestionnaireAnswer,
    MultiChoiceQuestionnaireAnswer,
    QuestionnaireAnswer,
    QuestionnaireAnswerPayload,
    ScaleQuestionnaireAnswer,
    TextQuestionnaireAnswer,
)


class QuestionnaireValidationError(ValueError):
    def __init__(self, error: StructuredError) -> None:
        self.error = error
        super().__init__(error.message)


def validate_questionnaire_answers(
    questionnaire: Questionnaire,
    payload: QuestionnaireAnswerPayload,
) -> None:
    if payload.questionnaire_id != questionnaire.questionnaire_id:
        raise QuestionnaireValidationError(
            StructuredError(
                code="questionnaire_mismatch",
                message="答案所屬問卷與目前問卷不一致。",
                suggestion="重新整理問卷後再送出答案。",
            )
        )

    questions_by_id = {question.question_id: question for question in questionnaire.questions}
    answers_by_id = {answer.question_id: answer for answer in payload.answers}

    unknown_ids = sorted(set(answers_by_id) - set(questions_by_id))
    if unknown_ids:
        raise QuestionnaireValidationError(
            StructuredError(
                code="unknown_question",
                message=f"答案包含未知問題：{', '.join(unknown_ids)}。",
                suggestion="請重新載入問卷後再送出。",
            )
        )

    for question in questionnaire.questions:
        answer = answers_by_id.get(question.question_id)
        if question.required and answer is None:
            raise QuestionnaireValidationError(
                StructuredError(
                    code="required_answer_missing",
                    message=f"「{question.label}」是必填問題。",
                    suggestion="請補上必填答案後再送出。",
                )
            )
        if answer is not None:
            _validate_answer_matches_question(question, answer)


def _validate_answer_matches_question(question: Question, answer: QuestionnaireAnswer) -> None:
    if question.kind == "text":
        if not isinstance(answer, TextQuestionnaireAnswer) or not answer.value.strip():
            raise _invalid_answer(question, "請輸入文字答案。")
        if question.max_length is not None and len(answer.value) > question.max_length:
            raise _invalid_answer(question, f"答案長度不可超過 {question.max_length} 字。")
        return

    if question.kind == "boolean":
        if not isinstance(answer, BooleanQuestionnaireAnswer):
            raise _invalid_answer(question, "請選擇是或否。")
        return

    if question.kind == "scale":
        if not isinstance(answer, ScaleQuestionnaireAnswer):
            raise _invalid_answer(question, "請選擇量表數值。")
        _validate_scale_answer(question, answer)
        return

    if question.kind == "choice":
        _validate_choice_answer(question, answer)


def _validate_choice_answer(question: ChoiceQuestion, answer: QuestionnaireAnswer) -> None:
    allowed_values = {option.value for option in question.options}
    if question.allow_multiple:
        if not isinstance(answer, MultiChoiceQuestionnaireAnswer):
            raise _invalid_answer(question, "請至少選擇一個選項。")
        invalid_values = sorted(set(answer.values) - allowed_values)
        if invalid_values:
            raise _invalid_answer(question, f"未知選項：{', '.join(invalid_values)}。")
    else:
        if not isinstance(answer, ChoiceQuestionnaireAnswer):
            raise _invalid_answer(question, "請選擇一個選項。")
        if answer.value not in allowed_values:
            raise _invalid_answer(question, f"未知選項：{answer.value}。")


def _validate_scale_answer(question: ScaleQuestion, answer: ScaleQuestionnaireAnswer) -> None:
    if answer.value < question.min_value or answer.value > question.max_value:
        raise _invalid_answer(
            question,
            f"數值需介於 {question.min_value} 到 {question.max_value}。",
        )
    if (answer.value - question.min_value) % question.step != 0:
        raise _invalid_answer(question, f"數值需符合步進 {question.step}。")


def _invalid_answer(question: Question, message: str) -> QuestionnaireValidationError:
    return QuestionnaireValidationError(
        StructuredError(
            code="invalid_answer",
            message=f"「{question.label}」答案格式不正確。{message}",
            suggestion="請確認問卷答案後再送出。",
        )
    )
