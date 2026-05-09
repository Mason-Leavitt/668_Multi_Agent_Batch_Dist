from typing import Literal, TypedDict


ExperimentFollowupIntent = Literal[
    "select_option",
    "try_custom_value",
    "shift_samples_higher",
    "shift_samples_lower",
    "switch_sampling_axis",
    "end_experiment",
    "explain_option",
    "unknown",
]

ExperimentFollowupConfidence = Literal["explicit", "interpreted", "low"]


class ExperimentIntentResult(TypedDict):
    is_experiment_followup: bool
    intent: ExperimentFollowupIntent
    option_index: int | None
    target_variable: str | None
    value: float | None
    relative_choice: str | None
    needs_clarification: bool
    clarification_question: str | None
    confidence: ExperimentFollowupConfidence


def unknown_experiment_intent() -> ExperimentIntentResult:
    return {
        "is_experiment_followup": False,
        "intent": "unknown",
        "option_index": None,
        "target_variable": None,
        "value": None,
        "relative_choice": None,
        "needs_clarification": False,
        "clarification_question": None,
        "confidence": "low",
    }


def build_experiment_intent(
    intent: ExperimentFollowupIntent,
    *,
    is_experiment_followup: bool = True,
    option_index: int | None = None,
    target_variable: str | None = None,
    value: float | None = None,
    relative_choice: str | None = None,
    needs_clarification: bool = False,
    clarification_question: str | None = None,
    confidence: ExperimentFollowupConfidence = "explicit",
) -> ExperimentIntentResult:
    return {
        "is_experiment_followup": is_experiment_followup,
        "intent": intent,
        "option_index": option_index,
        "target_variable": target_variable,
        "value": value,
        "relative_choice": relative_choice,
        "needs_clarification": needs_clarification,
        "clarification_question": clarification_question,
        "confidence": confidence,
    }
