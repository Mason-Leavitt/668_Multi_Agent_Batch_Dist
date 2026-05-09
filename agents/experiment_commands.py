"""
Deterministic parser for explicit experiment follow-up commands.

This module returns a shared experiment-followup intent shape so a future LLM
fallback can emit the same schema, while execution remains deterministic.
"""

import re
from typing import Literal, TypedDict

from agents.session_commands import COMPOSITION_VARIABLES, normalize_variable_name


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


class ExperimentCommandResult(TypedDict):
    is_experiment_followup: bool
    intent: ExperimentFollowupIntent
    option_index: int | None
    target_variable: str | None
    value: float | None
    relative_choice: str | None
    needs_clarification: bool
    clarification_question: str | None
    confidence: ExperimentFollowupConfidence


def _unknown_intent() -> ExperimentCommandResult:
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


def _explicit_intent(
    intent: ExperimentFollowupIntent,
    *,
    option_index: int | None = None,
    target_variable: str | None = None,
    value: float | None = None,
    relative_choice: str | None = None,
) -> ExperimentCommandResult:
    return {
        "is_experiment_followup": True,
        "intent": intent,
        "option_index": option_index,
        "target_variable": target_variable,
        "value": value,
        "relative_choice": relative_choice,
        "needs_clarification": False,
        "clarification_question": None,
        "confidence": "explicit",
    }


def _parse_numeric_value(variable: str | None, raw_value: str) -> float | None:
    value_text = raw_value.strip().lower()
    if not value_text:
        return None
    if value_text.endswith("%"):
        if variable not in COMPOSITION_VARIABLES:
            return None
        try:
            return float(value_text[:-1].strip()) / 100.0
        except ValueError:
            return None
    try:
        return float(value_text)
    except ValueError:
        return None


def parse_experiment_command(user_message: str) -> ExperimentCommandResult:
    message = user_message.strip()
    lower_message = message.lower()

    if lower_message in {"done with this experiment", "finish experiment"}:
        return _explicit_intent("end_experiment")

    option_match = re.fullmatch(
        r"(use|choose)\s+option\s+(\d+)",
        message,
        flags=re.IGNORECASE,
    )
    if option_match:
        return _explicit_intent(
            "select_option",
            option_index=int(option_match.group(2)),
        )

    try_match = re.fullmatch(
        r"try\s+(.+?)(?:\s*=\s*|\s+)([-+]?\d*\.?\d+%?)",
        message,
        flags=re.IGNORECASE,
    )
    if try_match:
        variable = normalize_variable_name(try_match.group(1))
        value = _parse_numeric_value(variable, try_match.group(2))
        if variable is None or value is None:
            return _unknown_intent()
        return _explicit_intent(
            "try_custom_value",
            target_variable=variable,
            value=value,
        )

    higher_match = re.fullmatch(
        r"show\s+higher\s+(.+?)(?:\s+values?)?",
        message,
        flags=re.IGNORECASE,
    )
    if higher_match:
        variable = normalize_variable_name(higher_match.group(1))
        if variable is None:
            return _unknown_intent()
        return _explicit_intent(
            "shift_samples_higher",
            target_variable=variable,
        )

    lower_match = re.fullmatch(
        r"show\s+lower\s+(.+?)(?:\s+values?)?",
        message,
        flags=re.IGNORECASE,
    )
    if lower_match:
        variable = normalize_variable_name(lower_match.group(1))
        if variable is None:
            return _unknown_intent()
        return _explicit_intent(
            "shift_samples_lower",
            target_variable=variable,
        )

    compare_match = re.fullmatch(
        r"compare\s+(.+?)\s+instead",
        message,
        flags=re.IGNORECASE,
    )
    if compare_match:
        variable = normalize_variable_name(compare_match.group(1))
        if variable is None:
            return _unknown_intent()
        return _explicit_intent(
            "switch_sampling_axis",
            target_variable=variable,
        )

    return _unknown_intent()
