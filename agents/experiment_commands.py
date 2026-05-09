"""
Deterministic parser for explicit experiment follow-up commands.

This module returns a shared experiment-followup intent shape so a future LLM
fallback can emit the same schema, while execution remains deterministic.
"""

import re

from agents.experiment_intents import (
    ExperimentIntentResult,
    build_experiment_intent,
    unknown_experiment_intent,
)
from agents.session_commands import COMPOSITION_VARIABLES, normalize_variable_name


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


def parse_experiment_command(user_message: str) -> ExperimentIntentResult:
    message = user_message.strip()
    lower_message = message.lower()

    if lower_message in {"done with this experiment", "finish experiment", "end this experiment"}:
        return build_experiment_intent("end_experiment")

    explain_option_match = re.fullmatch(
        r"explain\s+option\s+(\d+)",
        message,
        flags=re.IGNORECASE,
    )
    if explain_option_match:
        return build_experiment_intent(
            "explain_option",
            option_index=int(explain_option_match.group(1)),
        )

    option_match = re.fullmatch(
        r"(use|choose)\s+option\s+(\d+)",
        message,
        flags=re.IGNORECASE,
    )
    if option_match:
        return build_experiment_intent(
            "select_option",
            option_index=int(option_match.group(2)),
        )

    relative_option_match = re.fullmatch(
        r"(choose|use|pick|go with)\s+(?:the\s+)?(first|second|middle|last)\s+(?:one|option|case)?",
        lower_message,
    )
    if relative_option_match:
        relative_word = relative_option_match.group(2)
        if relative_word == "second":
            return build_experiment_intent("select_option", option_index=2)
        return build_experiment_intent(
            "select_option",
            relative_choice=relative_word,
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
            return unknown_experiment_intent()
        return build_experiment_intent(
            "try_custom_value",
            target_variable=variable,
            value=value,
        )

    what_if_match = re.fullmatch(
        r"(?:what if|suppose)\s+(.+?)\s+(?:is|=)\s+([-+]?\d*\.?\d+%?)\??",
        message,
        flags=re.IGNORECASE,
    )
    if what_if_match:
        variable = normalize_variable_name(what_if_match.group(1))
        value = _parse_numeric_value(variable, what_if_match.group(2))
        if variable is None or value is None:
            return unknown_experiment_intent()
        return build_experiment_intent(
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
            return unknown_experiment_intent()
        return build_experiment_intent(
            "shift_samples_higher",
            target_variable=variable,
        )

    if lower_message in {
        "show me higher stopping compositions",
        "show higher stopping compositions",
    }:
        return build_experiment_intent(
            "shift_samples_higher",
            target_variable="xB",
        )

    lower_match = re.fullmatch(
        r"show\s+lower\s+(.+?)(?:\s+values?)?",
        message,
        flags=re.IGNORECASE,
    )
    if lower_match:
        variable = normalize_variable_name(lower_match.group(1))
        if variable is None:
            return unknown_experiment_intent()
        return build_experiment_intent(
            "shift_samples_lower",
            target_variable=variable,
        )

    if lower_message in {"give me lower xb cases", "show me lower xb cases"}:
        return build_experiment_intent(
            "shift_samples_lower",
            target_variable="xB",
        )

    compare_match = re.fullmatch(
        r"compare\s+(.+?)\s+instead",
        message,
        flags=re.IGNORECASE,
    )
    if compare_match:
        variable = normalize_variable_name(compare_match.group(1))
        if variable is None:
            return unknown_experiment_intent()
        return build_experiment_intent(
            "switch_sampling_axis",
            target_variable=variable,
        )

    if lower_message in {
        "vary feed composition instead",
        "compare starting concentrations",
        "vary feed composition",
    }:
        return build_experiment_intent(
            "switch_sampling_axis",
            target_variable="x0",
        )

    return unknown_experiment_intent()
