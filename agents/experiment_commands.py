import re
from typing import Literal, TypedDict

from agents.session_commands import COMPOSITION_VARIABLES, normalize_variable_name


ExperimentCommandAction = Literal[
    "use_option",
    "try_value",
    "show_higher",
    "show_lower",
    "compare_variable",
    "clear_experiment",
    "none",
]


class ExperimentCommandResult(TypedDict):
    is_experiment_command: bool
    action: ExperimentCommandAction
    variable: str | None
    value: float | None
    option_index: int | None


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
        return {
            "is_experiment_command": True,
            "action": "clear_experiment",
            "variable": None,
            "value": None,
            "option_index": None,
        }

    option_match = re.fullmatch(
        r"(use|choose)\s+option\s+(\d+)",
        message,
        flags=re.IGNORECASE,
    )
    if option_match:
        return {
            "is_experiment_command": True,
            "action": "use_option",
            "variable": None,
            "value": None,
            "option_index": int(option_match.group(2)),
        }

    try_match = re.fullmatch(
        r"try\s+(.+?)(?:\s*=\s*|\s+)([-+]?\d*\.?\d+%?)",
        message,
        flags=re.IGNORECASE,
    )
    if try_match:
        variable = normalize_variable_name(try_match.group(1))
        value = _parse_numeric_value(variable, try_match.group(2))
        return {
            "is_experiment_command": variable is not None and value is not None,
            "action": "try_value" if variable is not None and value is not None else "none",
            "variable": variable,
            "value": value,
            "option_index": None,
        }

    higher_match = re.fullmatch(
        r"show\s+higher\s+(.+)",
        message,
        flags=re.IGNORECASE,
    )
    if higher_match:
        variable = normalize_variable_name(higher_match.group(1))
        return {
            "is_experiment_command": variable is not None,
            "action": "show_higher" if variable is not None else "none",
            "variable": variable,
            "value": None,
            "option_index": None,
        }

    lower_match = re.fullmatch(
        r"show\s+lower\s+(.+)",
        message,
        flags=re.IGNORECASE,
    )
    if lower_match:
        variable = normalize_variable_name(lower_match.group(1))
        return {
            "is_experiment_command": variable is not None,
            "action": "show_lower" if variable is not None else "none",
            "variable": variable,
            "value": None,
            "option_index": None,
        }

    compare_match = re.fullmatch(
        r"compare\s+(.+?)\s+instead",
        message,
        flags=re.IGNORECASE,
    )
    if compare_match:
        variable = normalize_variable_name(compare_match.group(1))
        return {
            "is_experiment_command": variable is not None,
            "action": "compare_variable" if variable is not None else "none",
            "variable": variable,
            "value": None,
            "option_index": None,
        }

    return {
        "is_experiment_command": False,
        "action": "none",
        "variable": None,
        "value": None,
        "option_index": None,
    }
