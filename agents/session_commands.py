import re
from typing import Literal, TypedDict


SessionCommandAction = Literal["reset", "forget", "set", "none"]


class SessionCommandResult(TypedDict):
    is_session_command: bool
    action: SessionCommandAction
    variable: str | None
    value: float | None
    message: str


class SessionStateUpdate(TypedDict):
    prior_knowns: dict[str, float]
    prior_needs_clarification: bool
    prior_clarification_question: str | None
    active_experiment: dict | None
    experiment_results: list[dict] | None
    experiment_sampled_variable: str | None
    experiment_knowns: dict[str, float] | None
    experiment_status: str | None
    pending_commit_variable: str | None
    pending_commit_value: float | None
    pending_commit_source: dict | None
    message: str


VARIABLE_ALIASES = {
    "w0": "W0",
    "b": "B",
    "d": "D",
    "x0": "x0",
    "xb": "xB",
    "xdavg": "xDavg",
    "xdavgtarget": "xDavg_target",
    "xdavg_target": "xDavg_target",
    "x_davg_target": "xDavg_target",
    "initial charge": "W0",
    "initial amount": "W0",
    "starting amount": "W0",
    "feed amount": "W0",
    "feed composition": "x0",
    "initial composition": "x0",
    "initial ethanol mole fraction": "x0",
    "final still composition": "xB",
    "final pot composition": "xB",
    "bottoms composition": "xB",
    "average distillate composition": "xDavg_target",
    "target average distillate composition": "xDavg_target",
}

COMPOSITION_VARIABLES = {"x0", "xB", "xDavg", "xDavg_target"}
VARIABLE_LABELS = {
    "W0": "initial charge (W0)",
    "B": "still amount (B)",
    "D": "distillate amount (D)",
    "x0": "initial ethanol mole fraction (x0)",
    "xB": "final still composition (xB)",
    "xDavg": "average distillate composition (xDavg)",
    "xDavg_target": "target average distillate composition (xDavg_target)",
}


def normalize_variable_name(raw_name: str) -> str | None:
    normalized = " ".join(raw_name.strip().lower().replace("-", " ").split())
    if not normalized:
        return None

    if normalized in VARIABLE_ALIASES:
        return VARIABLE_ALIASES[normalized]

    normalized_no_spaces = normalized.replace(" ", "")
    return VARIABLE_ALIASES.get(normalized_no_spaces)


def _parse_value_for_variable(variable: str, raw_value: str) -> tuple[float | None, str | None]:
    value_text = raw_value.strip().lower()
    if not value_text:
        return None, "Please provide a numeric value."

    if value_text.endswith("%"):
        if variable not in COMPOSITION_VARIABLES:
            return None, (
                f"`{variable}` is not a composition variable. Please provide its value without a percent sign."
            )
        percent_text = value_text[:-1].strip()
        try:
            return float(percent_text) / 100.0, None
        except ValueError:
            return None, "I could not parse that percentage. Please provide a numeric mole fraction like 0.05."

    try:
        return float(value_text), None
    except ValueError:
        return None, "I could not parse that value. Please provide a numeric value like 0.08 or 1500."


def parse_session_command(user_message: str) -> SessionCommandResult:
    message = user_message.strip()
    lower_message = message.lower()

    if lower_message in {"start over", "reset"}:
        return {
            "is_session_command": True,
            "action": "reset",
            "variable": None,
            "value": None,
            "message": "Session reset. I cleared the remembered values and clarification state.",
        }

    forget_match = re.fullmatch(r"forget\s+(.+)", message, flags=re.IGNORECASE)
    if forget_match:
        raw_variable = forget_match.group(1).strip()
        variable = normalize_variable_name(raw_variable)
        if variable is None:
            return {
                "is_session_command": True,
                "action": "forget",
                "variable": None,
                "value": None,
                "message": f"I did not recognize the variable '{raw_variable}'.",
            }
        return {
            "is_session_command": True,
            "action": "forget",
            "variable": variable,
            "value": None,
            "message": "",
        }

    set_match = re.fullmatch(r"(set|change)\s+(.+?)\s+to\s+(.+)", message, flags=re.IGNORECASE)
    if set_match:
        raw_variable = set_match.group(2).strip()
        raw_value = set_match.group(3).strip()
        variable = normalize_variable_name(raw_variable)
        if variable is None:
            return {
                "is_session_command": True,
                "action": "set",
                "variable": None,
                "value": None,
                "message": f"I did not recognize the variable '{raw_variable}'.",
            }

        value, error_message = _parse_value_for_variable(variable, raw_value)
        if error_message is not None:
            return {
                "is_session_command": True,
                "action": "set",
                "variable": variable,
                "value": None,
                "message": error_message,
            }

        return {
            "is_session_command": True,
            "action": "set",
            "variable": variable,
            "value": value,
            "message": "",
        }

    return {
        "is_session_command": False,
        "action": "none",
        "variable": None,
        "value": None,
        "message": "",
    }


def apply_session_command(
    command: SessionCommandResult,
    prior_knowns: dict[str, float],
    prior_needs_clarification: bool,
    prior_clarification_question: str | None,
    active_experiment: dict | None = None,
    experiment_results: list[dict] | None = None,
    experiment_sampled_variable: str | None = None,
    experiment_knowns: dict[str, float] | None = None,
    experiment_status: str | None = None,
    pending_commit_variable: str | None = None,
    pending_commit_value: float | None = None,
    pending_commit_source: dict | None = None,
) -> SessionStateUpdate:
    updated_knowns = dict(prior_knowns)

    def current_state(message: str) -> SessionStateUpdate:
        return {
            "prior_knowns": updated_knowns,
            "prior_needs_clarification": prior_needs_clarification,
            "prior_clarification_question": prior_clarification_question,
            "active_experiment": active_experiment,
            "experiment_results": experiment_results,
            "experiment_sampled_variable": experiment_sampled_variable,
            "experiment_knowns": experiment_knowns,
            "experiment_status": experiment_status,
            "pending_commit_variable": pending_commit_variable,
            "pending_commit_value": pending_commit_value,
            "pending_commit_source": pending_commit_source,
            "message": message,
        }

    if command["action"] == "reset":
        # Text-based reset clears remembered state; each interface can decide
        # separately whether visible chat history should also be cleared.
        return {
            "prior_knowns": {},
            "prior_needs_clarification": False,
            "prior_clarification_question": None,
            "active_experiment": None,
            "experiment_results": None,
            "experiment_sampled_variable": None,
            "experiment_knowns": None,
            "experiment_status": None,
            "pending_commit_variable": None,
            "pending_commit_value": None,
            "pending_commit_source": None,
            "message": command["message"],
        }

    if command["action"] == "forget":
        variable = command["variable"]
        if variable is None:
            return current_state(command["message"])

        if variable in updated_knowns:
            updated_knowns.pop(variable)
            return {
                "prior_knowns": updated_knowns,
                "prior_needs_clarification": False,
                "prior_clarification_question": None,
                "active_experiment": None,
                "experiment_results": None,
                "experiment_sampled_variable": None,
                "experiment_knowns": None,
                "experiment_status": None,
                "pending_commit_variable": None,
                "pending_commit_value": None,
                "pending_commit_source": None,
                "message": (
                    f"Forgot {VARIABLE_LABELS.get(variable, variable)} from the current session. "
                    "What would you like to do next?"
                ),
            }

        return current_state(f"`{variable}` was not currently remembered.")

    if command["action"] == "set":
        variable = command["variable"]
        value = command["value"]
        if variable is None or value is None:
            return current_state(command["message"])

        updated_knowns[variable] = value
        return {
            "prior_knowns": updated_knowns,
            "prior_needs_clarification": False,
            "prior_clarification_question": None,
            "active_experiment": None,
            "experiment_results": None,
            "experiment_sampled_variable": None,
            "experiment_knowns": None,
            "experiment_status": None,
            "pending_commit_variable": None,
            "pending_commit_value": None,
            "pending_commit_source": None,
            "message": (
                f"Updated {VARIABLE_LABELS.get(variable, variable)} to {value:g}. "
                "What would you like to calculate or explore next?"
            ),
        }

    return current_state("")
