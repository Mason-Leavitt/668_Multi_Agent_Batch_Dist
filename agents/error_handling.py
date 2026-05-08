from typing import Any


VARIABLE_DISPLAY_NAMES = {
    "W0": "initial charge amount (W0)",
    "B": "final still amount (B)",
    "D": "distillate amount (D)",
    "x0": "initial ethanol mole fraction (x0)",
    "xB": "final still ethanol mole fraction (xB)",
    "xDavg": "average distillate ethanol mole fraction (xDavg)",
    "xDavg_target": "target average distillate ethanol mole fraction (xDavg_target)",
}


def format_variable_list(variable_names: list[str]) -> str:
    labels = [VARIABLE_DISPLAY_NAMES.get(name, name) for name in variable_names]
    if not labels:
        return ""
    if len(labels) == 1:
        return labels[0]
    if len(labels) == 2:
        return f"{labels[0]} and {labels[1]}"
    return ", ".join(labels[:-1]) + f", and {labels[-1]}"


def normalize_error_for_user(error: Exception | str, context: dict[str, Any] | None = None) -> str:
    context = context or {}
    error_text = str(error).strip() if error is not None else ""
    lower_text = error_text.lower()

    missing_inputs = context.get("missing_inputs") or []
    if missing_inputs:
        missing_text = format_variable_list(missing_inputs)
        return (
            f"I’m missing {missing_text}, so I can’t run that deterministic calculation yet. "
            "Please provide those values, or ask me to help choose a design basis first."
        )

    if context.get("problem_type") == "unsupported":
        return (
            "I don’t currently have a deterministic calculation tool for that request. "
            "I can help you choose one of the supported workflows instead."
        )

    if "openai_api_key is missing" in lower_text:
        return (
            "The LLM-based problem structurer is not available right now because the OpenAI API key is missing. "
            "Add `OPENAI_API_KEY` to `.env` and try again."
        )

    if "api" in lower_text or "rate limit" in lower_text or "authentication" in lower_text:
        return (
            "The LLM service is not available right now. Please try again in a moment, "
            "or check your OpenAI API access and quota."
        )

    if "unsupported problem_type" in lower_text:
        return (
            "I don’t currently have a deterministic calculation tool for that request. "
            "I can help you choose one of the supported workflows instead."
        )

    if "mole fraction" in lower_text and "between 0 and 1" in lower_text:
        for variable_name, label in VARIABLE_DISPLAY_NAMES.items():
            if variable_name.lower() in lower_text:
                return f"The {label} must be between 0 and 1."
        return "That mole fraction must be between 0 and 1."

    if "must be less than x0" in lower_text or ("xb" in lower_text and "less than x0" in lower_text):
        return (
            "The final still ethanol mole fraction (xB) must be less than the initial ethanol mole fraction (x0)."
        )

    if "desired average distillate composition should usually be greater than the initial still composition" in lower_text:
        return (
            "The target average distillate ethanol mole fraction (xDavg_target) should be greater than "
            "the initial ethanol mole fraction (x0) for this simple batch distillation model."
        )

    if "could not bracket a solution" in lower_text or "not physically reachable" in lower_text:
        return (
            "That requested case does not appear reachable with the current deterministic model and VLE data. "
            "Try a different target composition or ask me to help explore feasible options."
        )

    if "did not converge" in lower_text:
        return (
            "The deterministic solver could not converge for that request. "
            "Try adjusting the target values or ask me to help explore alternatives."
        )

    if "missing required inputs" in lower_text:
        return error_text

    return (
        "I ran into an internal issue while processing that request. "
        "Try rephrasing it, resetting the session, or providing the values more explicitly."
    )
