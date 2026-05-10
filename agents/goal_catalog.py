"""Supported goals and related metadata for the interface agent prototype."""

SUPPORTED_GOALS = [
    "feed_to_product_sweep",
    "product_to_feed_sweep",
    "single_rayleigh_calculation",
    "mole_balance_calculation",
    "consistency_check",
    "explain_variable_or_workflow",
    "unsupported_or_unclear",
]

SUPPORTED_OUTPUT_MODES = [
    "numeric_answer",
    "table",
    "plot",
    "explanation",
    "mixed",
    "unknown",
]

SUPPORTED_INPUT_FORMATS = [
    "model_units",
    "volume_abv",
    "mixed_units",
    "unknown",
]

SUPPORTED_OUTPUT_FORMATS = [
    "model_units",
    "volume_abv",
    "mixed_units",
    "user_friendly",
    "unknown",
]

GOAL_DESCRIPTIONS = {
    "feed_to_product_sweep": (
        "The user provides feed information and wants possible distillate or "
        "product outcomes."
    ),
    "product_to_feed_sweep": (
        "The user provides a desired distillate target and wants possible "
        "feed requirements."
    ),
    "single_rayleigh_calculation": (
        "The user wants one direct Rayleigh-style calculation for a specific "
        "starting and stopping basis."
    ),
    "mole_balance_calculation": (
        "The user wants to solve one variable from an overall mole balance "
        "when the other required values are known."
    ),
    "consistency_check": (
        "The user wants to test whether a proposed set of values is "
        "mathematically or physically consistent."
    ),
    "explain_variable_or_workflow": (
        "The user is asking for an explanation of a variable, equation, or "
        "workflow rather than a calculation."
    ),
    "unsupported_or_unclear": (
        "The request does not clearly fit the supported first-prototype goals."
    ),
}
