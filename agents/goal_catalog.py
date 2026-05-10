"""Supported goals and related metadata for the interface agent prototype."""

from __future__ import annotations

from app.ui_metadata import get_workflow_reference

# Executable workflow metadata is registry-backed. Conversational classifier-only
# categories remain local here for compatibility with existing callers.
_EXECUTABLE_WORKFLOW_DESCRIPTIONS = {
    str(workflow["workflow_id"]): str(workflow["description"])
    for workflow in get_workflow_reference()
}

_CONVERSATIONAL_GOAL_DESCRIPTIONS = {
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

SUPPORTED_GOALS = list(_EXECUTABLE_WORKFLOW_DESCRIPTIONS) + list(
    _CONVERSATIONAL_GOAL_DESCRIPTIONS
)

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
    **_EXECUTABLE_WORKFLOW_DESCRIPTIONS,
    **_CONVERSATIONAL_GOAL_DESCRIPTIONS,
}
