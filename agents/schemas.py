"""Typed schemas for the deterministic interface agent prototype."""

from typing import Literal

from pydantic import BaseModel, Field


GoalName = Literal[
    "feed_to_product_sweep",
    "product_to_feed_sweep",
    "single_rayleigh_calculation",
    "mole_balance_calculation",
    "consistency_check",
    "explain_variable_or_workflow",
    "unsupported_or_unclear",
]

OutputMode = Literal[
    "numeric_answer",
    "table",
    "plot",
    "explanation",
    "mixed",
    "unknown",
]

InputFormat = Literal[
    "model_units",
    "volume_abv",
    "mixed_units",
    "unknown",
]

OutputFormat = Literal[
    "model_units",
    "volume_abv",
    "mixed_units",
    "user_friendly",
    "unknown",
]


class GoalClassification(BaseModel):
    """Interface-agent interpretation of a user request.

    This schema captures how the first interface agent classifies the user's
    intent, known variables, expected outputs, and format/conversion needs.
    It is a request interpretation layer, not the final engineering result.
    """

    goal: GoalName
    output_mode: OutputMode = "unknown"

    known_inputs: list[str] = Field(default_factory=list)
    requested_outputs: list[str] = Field(default_factory=list)
    missing_inputs: list[str] = Field(default_factory=list)

    input_format: InputFormat = "unknown"
    output_format: OutputFormat = "unknown"

    requires_input_conversion: bool = False
    requires_output_conversion: bool = False

    sweep_variable: str | None = None

    confidence: float = Field(ge=0.0, le=1.0)

    reasoning_summary: str
    user_facing_summary: str
