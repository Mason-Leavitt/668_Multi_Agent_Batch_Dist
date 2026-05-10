"""Schemas for deterministic workflow planning derived from classifier output."""

from pydantic import BaseModel, Field


class WorkflowPlan(BaseModel):
    """Deterministic preflight plan derived from GoalClassification.

    This plan describes which workflow should run, what inputs are needed,
    what normalization/calculation/result steps are expected, and whether the
    request is ready to execute. It is not a calculation result.
    """

    workflow_name: str
    ready_to_execute: bool

    required_inputs: list[str] = Field(default_factory=list)
    available_inputs: list[str] = Field(default_factory=list)
    missing_inputs: list[str] = Field(default_factory=list)

    normalization_steps: list[str] = Field(default_factory=list)
    calculation_steps: list[str] = Field(default_factory=list)
    result_steps: list[str] = Field(default_factory=list)

    warnings: list[str] = Field(default_factory=list)

    suggested_next_message: str
