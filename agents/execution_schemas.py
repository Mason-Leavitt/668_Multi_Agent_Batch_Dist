"""Schemas for deterministic workflow execution results."""

from pydantic import BaseModel, Field

from .schemas import ScalarValue


class WorkflowExecutionResult(BaseModel):
    """Structured output from deterministic workflow execution.

    This model captures the result of running a deterministic workflow step.
    It is not an LLM result.
    """

    workflow_name: str
    success: bool
    message: str

    columns: list[str] = Field(default_factory=list)
    rows: list[dict[str, ScalarValue]] = Field(default_factory=list)

    warnings: list[str] = Field(default_factory=list)
