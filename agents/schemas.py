from typing import Literal

from pydantic import BaseModel, Field

ProblemType = Literal[
    "solve_D_given_W0_x0_xDavg",
    "solve_batch_given_W0_x0_xB",
    "check_batch_consistency",
    "unknown",
]

IntentType = Literal[
    "calculation_request",
    "open_ended_guidance",
    "design_prototyping",
    "clarification_answer",
    "conceptual_question",
    "unknown",
]


class ProblemRequest(BaseModel):
    intent_type: IntentType
    problem_type: ProblemType
    knowns: dict[str, float] = Field(default_factory=dict)
    unknowns: list[str] = Field(default_factory=list)
    needs_clarification: bool
    clarification_question: str | None = None


class StructuredKnowns(BaseModel):
    W0: float | None = None
    x0: float | None = None
    xB: float | None = None
    xDavg_target: float | None = None
    xDavg: float | None = None
    B: float | None = None
    D: float | None = None


class LLMProblemRequest(BaseModel):
    intent_type: IntentType
    problem_type: ProblemType
    knowns: StructuredKnowns = Field(default_factory=StructuredKnowns)
    unknowns: list[str] = Field(default_factory=list)
    needs_clarification: bool
    clarification_question: str | None = None
