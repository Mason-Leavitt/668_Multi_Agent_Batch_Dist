from typing import Literal

from pydantic import BaseModel, Field

ProblemType = Literal[
    "solve_D_given_W0_x0_xDavg",
    "solve_batch_given_W0_x0_xB",
    "check_batch_consistency",
    "unknown",
]


class ProblemRequest(BaseModel):
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
    problem_type: ProblemType
    knowns: StructuredKnowns = Field(default_factory=StructuredKnowns)
    unknowns: list[str] = Field(default_factory=list)
    needs_clarification: bool
    clarification_question: str | None = None
