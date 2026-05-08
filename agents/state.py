from typing import Any, TypedDict

from agents.schemas import ProblemType


class BatchDistillationState(TypedDict, total=False):
    """
    Shared state passed through the LangGraph workflow.

    Each node receives this state and returns a partial update.
    """

    user_message: str
    user_goal: str
    prior_knowns: dict[str, float]
    prior_needs_clarification: bool
    prior_clarification_question: str | None

    problem_type: ProblemType
    knowns: dict[str, float]
    unknowns: list[str]
    needs_clarification: bool
    clarification_question: str | None

    calculation_success: bool
    result: dict[str, Any]
    consistency_check: dict[str, Any]
    errors: list[str]
    warnings: list[str]

    final_answer: str
