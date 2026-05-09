from typing import Any, TypedDict

from agents.schemas import IntentType, ProblemType


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
    active_experiment: dict[str, Any] | None
    experiment_results: list[dict[str, Any]] | None
    experiment_sampled_variable: str | None
    experiment_knowns: dict[str, float] | None
    experiment_status: str | None

    intent_type: IntentType
    problem_type: ProblemType
    knowns: dict[str, float]
    unknowns: list[str]
    needs_clarification: bool
    clarification_question: str | None
    guidance_response: str

    calculation_success: bool
    result: dict[str, Any]
    consistency_check: dict[str, Any]
    errors: list[str]
    warnings: list[str]

    final_answer: str
