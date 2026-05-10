"""Structured routing schema for conversational message handling."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


RouteName = Literal[
    "new_workflow_request",
    "workflow_clarification",
    "confirm_pending_workflow",
    "cancel_pending_workflow",
    "result_question",
    "result_explanation_request",
    "general_conversation",
]

LikelyGoal = Literal[
    "feed_to_product_sweep",
    "product_to_feed_sweep",
    "solve_rayleigh_batch_variables",
    "solve_mole_balance",
    "consistency_check",
    "explain_variable_or_workflow",
    "unsupported_or_unclear",
    "unknown",
]


class ConversationRoute(BaseModel):
    """Route the user's conversational message.

    This schema routes the user's conversational message. It does not execute
    calculations and does not replace GoalClassification.
    """

    route: RouteName
    likely_goal: LikelyGoal = "unknown"
    confidence: float = Field(ge=0.0, le=1.0)
    should_call_goal_classifier: bool = False
    uses_previous_context: bool = False
    needs_confirmation: bool = False
    reasoning_summary: str
    user_facing_summary: str
