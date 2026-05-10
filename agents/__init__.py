"""Foundations for the batch distillation interface agent prototype."""

from .interface_agent import classify_goal, classify_goal_deterministic, classify_goal_llm
from .schemas import GoalClassification

__all__ = [
    "GoalClassification",
    "classify_goal",
    "classify_goal_deterministic",
    "classify_goal_llm",
]
