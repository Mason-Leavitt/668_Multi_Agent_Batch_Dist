"""Foundations for the batch distillation interface agent prototype."""

from .interface_agent import classify_goal_deterministic
from .schemas import GoalClassification

__all__ = ["GoalClassification", "classify_goal_deterministic"]
