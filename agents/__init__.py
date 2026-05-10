"""Foundations for the batch distillation interface agent prototype."""

from .interface_agent import (
    analyze_message_features,
    classify_goal,
    classify_goal_deterministic,
    classify_goal_llm,
)
from .execution_schemas import WorkflowExecutionResult
from .schemas import GoalClassification
from .workflow_executor import execute_workflow
from .workflow_planner import create_workflow_plan
from .workflow_schemas import WorkflowPlan

__all__ = [
    "GoalClassification",
    "WorkflowExecutionResult",
    "WorkflowPlan",
    "analyze_message_features",
    "classify_goal",
    "classify_goal_deterministic",
    "classify_goal_llm",
    "execute_workflow",
    "create_workflow_plan",
]
