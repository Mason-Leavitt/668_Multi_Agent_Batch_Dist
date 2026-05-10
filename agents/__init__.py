"""Foundations for the batch distillation interface agent prototype."""

from .interface_agent import (
    analyze_message_features,
    classify_goal,
    classify_goal_deterministic,
    classify_goal_llm,
)
from .router_schemas import ConversationRoute
from .execution_schemas import WorkflowExecutionResult
from .schemas import GoalClassification
from .conversation_responder import respond_to_general_message
from .conversation_router import (
    extract_volume_abv_pair,
    infer_likely_goal,
    infer_target_role,
    route_conversation_message,
    safe_confirmation_fallback,
)
from .result_explainer import explain_execution_result
from .workflow_executor import execute_workflow
from .workflow_planner import create_workflow_plan
from .workflow_schemas import WorkflowPlan

__all__ = [
    "GoalClassification",
    "ConversationRoute",
    "WorkflowExecutionResult",
    "WorkflowPlan",
    "analyze_message_features",
    "classify_goal",
    "classify_goal_deterministic",
    "classify_goal_llm",
    "respond_to_general_message",
    "extract_volume_abv_pair",
    "infer_likely_goal",
    "infer_target_role",
    "route_conversation_message",
    "safe_confirmation_fallback",
    "explain_execution_result",
    "execute_workflow",
    "create_workflow_plan",
]
