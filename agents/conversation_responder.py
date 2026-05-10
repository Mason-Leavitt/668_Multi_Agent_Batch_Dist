"""LLM-backed conversational responses that do not trigger new workflow execution."""

from __future__ import annotations

import json
from pathlib import Path

from .execution_schemas import WorkflowExecutionResult
from .schemas import GoalClassification
from .workflow_schemas import WorkflowPlan


def _load_background_text() -> str:
    background_path = Path(__file__).resolve().parents[1] / "docs" / "agent_background.md"
    try:
        return background_path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _sample_rows(rows: list[dict], limit: int = 3) -> list[dict]:
    if len(rows) <= limit * 2:
        return rows
    return rows[:limit] + rows[-limit:]


def respond_to_general_message(
    user_message: str,
    last_user_request: str | None = None,
    last_classification: GoalClassification | None = None,
    last_plan: WorkflowPlan | None = None,
    last_execution_result: WorkflowExecutionResult | None = None,
    model_name: str = "gpt-4o-mini",
    api_key: str | None = None,
) -> str:
    """Respond conversationally without treating the message as a new workflow request."""

    from dotenv import load_dotenv
    from langchain_openai import ChatOpenAI

    load_dotenv()
    llm = ChatOpenAI(model=model_name, temperature=0, api_key=api_key)
    background = _load_background_text()

    system_prompt = (
        "You are the conversational interface for a batch distillation assistant. "
        "You can discuss the app, the latest result, terminology, and general batch distillation concepts. "
        "The deterministic engineering functions are the source of numerical truth. "
        "Do not perform new calculations or invent values. "
        "You can run deterministic engineering workflows once the goal and required inputs are clear. "
        "If the user asks what the app can do, describe the currently supported workflows and explain that calculations run through deterministic engineering functions after planning and confirmation. "
        "This general conversation path is for corrections, conceptual questions, capability questions, interpretation questions, and non-actionable discussion. "
        "If the user asks for a new calculation or design task, briefly redirect them to describe the desired task clearly instead of trying to solve it here. "
        "Do not tell users they must provide W0, x0, D, or xDavg when user-facing volume and ABV inputs are already sufficient for a supported sweep."
    )

    compact_result = None
    if last_execution_result is not None:
        compact_result = {
            "success": last_execution_result.success,
            "message": last_execution_result.message,
            "workflow_name": last_execution_result.workflow_name,
            "normalized_inputs": last_execution_result.normalized_inputs,
            "execution_parameters": last_execution_result.execution_parameters,
            "warnings": last_execution_result.warnings,
            "columns": last_execution_result.columns,
            "row_count": len(last_execution_result.rows),
            "sample_rows": _sample_rows(last_execution_result.rows),
        }

    payload = {
        "background": background,
        "user_message": user_message,
        "last_user_request": last_user_request,
        "last_classification": last_classification.model_dump() if last_classification else None,
        "last_plan": last_plan.model_dump() if last_plan else None,
        "last_execution_result_summary": compact_result,
    }

    response = llm.invoke(
        [
            ("system", system_prompt),
            (
                "human",
                "Conversation context:\n"
                f"{json.dumps(payload, indent=2)}\n\n"
                "Respond conversationally. If the user is correcting terminology, acknowledge and clarify it. "
                "If they are asking what the app can do, describe supported workflows and limitations. "
                "Do not invent numbers. Do not lecture the user about moles or mole fractions when volume and ABV are sufficient user-facing inputs. "
                "If this function receives an actionable workflow-like message by mistake, respond briefly and invite the user to confirm the intended workflow instead of redirecting them into internal units.",
            ),
        ]
    )
    return response.content if isinstance(response.content, str) else str(response.content)
