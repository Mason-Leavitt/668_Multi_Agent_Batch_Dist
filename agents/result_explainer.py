"""LLM explanation of deterministic workflow execution results."""

from __future__ import annotations

import json
from pathlib import Path

from .execution_schemas import WorkflowExecutionResult
from .schemas import GoalClassification
from .workflow_schemas import WorkflowPlan


def _build_sample_rows(rows: list[dict], limit: int = 5) -> list[dict]:
    if len(rows) <= limit * 2:
        return rows
    return rows[:limit] + rows[-limit:]


def _load_background_text() -> str:
    background_path = Path(__file__).resolve().parents[1] / "docs" / "agent_background.md"
    try:
        return background_path.read_text(encoding="utf-8")
    except OSError:
        return ""


def explain_execution_result(
    user_message: str,
    classification: GoalClassification,
    plan: WorkflowPlan,
    execution_result: WorkflowExecutionResult,
    model_name: str = "gpt-4o-mini",
) -> str:
    """Generate a concise explanation of deterministic workflow results."""

    from dotenv import load_dotenv
    from langchain_openai import ChatOpenAI

    load_dotenv()
    llm = ChatOpenAI(model=model_name, temperature=0)
    background = _load_background_text()

    system_prompt = (
        "You are a batch distillation results explainer. "
        "The numeric results come from deterministic engineering functions. "
        "Do not do new calculations or invent values. "
        "Use only the provided execution result. "
        "If a number is not present, say it is not shown. "
        "Explain units clearly. "
        "Explain that internal calculations use moles and mole fractions. "
        "Explain user-friendly columns such as liters and ABV when present. "
        "Terminology rules: "
        "D is total distillate/product amount in moles of the ethanol-water mixture, not moles of ethanol. "
        "D_volume_L is total distillate/product volume in liters of the ethanol-water mixture, not liters of ethanol. "
        "W0 is total feed/still-charge amount in moles of the ethanol-water mixture. "
        "W0_volume_L is total feed/still-charge volume in liters of the ethanol-water mixture. "
        "x0 is ethanol mole fraction in the feed and x0_abv_percent is estimated feed ABV percent. "
        "xB is ethanol mole fraction in the remaining still bottoms and xB_abv_percent is estimated bottoms ABV percent. "
        "xDavg is average ethanol mole fraction in the collected distillate mixture and xDavg_abv_percent is estimated ABV percent of that mixture. "
        "Never describe D, D_volume_L, W0, or W0_volume_L as pure ethanol amounts by themselves. "
        "If the execution result includes both D and D_volume_L, explain that they represent the total product mixture amount in different units. "
        "For feed_to_product_sweep, explain that the rows are possible product mixture outcomes for different stopping compositions. "
        "For product_to_feed_sweep results, explain that the table shows possible starting feed conditions "
        "that could produce the desired product target. "
        "For solve_mole_balance results, explain that the workflow solves algebraic batch-balance variables "
        "using W0 = D + B and W0*x0 = D*xDavg + B*xB. "
        "Explain that it does not guarantee the values satisfy Rayleigh or VLE batch-distillation behavior. "
        "Explain that amount columns are total ethanol-water mixture amounts and composition columns are ethanol composition values. "
        "For solve_rayleigh_batch_variables results, explain that the workflow solves one direct Rayleigh-constrained batch case "
        "using the total balance, ethanol balance, and the Rayleigh equation. "
        "Explain that W0, D, and B are total mixture amounts while x0, xDavg, and xB are ethanol composition values. "
        "User-friendly columns convert total mixture amounts to liters and compositions to ABV. "
        "W0 is internal feed amount in moles, W0_volume_L is estimated feed volume in liters, "
        "x0 is initial ethanol mole fraction, x0_abv_percent is estimated feed strength in ABV percent, "
        "and xB or xB_abv_percent represents the final still composition when shown. "
        "Mention warnings if present. "
        "Keep the explanation concise and conversational."
    )

    payload = {
        "background": background,
        "user_request": user_message,
        "classification_summary": {
            "goal": classification.goal,
            "output_mode": classification.output_mode,
        },
        "workflow_plan": {
            "workflow_name": plan.workflow_name,
            "normalization_steps": plan.normalization_steps,
            "calculation_steps": plan.calculation_steps,
        },
        "execution_result": {
            "success": execution_result.success,
            "message": execution_result.message,
            "normalized_inputs": execution_result.normalized_inputs,
            "execution_parameters": execution_result.execution_parameters,
            "warnings": execution_result.warnings,
            "columns": execution_result.columns,
            "row_count": len(execution_result.rows),
            "sample_rows": _build_sample_rows(execution_result.rows),
        },
    }

    response = llm.invoke(
        [
            ("system", system_prompt),
            (
                "human",
                "User request and deterministic result context:\n"
                f"{json.dumps(payload, indent=2)}\n\n"
                "Write a concise conversational explanation of the results.",
            ),
        ]
    )
    return response.content if isinstance(response.content, str) else str(response.content)
