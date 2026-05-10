"""Deterministic workflow execution for implemented planner paths."""

from __future__ import annotations

import re

from engineering.calculations import find_D_xDavg_combinations_from_W0_x0
from engineering.conversions import (
    get_mixture_volume_L_from_moles,
    get_moles_and_etoh_frac_from_volume_L_and_abv,
)

from .execution_schemas import WorkflowExecutionResult
from .schemas import GoalClassification, ScalarValue
from .workflow_schemas import WorkflowPlan

GALLON_TO_LITER = 3.78541


def _extract_first_number(value: ScalarValue, field_name: str) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        match = re.search(r"[-+]?\d*\.?\d+", value)
        if match:
            return float(match.group(0))
    raise ValueError(f"Could not parse a numeric value for {field_name}.")


def _has_volume_units(value: ScalarValue) -> bool:
    if not isinstance(value, str):
        return False
    lowered = value.lower()
    return any(unit in lowered for unit in [" gallon", " gallons", " gal", " liter", " liters", " litre", " litres", " l", " ml"])


def _parse_volume_to_liters(value: ScalarValue, field_name: str) -> float:
    amount = _extract_first_number(value, field_name)
    if not isinstance(value, str):
        return amount

    lowered = value.lower()
    if "gallon" in lowered or re.search(r"\bgal\b", lowered):
        return amount * GALLON_TO_LITER
    if "ml" in lowered:
        return amount / 1000.0
    if any(unit in lowered for unit in ["liter", "liters", "litre", "litres"]) or re.search(r"\b l\b", lowered) or lowered.endswith("l"):
        return amount
    return amount


def _parse_abv_to_percent(value: ScalarValue, field_name: str) -> float:
    amount = _extract_first_number(value, field_name)
    if amount <= 1.0:
        return amount * 100.0
    return amount


def _parse_model_scalar(value: ScalarValue, field_name: str) -> float:
    return _extract_first_number(value, field_name)


def _w0_is_volume_like(classification: GoalClassification) -> bool:
    return _has_volume_units(classification.variable_assignments.get("W0"))


def _normalize_feed_inputs(classification: GoalClassification) -> tuple[float, float, list[str]]:
    warnings: list[str] = []
    assignments = classification.variable_assignments

    if "feed_volume" in assignments and "feed_abv" in assignments:
        volume_l = _parse_volume_to_liters(assignments["feed_volume"], "feed_volume")
        abv_percent = _parse_abv_to_percent(assignments["feed_abv"], "feed_abv")
        converted = get_moles_and_etoh_frac_from_volume_L_and_abv(
            volume_L=volume_l,
            abv_percent=abv_percent,
        )
        return converted["total_moles"], converted["x_etoh"], warnings

    if "W0" in assignments and "x0" in assignments and not _w0_is_volume_like(classification):
        return (
            _parse_model_scalar(assignments["W0"], "W0"),
            _parse_model_scalar(assignments["x0"], "x0"),
            warnings,
        )

    if "W0" in assignments and "feed_abv" in assignments and _w0_is_volume_like(classification):
        volume_l = _parse_volume_to_liters(assignments["W0"], "W0")
        abv_percent = _parse_abv_to_percent(assignments["feed_abv"], "feed_abv")
        converted = get_moles_and_etoh_frac_from_volume_L_and_abv(
            volume_L=volume_l,
            abv_percent=abv_percent,
        )
        warnings.append(
            "W0 was interpreted as a user-friendly feed volume and normalized to internal W0 and x0."
        )
        return converted["total_moles"], converted["x_etoh"], warnings

    raise ValueError(
        "Feed-to-product execution needs either W0 and x0 values, feed_volume and feed_abv values, or volume-like W0 plus feed_abv."
    )


def execute_feed_to_product_sweep(
    classification: GoalClassification,
    plan: WorkflowPlan,
) -> WorkflowExecutionResult:
    W0, x0, warnings = _normalize_feed_inputs(classification)
    raw_results = find_D_xDavg_combinations_from_W0_x0(W0=W0, x0=x0)

    rows: list[dict[str, ScalarValue]] = []
    include_user_friendly_output = classification.requires_output_conversion

    for result in raw_results:
        row: dict[str, ScalarValue] = {
            "xB": round(float(result["xB"]), 6),
            "D": round(float(result["D"]), 6),
            "xDavg": round(float(result["xDavg"]), 6),
        }
        if include_user_friendly_output:
            row["D_volume_L"] = round(
                float(get_mixture_volume_L_from_moles(float(result["D"]), float(result["xDavg"]))),
                6,
            )
        rows.append(row)

    columns = ["xB", "D", "xDavg"]
    if include_user_friendly_output:
        columns.append("D_volume_L")
        warnings.append(
            "User-friendly output conversion for xDavg_abv is planned but not implemented in this execution step."
        )

    message = (
        f"Executed feed_to_product_sweep using W0={W0:.6f} mol and x0={x0:.6f}. "
        f"Generated {len(rows)} sweep rows."
    )
    return WorkflowExecutionResult(
        workflow_name=plan.workflow_name,
        success=True,
        message=message,
        columns=columns,
        rows=rows,
        warnings=warnings,
    )


def execute_workflow(
    classification: GoalClassification,
    plan: WorkflowPlan,
) -> WorkflowExecutionResult:
    if not plan.ready_to_execute:
        return WorkflowExecutionResult(
            workflow_name=plan.workflow_name,
            success=False,
            message="Workflow cannot run yet because required inputs are missing.",
            warnings=[plan.suggested_next_message],
        )

    if plan.workflow_name != "feed_to_product_sweep":
        return WorkflowExecutionResult(
            workflow_name=plan.workflow_name,
            success=False,
            message="This workflow is planned but execution is not implemented yet.",
        )

    return execute_feed_to_product_sweep(classification, plan)
