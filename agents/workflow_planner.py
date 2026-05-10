"""Deterministic workflow planning from interface-agent classifications."""

from __future__ import annotations

from .schemas import GoalClassification
from .workflow_schemas import WorkflowPlan


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _has_all(items: list[str], required: list[str]) -> bool:
    item_set = set(items)
    return all(name in item_set for name in required)


def _w0_looks_user_friendly(classification: GoalClassification) -> bool:
    value = classification.variable_assignments.get("W0")
    if value is None:
        return False
    if isinstance(value, str):
        lowered = value.lower()
        return any(unit in lowered for unit in ["l", "liter", "litre", "gal", "gallon", "ml"])
    return classification.input_format in {"volume_abv", "mixed_units"}


def _base_plan(classification: GoalClassification) -> WorkflowPlan:
    return WorkflowPlan(
        workflow_name=classification.goal,
        ready_to_execute=False,
        required_inputs=[],
        available_inputs=list(classification.known_inputs),
        missing_inputs=list(classification.missing_inputs),
        normalization_steps=[],
        calculation_steps=[],
        result_steps=[],
        warnings=[],
        suggested_next_message="Review the detected inputs and confirm the next workflow step.",
    )


def _plan_feed_to_product_sweep(classification: GoalClassification) -> WorkflowPlan:
    plan = _base_plan(classification)
    conceptual_available = list(classification.known_inputs)
    input_format = classification.input_format

    if input_format == "model_units":
        plan.required_inputs = ["W0", "x0"]
    else:
        plan.required_inputs = ["feed_volume", "feed_abv"]

    if "W0" in classification.known_inputs and "feed_abv" in classification.known_inputs and input_format == "mixed_units":
        plan.warnings.append(
            "W0 was provided with user-friendly units; it will need normalization before internal calculations."
        )

    if _w0_looks_user_friendly(classification) and "feed_volume" not in conceptual_available:
        conceptual_available.append("feed_volume")
        if "W0" in classification.known_inputs and "feed_abv" in classification.known_inputs:
            plan.warnings.append(
                "W0 was provided with user-friendly units; it will need normalization before internal calculations."
            )

    if classification.requires_input_conversion:
        plan.normalization_steps.append(
            "Convert user-friendly feed volume and ABV to internal W0 and x0."
        )

    plan.calculation_steps.extend(
        [
            "Sweep xB across feasible stopping compositions.",
            "For each xB, calculate D and xDavg.",
        ]
    )

    if classification.sweep_variable is None and classification.output_mode in {"table", "mixed"}:
        plan.warnings.append(
            "No explicit sweep variable was provided; default planning assumption is to sweep xB."
        )

    if classification.requires_output_conversion:
        plan.result_steps.append(
            "Convert internal D and xDavg results to user-friendly product volume and ABV where possible."
        )
    plan.result_steps.append("Return requested D and xDavg combinations.")

    plan.available_inputs = _dedupe(conceptual_available)
    plan.missing_inputs = [
        name for name in plan.required_inputs if name not in set(plan.available_inputs)
    ]
    plan.ready_to_execute = (
        _has_all(plan.available_inputs, ["W0", "x0"])
        or _has_all(plan.available_inputs, ["feed_volume", "feed_abv"])
    )
    if plan.ready_to_execute:
        plan.suggested_next_message = (
            "This request is ready for a feed-to-product sweep once the deterministic engineering workflow is connected."
        )
    else:
        plan.suggested_next_message = (
            "Provide both the starting feed amount and the starting feed composition so the feed-to-product sweep can run."
        )
    plan.warnings = _dedupe(plan.warnings)
    return plan


def _plan_product_to_feed_sweep(classification: GoalClassification) -> WorkflowPlan:
    plan = _base_plan(classification)

    if classification.input_format == "model_units":
        plan.required_inputs = ["D", "xDavg"]
    else:
        plan.required_inputs = ["product_volume", "product_abv"]

    if classification.requires_input_conversion:
        plan.normalization_steps.append(
            "Convert user-friendly product volume and ABV to internal D and xDavg."
        )

    plan.calculation_steps.extend(
        [
            "Explore possible feed amount and feed composition combinations.",
            "Check candidate combinations against the Rayleigh and mole-balance relationships.",
        ]
    )

    if classification.requires_output_conversion:
        plan.result_steps.append(
            "Convert internal W0 and x0 results to user-friendly feed volume and ABV where possible."
        )
    plan.result_steps.append("Return possible feed requirement combinations.")

    plan.missing_inputs = [
        name for name in plan.required_inputs if name not in set(plan.available_inputs)
    ]
    plan.ready_to_execute = (
        _has_all(plan.available_inputs, ["D", "xDavg"])
        or _has_all(plan.available_inputs, ["product_volume", "product_abv"])
    )
    if plan.ready_to_execute:
        plan.suggested_next_message = (
            "This request is ready for a product-to-feed sweep once the deterministic engineering workflow is connected."
        )
    else:
        plan.suggested_next_message = (
            "Provide both the desired product amount and the desired product composition so the feed-requirement sweep can run."
        )
    return plan


def _plan_single_rayleigh(classification: GoalClassification) -> WorkflowPlan:
    plan = _base_plan(classification)
    plan.required_inputs = ["W0", "x0", "xB"]
    plan.missing_inputs = [
        name for name in plan.required_inputs if name not in set(plan.available_inputs)
    ]
    plan.ready_to_execute = not plan.missing_inputs
    plan.calculation_steps.extend(
        [
            "Run one Rayleigh calculation for the specified starting and stopping basis.",
            "Calculate requested outputs such as B, D, or xDavg.",
        ]
    )
    if plan.ready_to_execute:
        plan.suggested_next_message = (
            "This request is ready for a direct Rayleigh-style calculation once execution is connected."
        )
    else:
        plan.suggested_next_message = (
            "Provide W0, x0, and xB so the direct Rayleigh-style calculation can run."
        )
    return plan


def _plan_mole_balance(classification: GoalClassification) -> WorkflowPlan:
    plan = _base_plan(classification)
    plan.calculation_steps.append(
        "Use the overall mole balance relationship to solve the requested unknown."
    )
    plan.ready_to_execute = len(classification.known_inputs) >= 4 and len(classification.requested_outputs) >= 1
    plan.missing_inputs = list(classification.missing_inputs)
    if plan.ready_to_execute:
        plan.suggested_next_message = (
            "This request has enough known values to attempt a mole-balance solve once execution is connected."
        )
    else:
        plan.suggested_next_message = (
            "Provide at least four known variables and indicate which unknown should be solved from the mole balance."
        )
    return plan


def _plan_consistency_check(classification: GoalClassification) -> WorkflowPlan:
    plan = _base_plan(classification)
    plan.ready_to_execute = len(classification.known_inputs) >= 4
    plan.calculation_steps.extend(
        [
            "Check whether the provided values satisfy the material balance.",
            "Check whether values are inside valid physical ranges.",
        ]
    )
    if plan.ready_to_execute:
        plan.suggested_next_message = (
            "This request has enough values for a consistency check once the validation workflow is connected."
        )
    else:
        plan.suggested_next_message = (
            "Provide more batch values so the consistency check can test both balance closure and physical validity."
        )
    return plan


def _plan_explanation(classification: GoalClassification) -> WorkflowPlan:
    plan = _base_plan(classification)
    plan.ready_to_execute = True
    plan.result_steps.append("Generate an explanation of the requested variable or workflow.")
    plan.suggested_next_message = (
        "This request is ready for an explanation response without any engineering calculation."
    )
    return plan


def _plan_unsupported(classification: GoalClassification) -> WorkflowPlan:
    plan = _base_plan(classification)
    plan.ready_to_execute = False
    plan.suggested_next_message = (
        "Please restate the request with the known inputs, desired outputs, and whether you want a sweep, a direct calculation, a consistency check, or an explanation."
    )
    return plan


def create_workflow_plan(classification: GoalClassification) -> WorkflowPlan:
    """Create a deterministic workflow preflight plan from a GoalClassification."""

    planners = {
        "feed_to_product_sweep": _plan_feed_to_product_sweep,
        "product_to_feed_sweep": _plan_product_to_feed_sweep,
        "single_rayleigh_calculation": _plan_single_rayleigh,
        "mole_balance_calculation": _plan_mole_balance,
        "consistency_check": _plan_consistency_check,
        "explain_variable_or_workflow": _plan_explanation,
        "unsupported_or_unclear": _plan_unsupported,
    }
    planner = planners.get(classification.goal, _plan_unsupported)
    return planner(classification)
