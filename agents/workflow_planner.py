"""Deterministic workflow planning from interface-agent classifications."""

from __future__ import annotations

from .intent_analysis import analyze_batch_capabilities
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


def _has_value(classification: GoalClassification, key: str) -> bool:
    if key not in classification.variable_assignments:
        return False
    value = classification.variable_assignments[key]
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip() != ""
    return True


def _has_any_value(classification: GoalClassification, keys: list[str]) -> bool:
    return any(_has_value(classification, key) for key in keys)


def _assigned_keys(classification: GoalClassification) -> list[str]:
    return [
        key
        for key in classification.variable_assignments
        if _has_value(classification, key)
    ]


def _normalized_balance_knowns(classification: GoalClassification) -> list[str]:
    knowns: list[str] = []

    for key in ["W0", "x0", "D", "xDavg", "B", "xB"]:
        if _has_value(classification, key):
            knowns.append(key)

    if _has_value(classification, "feed_volume") and _has_value(classification, "feed_abv"):
        knowns.extend(["W0", "x0"])
    if _has_value(classification, "product_volume") and _has_value(classification, "product_abv"):
        knowns.extend(["D", "xDavg"])
    if _has_value(classification, "bottoms_volume") and _has_value(classification, "bottoms_abv"):
        knowns.extend(["B", "xB"])

    return _dedupe(knowns)


def _normalized_rayleigh_knowns(classification: GoalClassification) -> list[str]:
    knowns: list[str] = []

    for key in ["W0", "x0", "D", "xDavg", "B", "xB"]:
        if _has_value(classification, key):
            knowns.append(key)

    if _has_value(classification, "feed_volume") and _has_value(classification, "feed_abv"):
        knowns.extend(["W0", "x0"])
    elif _has_value(classification, "feed_abv"):
        knowns.append("x0")

    if _has_value(classification, "product_volume") and _has_value(classification, "product_abv"):
        knowns.extend(["D", "xDavg"])
    elif _has_value(classification, "product_abv"):
        knowns.append("xDavg")

    if _has_value(classification, "bottoms_volume") and _has_value(classification, "bottoms_abv"):
        knowns.extend(["B", "xB"])
    elif _has_value(classification, "bottoms_abv"):
        knowns.append("xB")

    return _dedupe(knowns)


def _w0_looks_user_friendly(classification: GoalClassification) -> bool:
    value = classification.variable_assignments.get("W0")
    if value is None:
        return False
    if isinstance(value, str):
        lowered = value.lower()
        return any(unit in lowered for unit in ["l", "liter", "litre", "gal", "gallon", "ml"])
    return False


def _base_plan(classification: GoalClassification) -> WorkflowPlan:
    return WorkflowPlan(
        workflow_name=classification.goal,
        ready_to_execute=False,
        required_inputs=[],
        available_inputs=_assigned_keys(classification),
        missing_inputs=[],
        normalization_steps=[],
        calculation_steps=[],
        result_steps=[],
        warnings=[],
        suggested_next_message="Review the detected inputs and confirm the next workflow step.",
    )


def _plan_feed_to_product_sweep(classification: GoalClassification) -> WorkflowPlan:
    plan = _base_plan(classification)
    assigned_inputs = _assigned_keys(classification)
    input_format = classification.input_format

    if input_format == "model_units":
        plan.required_inputs = ["W0", "x0"]
    else:
        plan.required_inputs = ["feed_volume", "feed_abv"]

    if "W0" in classification.known_inputs and "feed_abv" in classification.known_inputs and input_format == "mixed_units":
        plan.warnings.append(
            "W0 was provided with user-friendly units; it will need normalization before internal calculations."
        )

    has_model_values = _has_value(classification, "W0") and _has_value(classification, "x0")
    has_user_friendly_values = _has_value(classification, "feed_volume") and _has_value(classification, "feed_abv")
    has_mixed_volume_values = _w0_looks_user_friendly(classification) and _has_value(classification, "feed_abv")

    if has_mixed_volume_values:
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

    plan.available_inputs = _dedupe(assigned_inputs)
    if has_model_values or has_user_friendly_values or has_mixed_volume_values:
        plan.missing_inputs = []
    else:
        plan.missing_inputs = [name for name in plan.required_inputs if not _has_value(classification, name)]
        if input_format in {"volume_abv", "mixed_units"} and not has_user_friendly_values:
            plan.missing_inputs = []
            if not (_has_value(classification, "feed_volume") or _w0_looks_user_friendly(classification)):
                plan.missing_inputs.append("feed_volume")
            if not _has_value(classification, "feed_abv"):
                plan.missing_inputs.append("feed_abv")
        if input_format == "model_units" and not has_model_values:
            plan.missing_inputs = []
            if not _has_value(classification, "W0"):
                plan.missing_inputs.append("W0")
            if not _has_value(classification, "x0"):
                plan.missing_inputs.append("x0")

    plan.ready_to_execute = has_model_values or has_user_friendly_values or has_mixed_volume_values
    if plan.ready_to_execute:
        plan.suggested_next_message = (
            "This request is ready for a feed-to-product sweep once the deterministic engineering workflow is connected."
        )
    else:
        plan.suggested_next_message = (
            f"I can do this feed-to-product sweep, but I need values for {', '.join(plan.missing_inputs)} first."
        )
    plan.warnings = _dedupe(plan.warnings)
    return plan


def _plan_product_to_feed_sweep(classification: GoalClassification) -> WorkflowPlan:
    plan = _base_plan(classification)
    assigned_inputs = _assigned_keys(classification)

    if classification.input_format == "model_units":
        plan.required_inputs = ["D", "xDavg"]
    else:
        plan.required_inputs = ["product_volume", "product_abv"]

    has_model_values = _has_value(classification, "D") and _has_value(classification, "xDavg")
    has_user_friendly_values = _has_value(classification, "product_volume") and _has_value(classification, "product_abv")

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

    plan.available_inputs = _dedupe(assigned_inputs)
    if has_model_values or has_user_friendly_values:
        plan.missing_inputs = []
    else:
        if classification.input_format == "model_units":
            plan.missing_inputs = []
            if not _has_value(classification, "D"):
                plan.missing_inputs.append("D")
            if not _has_value(classification, "xDavg"):
                plan.missing_inputs.append("xDavg")
        else:
            plan.missing_inputs = []
            if not _has_value(classification, "product_volume"):
                plan.missing_inputs.append("product_volume")
            if not _has_value(classification, "product_abv"):
                plan.missing_inputs.append("product_abv")

    plan.ready_to_execute = has_model_values or has_user_friendly_values
    if plan.ready_to_execute:
        plan.suggested_next_message = (
            "This request is ready for a product-to-feed sweep once the deterministic engineering workflow is connected."
        )
    else:
        plan.suggested_next_message = (
            "I can explore feed requirements, but I need the target product amount and product strength first."
        )
    return plan


def _plan_solve_rayleigh_batch_variables(classification: GoalClassification) -> WorkflowPlan:
    plan = _base_plan(classification)
    normalized_knowns = _normalized_rayleigh_knowns(classification)
    known_set = set(normalized_knowns)
    capability = analyze_batch_capabilities(
        known_quantities=classification.variable_assignments,
        requested_outputs=classification.requested_outputs,
        candidate_goals=[classification.goal],
    )
    plan.available_inputs = _dedupe(_assigned_keys(classification))
    plan.normalization_steps.append(
        "Convert user-facing volume and ABV inputs to internal moles and mole fractions where needed."
    )
    plan.calculation_steps.extend(
        [
            "Apply the Rayleigh equation to relate W0, B, x0, and xB.",
            "Apply total balance W0 = D + B.",
            "Apply ethanol balance W0*x0 = D*xDavg + B*xB.",
            "Solve the missing batch variables.",
        ]
    )
    plan.result_steps.extend(
        [
            "Return internal model variables.",
            "Convert total mixture amounts to liters and compositions to ABV where possible.",
        ]
    )
    supported_case_sets = [
        {"W0", "x0", "xB"},
        {"W0", "x0", "D"},
        {"D", "xDavg", "x0"},
        {"D", "xDavg", "xB"},
    ]
    plan.ready_to_execute = any(case.issubset(known_set) for case in supported_case_sets) or capability.can_calculate_now
    if plan.ready_to_execute:
        plan.missing_inputs = []
    else:
        if capability.blocking_missing_values:
            plan.missing_inputs = list(capability.blocking_missing_values)
        else:
            candidate_missing_lists = [
                [name for name in case if name not in known_set]
                for case in supported_case_sets
            ]
            candidate_missing_lists.sort(key=lambda items: (len(items), items))
            plan.missing_inputs = candidate_missing_lists[0] if candidate_missing_lists else list(classification.missing_inputs)
    if plan.ready_to_execute:
        plan.suggested_next_message = (
            "This request is ready for a direct Rayleigh-constrained solve."
        )
    else:
        plan.suggested_next_message = (
            f"Provide the missing values for one supported Rayleigh solve case: {', '.join(plan.missing_inputs)}."
        )
    return plan


def _plan_solve_mole_balance(classification: GoalClassification) -> WorkflowPlan:
    plan = _base_plan(classification)
    normalized_knowns = _normalized_balance_knowns(classification)
    capability = analyze_batch_capabilities(
        known_quantities=classification.variable_assignments,
        requested_outputs=classification.requested_outputs,
        candidate_goals=[classification.goal],
    )
    plan.available_inputs = _dedupe(_assigned_keys(classification))
    plan.normalization_steps.append(
        "Convert user-facing volume and ABV inputs to internal moles and mole fractions where needed."
    )
    plan.calculation_steps.extend(
        [
            "Apply total balance W0 = D + B.",
            "Apply ethanol balance W0*x0 = D*xDavg + B*xB.",
            "Solve requested unknown variable(s).",
        ]
    )
    plan.result_steps.extend(
        [
            "Return internal model variables.",
            "Convert amounts to liters and compositions to ABV when possible.",
        ]
    )
    supported_case_sets = [
        {"W0", "x0", "D", "xDavg"},
        {"W0", "x0", "B", "xB"},
        {"D", "xDavg", "B", "xB"},
        {"W0", "D", "xDavg", "xB"},
        {"W0", "x0", "D", "xB"},
    ]
    plan.ready_to_execute = any(case.issubset(set(normalized_knowns)) for case in supported_case_sets) or capability.can_calculate_now
    plan.missing_inputs = list(capability.blocking_missing_values or classification.missing_inputs)
    if plan.ready_to_execute:
        plan.suggested_next_message = (
            "This request has enough known values for a mole-balance solve."
        )
    else:
        plan.suggested_next_message = (
            "Provide more known batch values so I can solve the remaining variables from the mole balance."
        )
    return plan


def _plan_consistency_check(classification: GoalClassification) -> WorkflowPlan:
    plan = _base_plan(classification)
    plan.available_inputs = _assigned_keys(classification)
    plan.ready_to_execute = len(plan.available_inputs) >= 4
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
        "solve_rayleigh_batch_variables": _plan_solve_rayleigh_batch_variables,
        "solve_mole_balance": _plan_solve_mole_balance,
        "consistency_check": _plan_consistency_check,
        "explain_variable_or_workflow": _plan_explanation,
        "unsupported_or_unclear": _plan_unsupported,
    }
    planner = planners.get(classification.goal, _plan_unsupported)
    return planner(classification)
