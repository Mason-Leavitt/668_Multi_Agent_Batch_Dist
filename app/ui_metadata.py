"""Shared metadata and lightweight formatters for UI and agent-support code.

This module is a helper/support layer. It does not classify requests, plan
workflows, or perform deterministic engineering calculations.
"""

from __future__ import annotations

from typing import Any

WORKFLOW_REGISTRY: list[dict[str, object]] = [
    {
        "workflow_id": "feed_to_product_sweep",
        "id": "feed_to_product_sweep",
        "display_name": "Feed to Product Sweep",
        "description": "Explore possible distillate outcomes from a known starting feed over different stopping compositions.",
        "required_inputs": ["feed volume + ABV or internal W0 + x0"],
        "optional_inputs": ["stopping-basis preferences when the workflow supports them"],
        "outputs": ["D", "xDavg", "candidate product rows", "product sweep plot"],
        "supported_status": "implemented",
        "related_goal_names": ["feed_to_product_sweep"],
        "notes": "Best when the user knows the starting feed and wants a range of product outcomes.",
    },
    {
        "workflow_id": "product_to_feed_sweep",
        "id": "product_to_feed_sweep",
        "display_name": "Product to Feed Sweep",
        "description": "Explore candidate feed conditions that could produce a target product amount and composition.",
        "required_inputs": ["product volume + ABV or internal D + xDavg"],
        "optional_inputs": ["feed-side constraints if the user wants to narrow the sweep"],
        "outputs": ["W0", "x0", "candidate feed rows", "feed sweep plot"],
        "supported_status": "implemented",
        "related_goal_names": ["product_to_feed_sweep"],
        "notes": "Best when the user knows the desired product target and wants feasible feed combinations.",
    },
    {
        "workflow_id": "solve_mole_balance",
        "id": "solve_mole_balance",
        "display_name": "Mole-Balance Variable Solver",
        "description": "Solve selected batch variables from the total balance and ethanol balance only.",
        "required_inputs": ["a supported known-variable set for W0, x0, D, xDavg, B, and xB"],
        "optional_inputs": ["user-facing liters and ABV pairs that can be normalized internally"],
        "outputs": ["one solved batch row", "user-friendly liters and ABV when convertible"],
        "supported_status": "implemented",
        "related_goal_names": ["solve_mole_balance"],
        "notes": "Does not enforce Rayleigh batch-distillation behavior.",
    },
    {
        "workflow_id": "solve_rayleigh_batch_variables",
        "id": "solve_rayleigh_batch_variables",
        "display_name": "Rayleigh-Constrained Variable Solver",
        "description": "Solve direct batch-distillation cases using the Rayleigh equation plus total and ethanol balances.",
        "required_inputs": ["a supported known-variable set for one direct Rayleigh solve case"],
        "optional_inputs": ["user-facing feed, product, or bottoms values that can be normalized"],
        "outputs": ["one solved Rayleigh-constrained row", "user-friendly liters and ABV when convertible"],
        "supported_status": "implemented",
        "related_goal_names": ["solve_rayleigh_batch_variables"],
        "notes": "Used for single direct Rayleigh-constrained cases rather than broad sweeps.",
    },
]


VARIABLE_REGISTRY: list[dict[str, object]] = [
    {
        "key": "W0",
        "symbol": "W0",
        "display_name": "Initial feed amount",
        "description": "Initial amount of ethanol-water mixture charged to the still.",
        "meaning": "Initial total feed or still-charge amount of the ethanol-water mixture.",
        "user_facing_units": "L or gal",
        "internal_units": "total moles of mixture",
        "units": "User-facing: liters or gallons. Internal: total moles of mixture.",
        "aliases": ["initial feed", "starting charge", "feed amount", "feed volume", "starting volume", "W0"],
        "keywords": ["initial feed", "starting charge", "feed amount", "feed volume", "starting volume"],
        "display_order": 10,
        "related_variables": ["x0"],
    },
    {
        "key": "x0",
        "symbol": "x0",
        "display_name": "Initial feed composition",
        "description": "Initial ethanol composition of the feed or still charge.",
        "meaning": "Initial ethanol composition of the feed or still charge.",
        "user_facing_units": "ABV %",
        "internal_units": "ethanol liquid mole fraction",
        "units": "User-facing: ABV %. Internal: ethanol liquid mole fraction.",
        "aliases": ["feed ABV", "initial ABV", "starting ABV", "feed composition", "initial concentration", "x0"],
        "keywords": ["feed ABV", "initial ABV", "starting ABV", "feed composition", "initial concentration"],
        "display_order": 20,
        "related_variables": ["W0", "mole_fraction", "ABV"],
    },
    {
        "key": "D",
        "symbol": "D",
        "display_name": "Product amount",
        "description": "Total collected distillate or product amount of the ethanol-water mixture.",
        "meaning": "Total collected distillate or product amount of the ethanol-water mixture.",
        "user_facing_units": "L or gal",
        "internal_units": "total moles of mixture",
        "units": "User-facing: liters or gallons. Internal: total moles of mixture.",
        "aliases": ["product amount", "distillate amount", "product volume", "collected amount", "D"],
        "keywords": ["product amount", "distillate amount", "product volume", "collected amount", "how much product"],
        "display_order": 30,
        "related_variables": ["xDavg"],
    },
    {
        "key": "xDavg",
        "symbol": "xDavg",
        "display_name": "Average distillate composition",
        "description": "Average ethanol composition of the collected distillate mixture.",
        "meaning": "Average ethanol composition of the collected distillate mixture.",
        "user_facing_units": "ABV %",
        "internal_units": "ethanol liquid mole fraction",
        "units": "User-facing: ABV %. Internal: ethanol liquid mole fraction.",
        "aliases": ["product ABV", "distillate ABV", "average product composition", "average distillate strength", "xDavg"],
        "keywords": ["product ABV", "distillate ABV", "average product composition", "average distillate strength"],
        "display_order": 40,
        "related_variables": ["D", "mole_fraction", "ABV"],
    },
    {
        "key": "B",
        "symbol": "B / W",
        "display_name": "Remaining still amount",
        "description": "Final remaining still or bottoms amount of the ethanol-water mixture.",
        "meaning": "Final remaining still or bottoms amount of the ethanol-water mixture.",
        "user_facing_units": "L when converted",
        "internal_units": "total moles of mixture",
        "units": "User-facing: liters when converted. Internal: total moles of mixture.",
        "aliases": ["bottoms amount", "remaining still amount", "what is left in the still", "boiler amount", "B", "W"],
        "keywords": ["bottoms amount", "remaining still amount", "what is left in the still", "boiler amount"],
        "display_order": 50,
        "related_variables": ["xB"],
    },
    {
        "key": "xB",
        "symbol": "xB",
        "display_name": "Bottoms composition",
        "description": "Final ethanol composition of the remaining still or bottoms liquid.",
        "meaning": "Final ethanol composition of the remaining still or bottoms liquid.",
        "user_facing_units": "ABV % when converted",
        "internal_units": "ethanol liquid mole fraction",
        "units": "User-facing: ABV % when converted. Internal: ethanol liquid mole fraction.",
        "aliases": ["bottoms composition", "still composition", "remaining ABV", "boiler ABV", "stopping composition", "xB"],
        "keywords": ["bottoms composition", "still composition", "remaining ABV", "boiler ABV", "stopping composition"],
        "display_order": 60,
        "related_variables": ["B", "mole_fraction", "ABV"],
    },
    {
        "key": "ABV",
        "symbol": "ABV",
        "display_name": "Alcohol by volume",
        "description": "User-facing alcohol-by-volume basis for feed, product, or bottoms streams.",
        "meaning": "Alcohol by volume in user-facing terms for feed, product, or bottoms streams.",
        "user_facing_units": "percent ABV",
        "internal_units": "not used directly",
        "units": "Percent ABV.",
        "aliases": ["abv", "strength", "percent alcohol", "ethanol concentration"],
        "keywords": ["abv", "strength", "percent alcohol", "ethanol concentration"],
        "display_order": 70,
        "related_variables": ["x0", "xDavg", "xB"],
    },
    {
        "key": "mole_fraction",
        "symbol": "Mole Fraction",
        "display_name": "Internal composition basis",
        "description": "Internal ethanol liquid mole-fraction basis used by the engineering calculations.",
        "meaning": "Internal ethanol composition basis used by the engineering calculations.",
        "user_facing_units": "not shown directly",
        "internal_units": "unitless fraction from 0 to 1",
        "units": "Unitless fraction from 0 to 1.",
        "aliases": ["mole fraction", "x0", "xB", "xDavg", "internal composition"],
        "keywords": ["mole fraction", "x0", "xB", "xDavg", "internal composition"],
        "display_order": 80,
        "related_variables": ["x0", "xDavg", "xB"],
    },
]

IMPLEMENTED_WORKFLOWS = WORKFLOW_REGISTRY
VARIABLE_REFERENCE = VARIABLE_REGISTRY

USER_FACING_BATCH_INPUT_REGISTRY: list[dict[str, object]] = [
    {
        "key": "feed_volume",
        "display_name": "Feed volume",
        "description": "User-facing initial charge volume before normalization.",
        "user_facing_units": "L or gal",
        "related_internal_variable": "W0",
        "input_kind": "amount",
        "display_order": 10,
    },
    {
        "key": "feed_abv",
        "display_name": "Feed ABV",
        "description": "User-facing initial ethanol concentration before normalization.",
        "user_facing_units": "% ABV",
        "related_internal_variable": "x0",
        "input_kind": "composition",
        "display_order": 20,
    },
    {
        "key": "product_volume",
        "display_name": "Product volume",
        "description": "User-facing collected product volume before normalization.",
        "user_facing_units": "L or gal",
        "related_internal_variable": "D",
        "input_kind": "amount",
        "display_order": 30,
    },
    {
        "key": "product_abv",
        "display_name": "Product ABV",
        "description": "User-facing average product ethanol concentration before normalization.",
        "user_facing_units": "% ABV",
        "related_internal_variable": "xDavg",
        "input_kind": "composition",
        "display_order": 40,
    },
    {
        "key": "bottoms_volume",
        "display_name": "Bottoms volume",
        "description": "User-facing remaining still or bottoms volume before normalization.",
        "user_facing_units": "L or gal",
        "related_internal_variable": "B",
        "input_kind": "amount",
        "display_order": 50,
    },
    {
        "key": "bottoms_abv",
        "display_name": "Bottoms ABV",
        "description": "User-facing remaining still or bottoms ethanol concentration before normalization.",
        "user_facing_units": "% ABV",
        "related_internal_variable": "xB",
        "input_kind": "composition",
        "display_order": 60,
    },
]


def get_workflow_reference() -> list[dict[str, Any]]:
    return list(WORKFLOW_REGISTRY)


def get_workflow_metadata(workflow_id: str) -> dict[str, Any] | None:
    for workflow in WORKFLOW_REGISTRY:
        if workflow.get("workflow_id") == workflow_id or workflow.get("id") == workflow_id:
            return dict(workflow)
    return None


def get_variable_reference() -> list[dict[str, Any]]:
    return sorted(VARIABLE_REGISTRY, key=lambda item: int(item.get("display_order", 999)))


def get_variable_metadata(key: str) -> dict[str, Any] | None:
    normalized_key = key.strip()
    for variable in VARIABLE_REGISTRY:
        aliases = {str(alias).lower() for alias in variable.get("aliases", [])}
        if normalized_key == variable.get("key") or normalized_key == variable.get("symbol"):
            return dict(variable)
        if normalized_key.lower() in aliases:
            return dict(variable)
    return None


def get_core_batch_variable_keys() -> list[str]:
    return [
        str(variable["key"])
        for variable in get_variable_reference()
        if str(variable["key"]) not in {"ABV", "mole_fraction"}
    ]


def get_user_facing_batch_input_reference() -> list[dict[str, Any]]:
    return sorted(
        USER_FACING_BATCH_INPUT_REGISTRY,
        key=lambda item: int(item.get("display_order", 999)),
    )


def get_user_facing_batch_input_metadata(key: str) -> dict[str, Any] | None:
    normalized_key = key.strip()
    for item in USER_FACING_BATCH_INPUT_REGISTRY:
        if normalized_key == item.get("key"):
            return dict(item)
    return None


def get_user_facing_batch_input_keys() -> list[str]:
    return [
        str(item["key"])
        for item in get_user_facing_batch_input_reference()
    ]


def format_core_variable_reference() -> str:
    lines = ["Core variable glossary:"]
    for variable in get_variable_reference():
        symbol = str(variable["symbol"])
        display_name = str(variable["display_name"])
        description = str(variable["description"])
        user_units = str(variable["user_facing_units"])
        internal_units = str(variable["internal_units"])
        aliases = ", ".join(str(alias) for alias in variable.get("aliases", [])[:5])
        lines.append("")
        lines.append(f"{symbol} — {display_name}")
        lines.append(f"- Meaning: {description}")
        lines.append(f"- User-facing units: {user_units}")
        lines.append(f"- Internal units: {internal_units}")
        if aliases:
            lines.append(f"- Keywords: {aliases}")
    return "\n".join(lines)


def format_user_facing_batch_input_reference() -> str:
    lines = ["User-facing batch input forms:"]
    for item in get_user_facing_batch_input_reference():
        key = str(item["key"])
        display_name = str(item["display_name"])
        description = str(item["description"])
        units = str(item["user_facing_units"])
        related_internal = str(item["related_internal_variable"])
        input_kind = str(item["input_kind"])
        lines.append("")
        lines.append(f"{key} — {display_name}")
        lines.append(f"- Meaning: {description}")
        lines.append(f"- Units: {units}")
        lines.append(f"- Normalizes to: {related_internal}")
        lines.append(f"- Kind: {input_kind}")
    return "\n".join(lines)


def format_variable_and_input_reference() -> str:
    return (
        f"{format_core_variable_reference()}\n\n"
        f"{format_user_facing_batch_input_reference()}"
    )


def get_missing_value_display_name(key: str) -> str:
    metadata = get_variable_metadata(key)
    if metadata is None:
        return key
    if key == "B":
        return "remaining still amount B or W"
    return f"{metadata['display_name']} {metadata['symbol']}"


def _format_scalar(value: Any) -> str:
    if value is None:
        return "unspecified"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def get_current_known_variables(session_state: Any) -> list[dict[str, str]]:
    """Collect the current known variable state from existing Streamlit session state."""

    values: dict[str, dict[str, str]] = {}

    def upsert(key: str, value: Any, status: str, *, internal: bool = False) -> None:
        if value is None:
            return
        entry = values.setdefault(key, {})
        target = "internal_value" if internal else "value"
        status_key = "internal_status" if internal else "status"
        if target not in entry:
            entry[target] = _format_scalar(value)
            entry[status_key] = status

    pending_incomplete = getattr(session_state, "pending_incomplete_workflow", None)
    if pending_incomplete:
        analysis = pending_incomplete.get("intent_analysis")
        if analysis is not None:
            for key, value in analysis.known_quantities.items():
                upsert(key, value, "user-provided")
            for key, value in analysis.known_quantities_needing_normalization.items():
                upsert(key, value, "needs normalization")

    pending_classification = getattr(session_state, "pending_classification", None)
    if pending_classification is not None:
        for key, value in pending_classification.variable_assignments.items():
            upsert(key, value, "user-provided")

    last_classification = getattr(session_state, "last_classification", None)
    if last_classification is not None:
        for key, value in last_classification.variable_assignments.items():
            upsert(key, value, "user-provided")

    pending_plan = getattr(session_state, "pending_plan", None)
    if pending_plan is not None:
        for missing in pending_plan.missing_inputs:
            metadata = get_variable_metadata(str(missing))
            canonical_key = str(metadata["key"]) if metadata is not None else str(missing)
            values.setdefault(canonical_key, {})

    last_result = getattr(session_state, "last_execution_result", None)
    if last_result is not None:
        normalized = last_result.normalized_inputs or {}
        for key in get_core_batch_variable_keys():
            if key in normalized:
                upsert(key, normalized[key], "normalized", internal=True)
        row0 = last_result.rows[0] if last_result.rows else {}
        for key in get_core_batch_variable_keys():
            if key in row0:
                upsert(key, row0[key], "calculated", internal=True)

    rows: list[dict[str, str]] = []
    for variable in get_variable_reference():
        key = str(variable["key"])
        if key in {"ABV", "mole_fraction"}:
            continue

        value_entry = values.get(key, {})
        current_value = value_entry.get("value", "unspecified")
        status = value_entry.get("status", "not yet known")

        if "internal_value" in value_entry:
            if current_value == "unspecified":
                current_value = f"{value_entry['internal_value']} (internal)"
                status = value_entry["internal_status"]
            else:
                current_value = f"{current_value} | internal {value_entry['internal_value']}"
            if "internal_status" in value_entry and status != value_entry["internal_status"] and current_value != f"{value_entry['internal_value']} (internal)":
                status = f"{status}; {value_entry['internal_status']}"

        rows.append(
            {
                "Variable": str(variable["symbol"]),
                "Meaning": str(variable["display_name"]),
                "Current Value": current_value,
                "Status": status,
            }
        )

    return rows
