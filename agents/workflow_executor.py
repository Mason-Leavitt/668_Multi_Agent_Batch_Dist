"""Deterministic workflow execution for implemented planner paths."""

from __future__ import annotations

import math
import re

from engineering.calculations import (
    composite_simpson_rule_rayleigh,
    find_D_xDavg_combinations_from_W0_x0,
    find_W0_x0_combinations_from_xDavg_D,
    get_B_rayleigh,
    get_B_mole_balance,
    get_D_mole_balance,
    get_W0_mole_balance,
    get_x0_mole_balance,
    get_xB_mole_balance,
    get_xDavg_mole_balance_1,
)
from engineering.conversions import (
    get_abv_from_mol_frac,
    get_mixture_volume_L_from_moles,
    get_moles_and_etoh_frac_from_volume_L_and_abv,
)
from scipy.optimize import brentq

from .execution_schemas import ExecutionParameterValue, WorkflowExecutionResult
from .schemas import GoalClassification, ScalarValue
from .workflow_schemas import WorkflowPlan

GALLON_TO_LITER = 3.78541
DEFAULT_NUM_XB = 100
DEFAULT_XB_MIN = 0.001
DEFAULT_XB_BUFFER = 0.001
DEFAULT_SIMPSON_N = 100
DEFAULT_NUM_X0 = 250
DEFAULT_PRODUCT_NUM_XB = 250
DEFAULT_PRODUCT_TOLERANCE = 0.01
DEFAULT_RAYLEIGH_UPPER_X = 0.89
DEFAULT_ROOT_EPS = 1e-6


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


def _abv_percent_to_mol_frac(abv_percent: float) -> float:
    converted = get_moles_and_etoh_frac_from_volume_L_and_abv(
        volume_L=1.0,
        abv_percent=abv_percent,
    )
    return float(converted["x_etoh"])


def _parse_composition_value(value: ScalarValue, field_name: str) -> float:
    if isinstance(value, str):
        lowered = value.lower()
        if "%" in lowered or "abv" in lowered or "proof" in lowered:
            return _abv_percent_to_mol_frac(_parse_abv_to_percent(value, field_name))
    return _parse_model_scalar(value, field_name)


def _moles_from_volume_L_and_mol_frac(volume_L: float, x_etoh: float) -> float:
    one_mole_volume_L = float(get_mixture_volume_L_from_moles(1.0, x_etoh))
    if one_mole_volume_L <= 0:
        raise ValueError("Could not determine a positive mixture volume basis for the supplied composition.")
    return volume_L / one_mole_volume_L


def _w0_is_volume_like(classification: GoalClassification) -> bool:
    return _has_volume_units(classification.variable_assignments.get("W0"))


def _d_is_volume_like(classification: GoalClassification) -> bool:
    return _has_volume_units(classification.variable_assignments.get("D"))


def _collect_preliminary_normalized_inputs(
    classification: GoalClassification,
) -> dict[str, ScalarValue]:
    assignments = classification.variable_assignments
    normalized_inputs: dict[str, ScalarValue] = {}

    try:
        if "W0" in assignments and "feed_abv" in assignments and _w0_is_volume_like(classification):
            normalized_inputs["feed_volume_L"] = _parse_volume_to_liters(assignments["W0"], "W0")
            normalized_inputs["feed_abv_fraction"] = _parse_abv_to_percent(assignments["feed_abv"], "feed_abv") / 100.0
            normalized_inputs["input_source"] = "mixed_units_normalized_to_model_units"
        elif "W0" in assignments and "x0" in assignments and _w0_is_volume_like(classification):
            volume_l = _parse_volume_to_liters(assignments["W0"], "W0")
            x0 = _parse_composition_value(assignments["x0"], "x0")
            normalized_inputs["feed_volume_L"] = volume_l
            normalized_inputs["W0"] = _moles_from_volume_L_and_mol_frac(volume_l, x0)
            normalized_inputs["x0"] = x0
            normalized_inputs["input_source"] = "mixed_units_normalized_to_model_units"
        elif "feed_volume" in assignments and "feed_abv" in assignments:
            normalized_inputs["feed_volume_L"] = _parse_volume_to_liters(assignments["feed_volume"], "feed_volume")
            normalized_inputs["feed_abv_fraction"] = _parse_abv_to_percent(assignments["feed_abv"], "feed_abv") / 100.0
            normalized_inputs["input_source"] = "volume_abv"
        elif "W0" in assignments and "x0" in assignments and not _w0_is_volume_like(classification):
            normalized_inputs["W0"] = _parse_model_scalar(assignments["W0"], "W0")
            normalized_inputs["x0"] = _parse_composition_value(assignments["x0"], "x0")
            normalized_inputs["input_source"] = "model_units"
        elif "D" in assignments and "xDavg" in assignments and _d_is_volume_like(classification):
            normalized_inputs["product_volume_L"] = _parse_volume_to_liters(assignments["D"], "D")
            normalized_inputs["product_abv_fraction"] = _parse_abv_to_percent(assignments["xDavg"], "xDavg") / 100.0
            normalized_inputs["input_source"] = "mixed_units_normalized_to_model_units"
        elif "D" in assignments and "xDavg" in assignments and _d_is_volume_like(classification):
            volume_l = _parse_volume_to_liters(assignments["D"], "D")
            xDavg = _parse_composition_value(assignments["xDavg"], "xDavg")
            normalized_inputs["product_volume_L"] = volume_l
            normalized_inputs["D"] = _moles_from_volume_L_and_mol_frac(volume_l, xDavg)
            normalized_inputs["xDavg"] = xDavg
            normalized_inputs["input_source"] = "mixed_units_normalized_to_model_units"
        elif "product_volume" in assignments and "product_abv" in assignments:
            normalized_inputs["product_volume_L"] = _parse_volume_to_liters(assignments["product_volume"], "product_volume")
            normalized_inputs["product_abv_fraction"] = _parse_abv_to_percent(assignments["product_abv"], "product_abv") / 100.0
            normalized_inputs["input_source"] = "volume_abv"
        elif "D" in assignments and "xDavg" in assignments and not _d_is_volume_like(classification):
            normalized_inputs["D"] = _parse_model_scalar(assignments["D"], "D")
            normalized_inputs["xDavg"] = _parse_composition_value(assignments["xDavg"], "xDavg")
            normalized_inputs["input_source"] = "model_units"
        elif "B" in assignments and "bottoms_abv" in assignments and _has_volume_units(assignments["B"]):
            normalized_inputs["bottoms_volume_L"] = _parse_volume_to_liters(assignments["B"], "B")
            normalized_inputs["bottoms_abv_fraction"] = _parse_abv_to_percent(assignments["bottoms_abv"], "bottoms_abv") / 100.0
            normalized_inputs["input_source"] = "mixed_units_normalized_to_model_units"
        elif "B" in assignments and "xB" in assignments and _has_volume_units(assignments["B"]):
            volume_l = _parse_volume_to_liters(assignments["B"], "B")
            xB = _parse_composition_value(assignments["xB"], "xB")
            normalized_inputs["bottoms_volume_L"] = volume_l
            normalized_inputs["B"] = _moles_from_volume_L_and_mol_frac(volume_l, xB)
            normalized_inputs["xB"] = xB
            normalized_inputs["input_source"] = "mixed_units_normalized_to_model_units"
        elif "bottoms_volume" in assignments and "bottoms_abv" in assignments:
            normalized_inputs["bottoms_volume_L"] = _parse_volume_to_liters(assignments["bottoms_volume"], "bottoms_volume")
            normalized_inputs["bottoms_abv_fraction"] = _parse_abv_to_percent(assignments["bottoms_abv"], "bottoms_abv") / 100.0
            normalized_inputs["input_source"] = "volume_abv"
        elif "B" in assignments and "xB" in assignments and not _has_volume_units(assignments["B"]):
            normalized_inputs["B"] = _parse_model_scalar(assignments["B"], "B")
            normalized_inputs["xB"] = _parse_composition_value(assignments["xB"], "xB")
            normalized_inputs["input_source"] = "model_units"
    except Exception:
        return normalized_inputs

    return normalized_inputs


def _normalize_feed_inputs(
    classification: GoalClassification,
) -> tuple[float, float, dict[str, ScalarValue], list[str]]:
    warnings: list[str] = []
    normalized_inputs: dict[str, ScalarValue] = {}
    assignments = classification.variable_assignments

    if "W0" in assignments and "feed_abv" in assignments and _w0_is_volume_like(classification):
        volume_l = _parse_volume_to_liters(assignments["W0"], "W0")
        abv_percent = _parse_abv_to_percent(assignments["feed_abv"], "feed_abv")
        normalized_inputs["feed_volume_L"] = volume_l
        normalized_inputs["feed_abv_fraction"] = abv_percent / 100.0
        normalized_inputs["input_source"] = "mixed_units_normalized_to_model_units"
        if isinstance(assignments["W0"], str):
            lowered = assignments["W0"].lower()
            if "gallon" in lowered or re.search(r"\bgal\b", lowered):
                normalized_inputs["feed_volume_gal"] = _extract_first_number(assignments["W0"], "W0")
        converted = get_moles_and_etoh_frac_from_volume_L_and_abv(
            volume_L=volume_l,
            abv_percent=abv_percent,
        )
        normalized_inputs["W0"] = converted["total_moles"]
        normalized_inputs["x0"] = converted["x_etoh"]
        warnings.append(
            "W0 was interpreted as a user-friendly feed volume and normalized to internal W0 and x0."
        )
        return converted["total_moles"], converted["x_etoh"], normalized_inputs, warnings

    if "W0" in assignments and "x0" in assignments and _w0_is_volume_like(classification):
        volume_l = _parse_volume_to_liters(assignments["W0"], "W0")
        x0 = _parse_composition_value(assignments["x0"], "x0")
        normalized_inputs["feed_volume_L"] = volume_l
        normalized_inputs["W0"] = _moles_from_volume_L_and_mol_frac(volume_l, x0)
        normalized_inputs["x0"] = x0
        normalized_inputs["input_source"] = "mixed_units_normalized_to_model_units"
        warnings.append(
            "W0 was interpreted as a user-friendly feed volume and normalized to internal W0 using the provided x0 composition."
        )
        return normalized_inputs["W0"], x0, normalized_inputs, warnings

    if "feed_volume" in assignments and "feed_abv" in assignments:
        volume_l = _parse_volume_to_liters(assignments["feed_volume"], "feed_volume")
        abv_percent = _parse_abv_to_percent(assignments["feed_abv"], "feed_abv")
        normalized_inputs["feed_volume_L"] = volume_l
        normalized_inputs["feed_abv_fraction"] = abv_percent / 100.0
        normalized_inputs["input_source"] = "volume_abv"
        if isinstance(assignments["feed_volume"], str):
            lowered = assignments["feed_volume"].lower()
            if "gallon" in lowered or re.search(r"\bgal\b", lowered):
                normalized_inputs["feed_volume_gal"] = _extract_first_number(assignments["feed_volume"], "feed_volume")
        converted = get_moles_and_etoh_frac_from_volume_L_and_abv(
            volume_L=volume_l,
            abv_percent=abv_percent,
        )
        normalized_inputs["W0"] = converted["total_moles"]
        normalized_inputs["x0"] = converted["x_etoh"]
        return converted["total_moles"], converted["x_etoh"], normalized_inputs, warnings

    if "W0" in assignments and "x0" in assignments and not _w0_is_volume_like(classification):
        W0 = _parse_model_scalar(assignments["W0"], "W0")
        x0 = _parse_composition_value(assignments["x0"], "x0")
        normalized_inputs["W0"] = W0
        normalized_inputs["x0"] = x0
        normalized_inputs["input_source"] = "model_units"
        return (W0, x0, normalized_inputs, warnings)

    raise ValueError(
        "Feed normalization needs either W0 and x0 values, feed_volume and feed_abv values, or a volume-like W0 plus either x0 or feed_abv."
    )


def _normalize_product_inputs(
    classification: GoalClassification,
) -> tuple[float, float, dict[str, ScalarValue], list[str]]:
    warnings: list[str] = []
    normalized_inputs: dict[str, ScalarValue] = {}
    assignments = classification.variable_assignments

    if "D" in assignments and "product_abv" in assignments and _d_is_volume_like(classification):
        volume_l = _parse_volume_to_liters(assignments["D"], "D")
        abv_percent = _parse_abv_to_percent(assignments["product_abv"], "product_abv")
        normalized_inputs["product_volume_L"] = volume_l
        normalized_inputs["product_abv_fraction"] = abv_percent / 100.0
        normalized_inputs["input_source"] = "mixed_units_normalized_to_model_units"
        if isinstance(assignments["D"], str):
            lowered = assignments["D"].lower()
            if "gallon" in lowered or re.search(r"\bgal\b", lowered):
                normalized_inputs["product_volume_gal"] = _extract_first_number(assignments["D"], "D")
        converted = get_moles_and_etoh_frac_from_volume_L_and_abv(
            volume_L=volume_l,
            abv_percent=abv_percent,
        )
        normalized_inputs["D"] = converted["total_moles"]
        normalized_inputs["xDavg"] = converted["x_etoh"]
        warnings.append(
            "D was interpreted as a user-friendly product volume and normalized to internal D and xDavg."
        )
        return converted["total_moles"], converted["x_etoh"], normalized_inputs, warnings

    if "D" in assignments and "xDavg" in assignments and _d_is_volume_like(classification):
        volume_l = _parse_volume_to_liters(assignments["D"], "D")
        xDavg_value = _parse_composition_value(assignments["xDavg"], "xDavg")
        normalized_inputs["product_volume_L"] = volume_l
        normalized_inputs["input_source"] = "mixed_units_normalized_to_model_units"
        if isinstance(assignments["D"], str):
            lowered = assignments["D"].lower()
            if "gallon" in lowered or re.search(r"\bgal\b", lowered):
                normalized_inputs["product_volume_gal"] = _extract_first_number(assignments["D"], "D")
        normalized_inputs["D"] = _moles_from_volume_L_and_mol_frac(volume_l, xDavg_value)
        normalized_inputs["xDavg"] = xDavg_value
        warnings.append(
            "D was interpreted as a user-friendly product volume and normalized to internal D using the provided xDavg composition."
        )
        return normalized_inputs["D"], xDavg_value, normalized_inputs, warnings

    if "product_volume" in assignments and "product_abv" in assignments:
        volume_l = _parse_volume_to_liters(assignments["product_volume"], "product_volume")
        abv_percent = _parse_abv_to_percent(assignments["product_abv"], "product_abv")
        normalized_inputs["product_volume_L"] = volume_l
        normalized_inputs["product_abv_fraction"] = abv_percent / 100.0
        normalized_inputs["input_source"] = "volume_abv"
        if isinstance(assignments["product_volume"], str):
            lowered = assignments["product_volume"].lower()
            if "gallon" in lowered or re.search(r"\bgal\b", lowered):
                normalized_inputs["product_volume_gal"] = _extract_first_number(assignments["product_volume"], "product_volume")
        converted = get_moles_and_etoh_frac_from_volume_L_and_abv(
            volume_L=volume_l,
            abv_percent=abv_percent,
        )
        normalized_inputs["D"] = converted["total_moles"]
        normalized_inputs["xDavg"] = converted["x_etoh"]
        return converted["total_moles"], converted["x_etoh"], normalized_inputs, warnings

    if "D" in assignments and "xDavg" in assignments and not _d_is_volume_like(classification):
        D_value = _parse_model_scalar(assignments["D"], "D")
        xDavg_value = _parse_composition_value(assignments["xDavg"], "xDavg")
        normalized_inputs["D"] = D_value
        normalized_inputs["xDavg"] = xDavg_value
        normalized_inputs["input_source"] = "model_units"
        return D_value, xDavg_value, normalized_inputs, warnings

    raise ValueError(
        "Product normalization needs either D and xDavg values, product_volume and product_abv values, or a volume-like D plus either xDavg or product_abv."
    )


def _normalize_bottoms_inputs(
    classification: GoalClassification,
) -> tuple[float, float, dict[str, ScalarValue], list[str]]:
    warnings: list[str] = []
    normalized_inputs: dict[str, ScalarValue] = {}
    assignments = classification.variable_assignments

    if "B" in assignments and "bottoms_abv" in assignments and _has_volume_units(assignments["B"]):
        volume_l = _parse_volume_to_liters(assignments["B"], "B")
        abv_percent = _parse_abv_to_percent(assignments["bottoms_abv"], "bottoms_abv")
        normalized_inputs["bottoms_volume_L"] = volume_l
        normalized_inputs["bottoms_abv_fraction"] = abv_percent / 100.0
        normalized_inputs["input_source"] = "mixed_units_normalized_to_model_units"
        converted = get_moles_and_etoh_frac_from_volume_L_and_abv(
            volume_L=volume_l,
            abv_percent=abv_percent,
        )
        normalized_inputs["B"] = converted["total_moles"]
        normalized_inputs["xB"] = converted["x_etoh"]
        warnings.append(
            "B was interpreted as a user-friendly bottoms volume and normalized to internal B and xB."
        )
        return converted["total_moles"], converted["x_etoh"], normalized_inputs, warnings

    if "B" in assignments and "xB" in assignments and _has_volume_units(assignments["B"]):
        volume_l = _parse_volume_to_liters(assignments["B"], "B")
        xB_value = _parse_composition_value(assignments["xB"], "xB")
        normalized_inputs["bottoms_volume_L"] = volume_l
        normalized_inputs["B"] = _moles_from_volume_L_and_mol_frac(volume_l, xB_value)
        normalized_inputs["xB"] = xB_value
        normalized_inputs["input_source"] = "mixed_units_normalized_to_model_units"
        warnings.append(
            "B was interpreted as a user-friendly bottoms volume and normalized to internal B using the provided xB composition."
        )
        return normalized_inputs["B"], xB_value, normalized_inputs, warnings

    if "bottoms_volume" in assignments and "bottoms_abv" in assignments:
        volume_l = _parse_volume_to_liters(assignments["bottoms_volume"], "bottoms_volume")
        abv_percent = _parse_abv_to_percent(assignments["bottoms_abv"], "bottoms_abv")
        normalized_inputs["bottoms_volume_L"] = volume_l
        normalized_inputs["bottoms_abv_fraction"] = abv_percent / 100.0
        normalized_inputs["input_source"] = "volume_abv"
        converted = get_moles_and_etoh_frac_from_volume_L_and_abv(
            volume_L=volume_l,
            abv_percent=abv_percent,
        )
        normalized_inputs["B"] = converted["total_moles"]
        normalized_inputs["xB"] = converted["x_etoh"]
        return converted["total_moles"], converted["x_etoh"], normalized_inputs, warnings

    if "B" in assignments and "xB" in assignments and not _has_volume_units(assignments["B"]):
        B_value = _parse_model_scalar(assignments["B"], "B")
        xB_value = _parse_composition_value(assignments["xB"], "xB")
        normalized_inputs["B"] = B_value
        normalized_inputs["xB"] = xB_value
        normalized_inputs["input_source"] = "model_units"
        return B_value, xB_value, normalized_inputs, warnings

    raise ValueError(
        "Bottoms normalization needs either B and xB values, bottoms_volume and bottoms_abv values, or a volume-like B plus either xB or bottoms_abv."
    )


def _normalize_abv_only_inputs(classification: GoalClassification) -> dict[str, float]:
    assignments = classification.variable_assignments
    normalized: dict[str, float] = {}

    if "feed_abv" in assignments:
        normalized["x0"] = _abv_percent_to_mol_frac(_parse_abv_to_percent(assignments["feed_abv"], "feed_abv"))
    if "product_abv" in assignments:
        normalized["xDavg"] = _abv_percent_to_mol_frac(_parse_abv_to_percent(assignments["product_abv"], "product_abv"))
    if "bottoms_abv" in assignments:
        normalized["xB"] = _abv_percent_to_mol_frac(_parse_abv_to_percent(assignments["bottoms_abv"], "bottoms_abv"))

    return normalized


def _merge_normalized_inputs(*parts: dict[str, ScalarValue]) -> dict[str, ScalarValue]:
    merged: dict[str, ScalarValue] = {}
    input_sources: list[str] = []
    for part in parts:
        for key, value in part.items():
            if key == "input_source":
                if value is not None:
                    input_sources.append(str(value))
            else:
                merged[key] = value
    if input_sources:
        merged["input_source"] = input_sources[0] if len(set(input_sources)) == 1 else "mixed_inputs_normalized_to_model_units"
    return merged


def _populate_user_friendly_balance_columns(row: dict[str, ScalarValue], warnings: list[str]) -> None:
    try:
        if row.get("W0") is not None and row.get("x0") is not None:
            row["W0_volume_L"] = round(
                float(get_mixture_volume_L_from_moles(float(row["W0"]), float(row["x0"]))),
                6,
            )
            row["x0_abv_percent"] = round(
                float(get_abv_from_mol_frac(float(row["x0"]))),
                6,
            )
    except (ValueError, TypeError, ArithmeticError, ZeroDivisionError) as exc:
        row["W0_volume_L"] = None
        row["x0_abv_percent"] = None
        warnings.append(f"Feed conversion failed while formatting balance output: {exc}")

    try:
        if row.get("D") is not None and row.get("xDavg") is not None:
            row["D_volume_L"] = round(
                float(get_mixture_volume_L_from_moles(float(row["D"]), float(row["xDavg"]))),
                6,
            )
            row["xDavg_abv_percent"] = round(
                float(get_abv_from_mol_frac(float(row["xDavg"]))),
                6,
            )
    except (ValueError, TypeError, ArithmeticError, ZeroDivisionError) as exc:
        row["D_volume_L"] = None
        row["xDavg_abv_percent"] = None
        warnings.append(f"Product conversion failed while formatting balance output: {exc}")

    try:
        if row.get("B") is not None and row.get("xB") is not None:
            row["B_volume_L"] = round(
                float(get_mixture_volume_L_from_moles(float(row["B"]), float(row["xB"]))),
                6,
            )
            row["xB_abv_percent"] = round(
                float(get_abv_from_mol_frac(float(row["xB"]))),
                6,
            )
    except (ValueError, TypeError, ArithmeticError, ZeroDivisionError) as exc:
        row["B_volume_L"] = None
        row["xB_abv_percent"] = None
        warnings.append(f"Bottoms conversion failed while formatting balance output: {exc}")


def _solve_root_brentq(func, lower: float, upper: float, label: str) -> float:
    f_lower = func(lower)
    f_upper = func(upper)

    if f_lower == 0:
        return lower
    if f_upper == 0:
        return upper
    if f_lower * f_upper > 0:
        raise ValueError(f"Could not bracket a root for {label} on [{lower}, {upper}].")

    return float(brentq(func, lower, upper))


def execute_feed_to_product_sweep(
    classification: GoalClassification,
    plan: WorkflowPlan,
) -> WorkflowExecutionResult:
    warnings: list[str] = []
    execution_parameters: dict[str, ExecutionParameterValue] = {
        "sweep_variable": classification.sweep_variable or "xB",
        "num_xB": DEFAULT_NUM_XB,
        "xB_min": DEFAULT_XB_MIN,
        "xB_buffer": DEFAULT_XB_BUFFER,
        "simpson_n": DEFAULT_SIMPSON_N,
    }
    normalized_inputs: dict[str, ScalarValue] = {}

    try:
        W0, x0, normalized_inputs, normalization_warnings = _normalize_feed_inputs(classification)
        warnings.extend(normalization_warnings)
        raw_results = find_D_xDavg_combinations_from_W0_x0(
            W0=W0,
            x0=x0,
            num_xB=DEFAULT_NUM_XB,
            xB_min=DEFAULT_XB_MIN,
            xB_buffer=DEFAULT_XB_BUFFER,
            n=DEFAULT_SIMPSON_N,
        )
    except (ValueError, TypeError, ArithmeticError, ZeroDivisionError) as exc:
        if not normalized_inputs:
            normalized_inputs = _collect_preliminary_normalized_inputs(classification)
        return WorkflowExecutionResult(
            workflow_name=plan.workflow_name,
            success=False,
            message="Workflow execution failed during deterministic engineering calculation.",
            warnings=[str(exc)],
            normalized_inputs=normalized_inputs,
            execution_parameters=execution_parameters,
        )

    try:
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
                row["xDavg_abv_percent"] = round(
                    float(get_abv_from_mol_frac(float(result["xDavg"]))),
                    6,
                )
                row["xB_abv_percent"] = round(
                    float(get_abv_from_mol_frac(float(result["xB"]))),
                    6,
                )
            rows.append(row)

        columns = ["xB", "D", "xDavg"]
        warnings.append("xDavg is reported as an ethanol mole fraction.")
        if include_user_friendly_output:
            columns.extend(["D_volume_L", "xDavg_abv_percent", "xB_abv_percent"])
            warnings.append(
                "xDavg-to-ABV conversion is not shown yet because converting mole fraction to volume ABV requires a dedicated mixture conversion."
            )
    except (ValueError, TypeError, ArithmeticError, ZeroDivisionError) as exc:
        return WorkflowExecutionResult(
            workflow_name=plan.workflow_name,
            success=False,
            message="Workflow execution failed during deterministic engineering calculation.",
            warnings=[str(exc)],
            normalized_inputs=normalized_inputs,
            execution_parameters=execution_parameters,
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
        normalized_inputs=normalized_inputs,
        execution_parameters=execution_parameters,
    )


def execute_product_to_feed_sweep(
    classification: GoalClassification,
    plan: WorkflowPlan,
) -> WorkflowExecutionResult:
    warnings: list[str] = []
    execution_parameters: dict[str, ExecutionParameterValue] = {
        "workflow": "product_to_feed_sweep",
        "num_x0": DEFAULT_NUM_X0,
        "num_xB": DEFAULT_PRODUCT_NUM_XB,
        "tolerance": DEFAULT_PRODUCT_TOLERANCE,
        "simpson_n": DEFAULT_SIMPSON_N,
    }
    normalized_inputs: dict[str, ScalarValue] = {}

    try:
        D_target, xDavg_target, normalized_inputs, normalization_warnings = _normalize_product_inputs(classification)
        warnings.extend(normalization_warnings)
        execution_parameters["target_D"] = D_target
        execution_parameters["target_xDavg"] = xDavg_target
        raw_results = find_W0_x0_combinations_from_xDavg_D(
            D_target=D_target,
            xDavg_target=xDavg_target,
            num_x0=DEFAULT_NUM_X0,
            num_xB=DEFAULT_PRODUCT_NUM_XB,
            tolerance=DEFAULT_PRODUCT_TOLERANCE,
            n=DEFAULT_SIMPSON_N,
        )
    except (ValueError, TypeError, ArithmeticError, ZeroDivisionError) as exc:
        if not normalized_inputs:
            normalized_inputs = _collect_preliminary_normalized_inputs(classification)
        return WorkflowExecutionResult(
            workflow_name=plan.workflow_name,
            success=False,
            message="Workflow execution failed during deterministic engineering calculation.",
            warnings=[str(exc)],
            normalized_inputs=normalized_inputs,
            execution_parameters=execution_parameters,
        )

    try:
        rows: list[dict[str, ScalarValue]] = []
        row_warnings: list[str] = []

        for result in raw_results:
            row: dict[str, ScalarValue] = {
                "W0": round(float(result["W0"]), 6) if result.get("W0") is not None else None,
                "x0": round(float(result["x0"]), 6) if result.get("x0") is not None else None,
                "xB": round(float(result["xB"]), 6) if result.get("xB") is not None else None,
                "D": round(float(result["D"]), 6) if result.get("D") is not None else None,
                "xDavg": round(float(result["xDavg"]), 6) if result.get("xDavg") is not None else None,
                "error": round(float(result["error"]), 6) if result.get("error") is not None else None,
            }

            try:
                if result.get("W0") is not None and result.get("x0") is not None:
                    row["W0_volume_L"] = round(
                        float(get_mixture_volume_L_from_moles(float(result["W0"]), float(result["x0"]))),
                        6,
                    )
                    row["x0_abv_percent"] = round(
                        float(get_abv_from_mol_frac(float(result["x0"]))),
                        6,
                    )
            except (ValueError, TypeError, ArithmeticError, ZeroDivisionError) as exc:
                row["W0_volume_L"] = None
                row["x0_abv_percent"] = None
                row_warnings.append(f"Feed conversion failed for one candidate row: {exc}")

            try:
                if result.get("xB") is not None:
                    row["xB_abv_percent"] = round(
                        float(get_abv_from_mol_frac(float(result["xB"]))),
                        6,
                    )
            except (ValueError, TypeError, ArithmeticError, ZeroDivisionError) as exc:
                row["xB_abv_percent"] = None
                row_warnings.append(f"Bottoms ABV conversion failed for one candidate row: {exc}")

            try:
                if result.get("D") is not None and result.get("xDavg") is not None:
                    row["D_volume_L"] = round(
                        float(get_mixture_volume_L_from_moles(float(result["D"]), float(result["xDavg"]))),
                        6,
                    )
                    row["xDavg_abv_percent"] = round(
                        float(get_abv_from_mol_frac(float(result["xDavg"]))),
                        6,
                    )
                else:
                    row["D_volume_L"] = None
                    row["xDavg_abv_percent"] = None
            except (ValueError, TypeError, ArithmeticError, ZeroDivisionError) as exc:
                row["D_volume_L"] = None
                row["xDavg_abv_percent"] = None
                row_warnings.append(f"Product conversion failed for one candidate row: {exc}")

            rows.append(row)

        warnings.extend(dict.fromkeys(row_warnings))
        warnings.append("x0, xB, and xDavg are reported as ethanol mole fractions.")
        execution_parameters["row_count"] = len(rows)
        columns = [
            column
            for column in [
                "W0",
                "W0_volume_L",
                "x0",
                "x0_abv_percent",
                "xB",
                "xB_abv_percent",
                "D",
                "D_volume_L",
                "xDavg",
                "xDavg_abv_percent",
                "error",
            ]
            if any(row.get(column) is not None for row in rows)
        ]
    except (ValueError, TypeError, ArithmeticError, ZeroDivisionError) as exc:
        return WorkflowExecutionResult(
            workflow_name=plan.workflow_name,
            success=False,
            message="Workflow execution failed during deterministic engineering calculation.",
            warnings=[str(exc)],
            normalized_inputs=normalized_inputs,
            execution_parameters=execution_parameters,
        )

    message = (
        f"Executed product_to_feed_sweep for D={D_target:.6f} mol and xDavg={xDavg_target:.6f}. "
        f"Generated {len(rows)} candidate feed rows."
    )
    return WorkflowExecutionResult(
        workflow_name=plan.workflow_name,
        success=True,
        message=message,
        columns=columns,
        rows=rows,
        warnings=warnings,
        normalized_inputs=normalized_inputs,
        execution_parameters=execution_parameters,
    )


def execute_solve_mole_balance(
    classification: GoalClassification,
    plan: WorkflowPlan,
) -> WorkflowExecutionResult:
    warnings: list[str] = [
        "This calculation uses only the overall mole balance and ethanol balance; it does not enforce the Rayleigh batch-distillation relationship."
    ]
    execution_parameters: dict[str, ExecutionParameterValue] = {
        "workflow": "solve_mole_balance",
        "equations": [
            "W0 = D + B",
            "W0*x0 = D*xDavg + B*xB",
        ],
    }
    normalized_inputs: dict[str, ScalarValue] = {}

    try:
        feed_known = None
        product_known = None
        bottoms_known = None

        try:
            W0, x0, feed_inputs, feed_warnings = _normalize_feed_inputs(classification)
            feed_known = (W0, x0)
            normalized_inputs = _merge_normalized_inputs(normalized_inputs, feed_inputs)
            warnings.extend(feed_warnings)
        except ValueError:
            pass

        try:
            D_value, xDavg_value, product_inputs, product_warnings = _normalize_product_inputs(classification)
            product_known = (D_value, xDavg_value)
            normalized_inputs = _merge_normalized_inputs(normalized_inputs, product_inputs)
            warnings.extend(product_warnings)
        except ValueError:
            pass

        try:
            B_value, xB_value, bottoms_inputs, bottoms_warnings = _normalize_bottoms_inputs(classification)
            bottoms_known = (B_value, xB_value)
            normalized_inputs = _merge_normalized_inputs(normalized_inputs, bottoms_inputs)
            warnings.extend(bottoms_warnings)
        except ValueError:
            pass

        values: dict[str, float] = {}
        if feed_known is not None:
            values["W0"], values["x0"] = feed_known
        if product_known is not None:
            values["D"], values["xDavg"] = product_known
        if bottoms_known is not None:
            values["B"], values["xB"] = bottoms_known

        requested = set(classification.requested_outputs)

        if {"W0", "x0", "D", "xDavg"}.issubset(values):
            values["B"] = get_B_mole_balance(values["W0"], values["D"])
            values["xB"] = get_xB_mole_balance(values["W0"], values["B"], values["D"], values["x0"], values["xDavg"])
            execution_parameters["solve_case"] = "A"
        elif {"W0", "x0", "B", "xB"}.issubset(values):
            values["D"] = get_D_mole_balance(values["W0"], values["B"])
            values["xDavg"] = get_xDavg_mole_balance_1(values["W0"], values["B"], values["D"], values["x0"], values["xB"])
            execution_parameters["solve_case"] = "B"
        elif {"D", "xDavg", "B", "xB"}.issubset(values):
            values["W0"] = get_W0_mole_balance(values["B"], values["D"])
            values["x0"] = get_x0_mole_balance(values["W0"], values["B"], values["D"], values["xB"], values["xDavg"])
            execution_parameters["solve_case"] = "C"
        elif {"W0", "D", "xDavg", "xB"}.issubset(values):
            values["B"] = get_B_mole_balance(values["W0"], values["D"])
            values["x0"] = get_x0_mole_balance(values["W0"], values["B"], values["D"], values["xB"], values["xDavg"])
            execution_parameters["solve_case"] = "D"
        elif {"W0", "x0", "D", "xB"}.issubset(values):
            values["B"] = get_B_mole_balance(values["W0"], values["D"])
            values["xDavg"] = get_xDavg_mole_balance_1(values["W0"], values["B"], values["D"], values["x0"], values["xB"])
            execution_parameters["solve_case"] = "E"
        else:
            return WorkflowExecutionResult(
                workflow_name=plan.workflow_name,
                success=False,
                message="This mole-balance variable combination is not implemented yet.",
                warnings=warnings,
                normalized_inputs=normalized_inputs,
                execution_parameters=execution_parameters,
            )
    except (ValueError, TypeError, ArithmeticError, ZeroDivisionError) as exc:
        return WorkflowExecutionResult(
            workflow_name=plan.workflow_name,
            success=False,
            message="Workflow execution failed during deterministic engineering calculation.",
            warnings=warnings + [str(exc)],
            normalized_inputs=normalized_inputs,
            execution_parameters=execution_parameters,
        )

    row: dict[str, ScalarValue] = {
        key: round(float(values[key]), 6)
        for key in ["W0", "x0", "D", "xDavg", "B", "xB"]
        if key in values
    }
    _populate_user_friendly_balance_columns(row, warnings)

    columns = [
        column
        for column in [
            "W0",
            "W0_volume_L",
            "x0",
            "x0_abv_percent",
            "D",
            "D_volume_L",
            "xDavg",
            "xDavg_abv_percent",
            "B",
            "B_volume_L",
            "xB",
            "xB_abv_percent",
        ]
        if row.get(column) is not None
    ]
    execution_parameters["requested_outputs"] = list(requested)

    return WorkflowExecutionResult(
        workflow_name=plan.workflow_name,
        success=True,
        message="Solved mole-balance variables from the provided batch values.",
        columns=columns,
        rows=[row],
        warnings=list(dict.fromkeys(warnings)),
        normalized_inputs=normalized_inputs,
        execution_parameters=execution_parameters,
    )


def execute_solve_rayleigh_batch_variables(
    classification: GoalClassification,
    plan: WorkflowPlan,
) -> WorkflowExecutionResult:
    warnings: list[str] = [
        "This result is constrained by the Rayleigh equation and the total/ethanol mole balances."
    ]
    execution_parameters: dict[str, ExecutionParameterValue] = {
        "workflow": "solve_rayleigh_batch_variables",
        "n": DEFAULT_SIMPSON_N,
        "requested_outputs": list(classification.requested_outputs),
    }
    normalized_inputs: dict[str, ScalarValue] = {}

    try:
        values: dict[str, float] = {}
        provided_xDavg: float | None = None

        try:
            W0, x0, feed_inputs, feed_warnings = _normalize_feed_inputs(classification)
            values["W0"] = W0
            values["x0"] = x0
            normalized_inputs = _merge_normalized_inputs(normalized_inputs, feed_inputs)
            warnings.extend(feed_warnings)
        except ValueError:
            pass

        try:
            D_value, xDavg_value, product_inputs, product_warnings = _normalize_product_inputs(classification)
            values["D"] = D_value
            values["xDavg"] = xDavg_value
            normalized_inputs = _merge_normalized_inputs(normalized_inputs, product_inputs)
            warnings.extend(product_warnings)
        except ValueError:
            pass

        try:
            B_value, xB_value, bottoms_inputs, bottoms_warnings = _normalize_bottoms_inputs(classification)
            values["B"] = B_value
            values["xB"] = xB_value
            normalized_inputs = _merge_normalized_inputs(normalized_inputs, bottoms_inputs)
            warnings.extend(bottoms_warnings)
        except ValueError:
            pass

        abv_only_values = _normalize_abv_only_inputs(classification)
        for key, value in abv_only_values.items():
            values.setdefault(key, value)
            normalized_inputs.setdefault(key, value)

        assignments = classification.variable_assignments
        for key in ["W0", "D", "B"]:
            if key in assignments and key not in values and not _has_volume_units(assignments[key]):
                values[key] = _parse_model_scalar(assignments[key], key)
                normalized_inputs[key] = values[key]
        for key in ["x0", "xDavg", "xB"]:
            if key in assignments and key not in values:
                values[key] = _parse_composition_value(assignments[key], key)
                normalized_inputs[key] = values[key]
        if values and "input_source" not in normalized_inputs:
            normalized_inputs["input_source"] = "model_units"

        solve_case = None

        if {"W0", "x0", "xB"}.issubset(values):
            solve_case = "A"
            if "xDavg" in values:
                provided_xDavg = values["xDavg"]
            integral = composite_simpson_rule_rayleigh(values["x0"], values["xB"], n=DEFAULT_SIMPSON_N)
            B_over_W0 = math.exp(-integral)
            values["B"] = get_B_rayleigh(values["W0"], values["x0"], values["xB"], n=DEFAULT_SIMPSON_N)
            values["D"] = get_D_mole_balance(values["W0"], values["B"])
            values["xDavg"] = get_xDavg_mole_balance_1(values["W0"], values["B"], values["D"], values["x0"], values["xB"])
            execution_parameters["rayleigh_integral"] = integral
            execution_parameters["B_over_W0"] = B_over_W0
            execution_parameters["W_over_W0"] = B_over_W0
            if provided_xDavg is not None:
                xDavg_error = abs(provided_xDavg - values["xDavg"])
                execution_parameters["provided_xDavg"] = provided_xDavg
                execution_parameters["implied_xDavg"] = values["xDavg"]
                execution_parameters["xDavg_error"] = xDavg_error
                if xDavg_error > 1e-3:
                    warnings.append(
                        "Provided xDavg does not match the Rayleigh-implied average distillate composition. "
                        f"The Rayleigh solution implies xDavg = {values['xDavg']:.6f}, while the provided value was {provided_xDavg:.6f}."
                    )
        elif {"W0", "x0", "D"}.issubset(values):
            solve_case = "B"
            values["B"] = get_B_mole_balance(values["W0"], values["D"])
            target_integral = math.log(values["W0"] / values["B"])

            def func_case_b(xb: float) -> float:
                return composite_simpson_rule_rayleigh(values["x0"], xb, n=DEFAULT_SIMPSON_N) - target_integral

            values["xB"] = _solve_root_brentq(func_case_b, DEFAULT_ROOT_EPS, values["x0"] - DEFAULT_ROOT_EPS, "xB")
            values["xDavg"] = get_xDavg_mole_balance_1(values["W0"], values["B"], values["D"], values["x0"], values["xB"])
            execution_parameters["target_integral"] = target_integral
            execution_parameters["root_solver"] = "scipy.optimize.brentq"
            warnings.append("Root-finding was used to solve the unknown composition.")
        elif {"D", "xDavg", "x0"}.issubset(values):
            solve_case = "C"

            def func_case_c(xb: float) -> float:
                r_from_balance = (values["x0"] - values["xDavg"]) / (xb - values["xDavg"])
                if not (0.0 < r_from_balance < 1.0):
                    raise ValueError("No valid Rayleigh ratio exists for the requested D, xDavg, and x0.")
                return math.log(1.0 / r_from_balance) - composite_simpson_rule_rayleigh(values["x0"], xb, n=DEFAULT_SIMPSON_N)

            values["xB"] = _solve_root_brentq(func_case_c, DEFAULT_ROOT_EPS, values["x0"] - DEFAULT_ROOT_EPS, "xB")
            r_value = (values["x0"] - values["xDavg"]) / (values["xB"] - values["xDavg"])
            if not (0.0 < r_value < 1.0):
                raise ValueError("No valid Rayleigh ratio exists for this D, xDavg, and x0 combination.")
            values["W0"] = values["D"] / (1.0 - r_value)
            values["B"] = values["W0"] * r_value
            execution_parameters["B_over_W0"] = r_value
            execution_parameters["rayleigh_integral"] = composite_simpson_rule_rayleigh(values["x0"], values["xB"], n=DEFAULT_SIMPSON_N)
            execution_parameters["root_solver"] = "scipy.optimize.brentq"
            warnings.append("Root-finding was used to solve the unknown composition.")
        elif {"D", "xDavg", "xB"}.issubset(values):
            solve_case = "D"

            def func_case_d(x0_candidate: float) -> float:
                integral = composite_simpson_rule_rayleigh(x0_candidate, values["xB"], n=DEFAULT_SIMPSON_N)
                r_value = math.exp(-integral)
                if not (0.0 < r_value < 1.0):
                    raise ValueError("No valid Rayleigh ratio exists for the requested D, xDavg, and xB.")
                W0_candidate = values["D"] / (1.0 - r_value)
                B_candidate = W0_candidate * r_value
                xDavg_calc = get_xDavg_mole_balance_1(W0_candidate, B_candidate, values["D"], x0_candidate, values["xB"])
                return xDavg_calc - values["xDavg"]

            values["x0"] = _solve_root_brentq(func_case_d, values["xB"] + DEFAULT_ROOT_EPS, DEFAULT_RAYLEIGH_UPPER_X, "x0")
            integral = composite_simpson_rule_rayleigh(values["x0"], values["xB"], n=DEFAULT_SIMPSON_N)
            r_value = math.exp(-integral)
            values["W0"] = values["D"] / (1.0 - r_value)
            values["B"] = values["W0"] * r_value
            execution_parameters["B_over_W0"] = r_value
            execution_parameters["rayleigh_integral"] = integral
            execution_parameters["root_solver"] = "scipy.optimize.brentq"
            warnings.append("Root-finding was used to solve the unknown composition.")
        else:
            known_inputs = sorted(values.keys())
            return WorkflowExecutionResult(
                workflow_name=plan.workflow_name,
                success=False,
                message=(
                    f"Rayleigh solve case not implemented for known inputs: {known_inputs}. "
                    "Currently supported direct cases are W0 + x0 + xB -> B, D, implied xDavg; "
                    "W0 + x0 + D -> B, xB, xDavg; D + xDavg + x0 -> W0, B, xB; "
                    "and D + xDavg + xB -> W0, B, x0."
                ),
                warnings=warnings,
                normalized_inputs=normalized_inputs,
                execution_parameters=execution_parameters,
            )
    except (ValueError, TypeError, ArithmeticError, ZeroDivisionError) as exc:
        return WorkflowExecutionResult(
            workflow_name=plan.workflow_name,
            success=False,
            message="Workflow execution failed during deterministic engineering calculation.",
            warnings=warnings + [str(exc)],
            normalized_inputs=normalized_inputs,
            execution_parameters=execution_parameters,
        )

    execution_parameters["solve_case"] = solve_case
    row: dict[str, ScalarValue] = {
        key: round(float(values[key]), 6)
        for key in ["W0", "x0", "D", "xDavg", "B", "xB"]
        if key in values
    }
    if provided_xDavg is not None:
        row["provided_xDavg"] = round(float(provided_xDavg), 6)
        row["xDavg_implied"] = round(float(values["xDavg"]), 6)
        row["xDavg_error"] = round(float(abs(provided_xDavg - values["xDavg"])), 6)
    _populate_user_friendly_balance_columns(row, warnings)

    columns = [
        column
        for column in [
            "W0",
            "W0_volume_L",
            "x0",
            "x0_abv_percent",
            "D",
            "D_volume_L",
            "xDavg",
            "xDavg_implied",
            "provided_xDavg",
            "xDavg_error",
            "xDavg_abv_percent",
            "B",
            "B_volume_L",
            "xB",
            "xB_abv_percent",
        ]
        if row.get(column) is not None
    ]

    return WorkflowExecutionResult(
        workflow_name=plan.workflow_name,
        success=True,
        message="Solved Rayleigh-constrained batch variables from the provided values.",
        columns=columns,
        rows=[row],
        warnings=list(dict.fromkeys(warnings)),
        normalized_inputs=normalized_inputs,
        execution_parameters=execution_parameters,
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

    if plan.workflow_name == "feed_to_product_sweep":
        return execute_feed_to_product_sweep(classification, plan)

    if plan.workflow_name == "product_to_feed_sweep":
        return execute_product_to_feed_sweep(classification, plan)

    if plan.workflow_name == "solve_mole_balance":
        return execute_solve_mole_balance(classification, plan)

    if plan.workflow_name == "solve_rayleigh_batch_variables":
        return execute_solve_rayleigh_batch_variables(classification, plan)

    if plan.workflow_name not in {"feed_to_product_sweep", "product_to_feed_sweep", "solve_mole_balance", "solve_rayleigh_batch_variables"}:
        return WorkflowExecutionResult(
            workflow_name=plan.workflow_name,
            success=False,
            message="This workflow is planned but execution is not implemented yet.",
        )
    return WorkflowExecutionResult(
        workflow_name=plan.workflow_name,
        success=False,
        message="This workflow is planned but execution is not implemented yet.",
    )
