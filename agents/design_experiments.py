from agents.workflows import VARIABLE_DESCRIPTIONS, add_product_metrics
from engineering.tools import (
    prototype_design_given_D_xDavg,
    solve_batch_given_W0_x0_xB,
    solve_D_given_W0_x0_xDavg,
)


def _goal_mentions_options(user_goal: str) -> bool:
    goal = (user_goal or "").lower()
    option_phrases = (
        "option",
        "options",
        "compare",
        "different",
        "help choose",
        "choose",
        "explore",
        "sample",
    )
    return any(phrase in goal for phrase in option_phrases)


def _sample_xdavg_targets(x0: float | None) -> list[float]:
    values = [0.10, 0.15, 0.20]
    if x0 is None:
        return values
    return [value for value in values if value > x0]


def _sample_xb_values(x0: float | None) -> list[float]:
    values = [0.0025, 0.005, 0.01, 0.02]
    if x0 is None:
        return values
    return [value for value in values if value < x0]


def _sample_x0_values(xB: float | None) -> list[float]:
    values = [0.03, 0.05, 0.10, 0.15]
    if xB is None:
        return values
    return [value for value in values if value > xB]


def build_sample_possible_plan(
    knowns: dict[str, float],
    sampled_variable: str,
    sample_values: list[float],
    reason: str,
    next_question: str,
) -> dict:
    return {
        "status": "sample_possible",
        "knowns": dict(knowns),
        "candidate_sampling_variables": [sampled_variable],
        "recommended_sampling_variable": sampled_variable,
        "sample_values": {sampled_variable: sample_values[:5]},
        "sample_values_are_illustrative": True,
        "reason": reason,
        "next_question": next_question,
        "recommended_workflow": None,
    }


def shifted_sample_values(
    sampled_variable: str,
    current_values: list[float],
    knowns: dict[str, float],
    direction: str,
) -> list[float]:
    candidate_pools = {
        "xB": [0.0025, 0.0050, 0.0070, 0.0100, 0.0150, 0.0200, 0.0300, 0.0400],
        "x0": [0.03, 0.05, 0.07, 0.10, 0.15, 0.20],
        "xDavg_target": [0.10, 0.12, 0.15, 0.18, 0.20, 0.25],
    }
    pool = candidate_pools.get(sampled_variable, [])
    if not pool or not current_values:
        return []

    if direction == "higher":
        threshold = max(current_values)
        values = [value for value in pool if value > threshold]
    else:
        threshold = min(current_values)
        values = [value for value in pool if value < threshold]

    if sampled_variable == "xB" and "x0" in knowns:
        values = [value for value in values if value < knowns["x0"]]
    if sampled_variable == "x0" and "xB" in knowns:
        values = [value for value in values if value > knowns["xB"]]
    if sampled_variable == "xDavg_target" and "x0" in knowns:
        values = [value for value in values if value > knowns["x0"]]

    if direction == "higher":
        return values[:3]
    return values[-3:]


def plan_experiment_from_knowns(
    knowns: dict[str, float],
    requested_outputs: list[str] | None = None,
    user_goal: str | None = None,
) -> dict:
    requested_outputs = requested_outputs or []
    goal_requests_options = _goal_mentions_options(user_goal or "")

    result = {
        "status": "unsupported",
        "knowns": dict(knowns),
        "candidate_sampling_variables": [],
        "recommended_sampling_variable": None,
        "sample_values": {},
        "sample_values_are_illustrative": True,
        "reason": "",
        "next_question": "",
        "recommended_workflow": None,
    }

    if not knowns:
        result.update(
            {
                "status": "needs_more_information",
                "reason": (
                    "I do not have enough starting information yet to choose a supported calculation path."
                ),
                "next_question": (
                    "Do you know the initial charge amount (W0) and the initial ethanol mole fraction (x0), or what values do you already know?"
                ),
            }
        )
        return result

    if {"W0", "x0", "xDavg_target"}.issubset(knowns):
        result.update(
            {
                "status": "ready_to_calculate",
                "reason": "The target-average-distillate workflow is ready to run.",
                "recommended_workflow": "solve_D_given_W0_x0_xDavg",
                "next_question": (
                    "I have enough information to calculate the distillate amount (D), average distillate ethanol mole fraction (xDavg), and distillate/feed ratio (D/W0)."
                ),
            }
        )
        return result

    if {"W0", "x0", "xB"}.issubset(knowns):
        result.update(
            {
                "status": "ready_to_calculate",
                "reason": "The stopping-basis workflow is ready to run.",
                "recommended_workflow": "solve_batch_given_W0_x0_xB",
                "next_question": (
                    "I have enough information to calculate the distillate amount (D), average distillate ethanol mole fraction (xDavg), and distillate/feed ratio (D/W0)."
                ),
            }
        )
        return result

    if {"W0", "B", "D", "x0", "xB", "xDavg"}.issubset(knowns):
        result.update(
            {
                "status": "ready_to_calculate",
                "reason": "The consistency-check workflow is ready to run.",
                "recommended_workflow": "check_batch_consistency",
                "next_question": (
                    "I have enough information to check the total balance, ethanol component balance, and Rayleigh consistency."
                ),
            }
        )
        return result

    if {"W0", "x0"}.issubset(knowns):
        avg_targets = _sample_xdavg_targets(knowns["x0"])
        xb_values = _sample_xb_values(knowns["x0"])
        if goal_requests_options:
            result.update(
                {
                    "status": "sample_possible",
                    "candidate_sampling_variables": ["xDavg_target", "xB"],
                    "recommended_sampling_variable": "xDavg_target",
                    "sample_values": {
                        "xDavg_target": avg_targets,
                        "xB": xb_values,
                    },
                    "reason": (
                        "With the initial charge amount (W0) and initial ethanol mole fraction (x0) known, the most product-centered next step is to compare target average distillate compositions. I can also explore alternate stopping-basis assumptions using final still ethanol mole fraction (xB)."
                    ),
                    "next_question": (
                        "I can show compact examples for target average distillate composition (xDavg_target). If you want, I can also explore stopping-basis assumptions using final still ethanol mole fraction (xB)."
                    ),
                }
            )
        else:
            result.update(
                {
                    "status": "choose_sampling_axis",
                    "candidate_sampling_variables": ["xDavg_target", "xB"],
                    "sample_values": {
                        "xDavg_target": avg_targets,
                        "xB": xb_values,
                    },
                    "reason": (
                        "With the initial charge amount (W0) and initial ethanol mole fraction (x0) known, the most product-centered next step is to compare target average distillate compositions. A final still ethanol mole fraction (xB) can also be used as a stopping basis."
                    ),
                    "next_question": (
                        "I can start by comparing target average distillate compositions (xDavg_target). If you prefer, I can instead explore a stopping basis using final still ethanol mole fraction (xB)."
                    ),
                }
            )
        return result

    if "D" in knowns and "xDavg_target" in knowns and "x0" in knowns and "W0" not in knowns:
        xb_values = _sample_xb_values(knowns["x0"])
        result.update(
            {
                "status": "sample_possible",
                "candidate_sampling_variables": ["xB"],
                "recommended_sampling_variable": "xB",
                "sample_values": {"xB": xb_values},
                "reason": (
                    "You have defined the product goal, but the required initial charge amount still depends on the feed composition and a stopping assumption. Sampling final still ethanol mole fraction (xB) can show possible feed requirements."
                ),
                "next_question": (
                    "I can explore example stopping-basis values using final still ethanol mole fraction (xB) to show how the required initial charge amount could change. Do you want to sample xB?"
                ),
            }
        )
        return result

    if "D" in knowns and "xDavg_target" in knowns and "xB" in knowns and "x0" not in knowns:
        x0_values = _sample_x0_values(knowns["xB"])
        result.update(
            {
                "status": "sample_possible",
                "candidate_sampling_variables": ["x0"],
                "recommended_sampling_variable": "x0",
                "sample_values": {"x0": x0_values},
                "reason": (
                    "You have defined the product goal and a stopping basis, but the required initial charge amount still depends on feed composition. Sampling initial ethanol mole fraction (x0) can show possible feed requirements."
                ),
                "next_question": (
                    "I can explore example initial ethanol mole fraction (x0) values to show how the required initial charge amount could change. Do you want to sample x0?"
                ),
            }
        )
        return result

    if "D" in knowns and "xDavg_target" in knowns and "W0" not in knowns and "x0" not in knowns:
        result.update(
            {
                "status": "choose_sampling_axis",
                "candidate_sampling_variables": ["x0", "xB"],
                "sample_values": {
                    "x0": _sample_x0_values(knowns.get("xB")),
                    "xB": _sample_xb_values(knowns.get("x0")),
                },
                "reason": (
                    "You have defined the product goal, but the feed requirement is still not unique because it depends on feed composition and a stopping basis."
                ),
                "next_question": (
                    "To explore possible feed requirements, choose one variable for me to sample: feed composition (x0) or a stopping basis using final still ethanol mole fraction (xB)."
                ),
            }
        )
        return result

    if "W0" in knowns and "x0" not in knowns:
        result.update(
            {
                "status": "needs_more_information",
                "reason": (
                    "The initial charge amount (W0) alone is not enough to choose a supported calculation path."
                ),
                "next_question": (
                    "Do you know the initial ethanol mole fraction (x0), and do you want to target average distillate composition (xDavg_target) or final still ethanol mole fraction (xB)?"
                ),
            }
        )
        return result

    if "x0" in knowns and "W0" not in knowns:
        result.update(
            {
                "status": "needs_more_information",
                "reason": (
                    "The initial ethanol mole fraction (x0) alone is not enough to choose a supported calculation path."
                ),
                "next_question": (
                    "Do you know the initial charge amount (W0), and do you want to target average distillate composition (xDavg_target) or final still ethanol mole fraction (xB)?"
                ),
            }
        )
        return result

    if requested_outputs:
        described_outputs = ", ".join(
            VARIABLE_DESCRIPTIONS.get(name, name) for name in requested_outputs
        )
        result.update(
            {
                "status": "needs_more_information",
                "reason": (
                    f"I still need more information before I can determine {described_outputs}."
                ),
                "next_question": (
                    "Tell me which starting values or targets you already know, and I can help choose the next design basis."
                ),
            }
        )
        return result

    result.update(
        {
            "status": "unsupported",
            "reason": (
                "I do not yet have enough matching inputs to plan a supported batch-distillation calculation."
            ),
            "next_question": (
                "Tell me which of these you know: initial charge amount (W0), initial ethanol mole fraction (x0), target average distillate ethanol mole fraction (xDavg_target), or final still ethanol mole fraction (xB)."
            ),
        }
    )
    return result


def run_planned_scenarios(
    knowns: dict[str, float],
    plan: dict,
    n: int = 100,
) -> dict:
    scenario_result = {
        "status": "no_scenarios",
        "sampled_variable": plan.get("recommended_sampling_variable"),
        "rows": [],
        "notes": [],
        "is_illustrative_only": True,
    }

    if plan.get("status") != "sample_possible":
        scenario_result["notes"].append(
            "Scenario sampling is only available when the current plan supports a single illustrative sampling path."
        )
        return scenario_result

    recommended_variable = plan.get("recommended_sampling_variable")
    sample_values = plan.get("sample_values", {})

    if {"D", "xDavg_target", "x0"}.issubset(knowns) and recommended_variable == "xB":
        prototype = prototype_design_given_D_xDavg(
            D_target=knowns["D"],
            xDavg_target=knowns["xDavg_target"],
            x0_options=[knowns["x0"]],
            xB_options=sample_values.get("xB"),
            n=n,
        )
        for scenario in prototype["scenarios"][:5]:
            scenario_result["rows"].append(
                add_product_metrics(
                    {
                    "sampled_variable": "xB",
                    "sampled_value": scenario["xB"],
                    "W0": scenario["W0"],
                    "B": scenario["B"],
                    "D": scenario["D"],
                    "x0": scenario["x0"],
                    "xB": scenario["xB"],
                    "xDavg": knowns["xDavg_target"],
                    "is_fully_consistent": scenario.get("is_fully_consistent"),
                    "rayleigh_error": scenario.get("rayleigh_error"),
                    "status": "consistent" if scenario.get("is_fully_consistent") else "inconsistent",
                    "note": "",
                    }
                )
            )
        scenario_result["notes"].extend(prototype.get("notes", []))
        scenario_result["status"] = "ok" if scenario_result["rows"] else "no_scenarios"
        return scenario_result

    if {"D", "xDavg_target", "xB"}.issubset(knowns) and recommended_variable == "x0":
        prototype = prototype_design_given_D_xDavg(
            D_target=knowns["D"],
            xDavg_target=knowns["xDavg_target"],
            x0_options=sample_values.get("x0"),
            xB_options=[knowns["xB"]],
            n=n,
        )
        for scenario in prototype["scenarios"][:5]:
            scenario_result["rows"].append(
                add_product_metrics(
                    {
                    "sampled_variable": "x0",
                    "sampled_value": scenario["x0"],
                    "W0": scenario["W0"],
                    "B": scenario["B"],
                    "D": scenario["D"],
                    "x0": scenario["x0"],
                    "xB": scenario["xB"],
                    "xDavg": knowns["xDavg_target"],
                    "is_fully_consistent": scenario.get("is_fully_consistent"),
                    "rayleigh_error": scenario.get("rayleigh_error"),
                    "status": "consistent" if scenario.get("is_fully_consistent") else "inconsistent",
                    "note": "",
                    }
                )
            )
        scenario_result["notes"].extend(prototype.get("notes", []))
        scenario_result["status"] = "ok" if scenario_result["rows"] else "no_scenarios"
        return scenario_result

    if {"W0", "x0"}.issubset(knowns):
        if recommended_variable in {"both", "xDavg_target", None}:
            for xDavg_target in sample_values.get("xDavg_target", [])[:3]:
                try:
                    result = solve_D_given_W0_x0_xDavg(
                        W0=knowns["W0"],
                        x0=knowns["x0"],
                        xDavg_target=xDavg_target,
                        n=n,
                    )
                    scenario_result["rows"].append(
                        add_product_metrics(
                            {
                            "sampled_variable": "xDavg_target",
                            "sampled_value": xDavg_target,
                            "W0": result["W0"],
                            "D": result["D"],
                            "B": result["B"],
                            "xB": result["xB"],
                            "xDavg": result["xDavg"],
                            "status": "consistent",
                            "note": "",
                            }
                        )
                    )
                except Exception as exc:
                    scenario_result["rows"].append(
                        {
                            "sampled_variable": "xDavg_target",
                            "sampled_value": xDavg_target,
                            "status": "error",
                            "note": str(exc),
                        }
                    )

        if recommended_variable in {"both", "xB"}:
            for xB in sample_values.get("xB", [])[:3]:
                try:
                    result = solve_batch_given_W0_x0_xB(
                        W0=knowns["W0"],
                        x0=knowns["x0"],
                        xB=xB,
                        n=n,
                    )
                    scenario_result["rows"].append(
                        add_product_metrics(
                            {
                            "sampled_variable": "xB",
                            "sampled_value": xB,
                            "W0": result["W0"],
                            "D": result["D"],
                            "B": result["B"],
                            "xB": result["xB"],
                            "xDavg": result["xDavg"],
                            "status": "consistent",
                            "note": "",
                            }
                        )
                    )
                except Exception as exc:
                    scenario_result["rows"].append(
                        {
                            "sampled_variable": "xB",
                            "sampled_value": xB,
                            "status": "error",
                            "note": str(exc),
                        }
                    )

        scenario_result["status"] = "ok" if scenario_result["rows"] else "no_scenarios"
        return scenario_result

    scenario_result["notes"].append(
        "No compact deterministic sampling path is available for the current known inputs."
    )
    return scenario_result
