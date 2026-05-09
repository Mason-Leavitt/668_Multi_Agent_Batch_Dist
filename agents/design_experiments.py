from agents.workflows import VARIABLE_DESCRIPTIONS
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
                    "I have enough information to calculate the distillate amount (D), the final still amount (B), and the final still ethanol mole fraction (xB)."
                ),
            }
        )
        return result

    if {"W0", "x0", "xB"}.issubset(knowns):
        result.update(
            {
                "status": "ready_to_calculate",
                "reason": "The target-final-still-composition workflow is ready to run.",
                "recommended_workflow": "solve_batch_given_W0_x0_xB",
                "next_question": (
                    "I have enough information to calculate the distillate amount (D) and the average distillate ethanol mole fraction (xDavg)."
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
                    "recommended_sampling_variable": "both",
                    "sample_values": {
                        "xDavg_target": avg_targets,
                        "xB": xb_values,
                    },
                    "reason": (
                        "With the initial charge amount (W0) and initial ethanol mole fraction (x0) known, you can compare either target average distillate compositions or target final still compositions."
                    ),
                    "next_question": (
                        "I can show compact examples for both target average distillate composition (xDavg_target) and final still ethanol mole fraction (xB)."
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
                        "With the initial charge amount (W0) and initial ethanol mole fraction (x0) known, the next design choice is which target to explore."
                    ),
                    "next_question": (
                        "Do you want to explore target average distillate composition (xDavg_target) or final still ethanol mole fraction (xB)?"
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
                    "The distillate amount (D), target average distillate ethanol mole fraction (xDavg_target), and initial ethanol mole fraction (x0) still leave the required initial charge amount open. Sampling final still ethanol mole fraction (xB) can show possible setups."
                ),
                "next_question": (
                    "I can explore example final still ethanol mole fraction (xB) values to show how the required initial charge amount could change. Do you want to sample xB?"
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
                    "The distillate amount (D), target average distillate ethanol mole fraction (xDavg_target), and final still ethanol mole fraction (xB) still leave the feed composition open. Sampling initial ethanol mole fraction (x0) can show possible setups."
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
                    "The distillate amount (D) and target average distillate ethanol mole fraction (xDavg_target) do not uniquely determine both the starting charge and the feed composition."
                ),
                "next_question": (
                    "To explore possible setups, choose one variable for me to sample: feed composition (x0) or final still ethanol mole fraction (xB)."
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
                        {
                            "sampled_variable": "xDavg_target",
                            "sampled_value": xDavg_target,
                            "D": result["D"],
                            "B": result["B"],
                            "xB": result["xB"],
                            "xDavg": result["xDavg"],
                            "status": "consistent",
                            "note": "",
                        }
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
                        {
                            "sampled_variable": "xB",
                            "sampled_value": xB,
                            "D": result["D"],
                            "B": result["B"],
                            "xB": result["xB"],
                            "xDavg": result["xDavg"],
                            "status": "consistent",
                            "note": "",
                        }
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
