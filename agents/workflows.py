from engineering.tools import (
    solve_batch_given_W0_x0_xB,
    solve_D_given_W0_x0_xDavg,
)

SUPPORTED_WORKFLOWS = {
    "solve_D_given_W0_x0_xDavg": {
        "label": "Choose a target average distillate composition",
        "problem_type": "solve_D_given_W0_x0_xDavg",
        "description": (
            "Use this workflow when you know the initial charge amount (W0), the initial "
            "ethanol mole fraction (x0), and the target average distillate ethanol mole fraction (xDavg_target)."
        ),
        "required_inputs": ["W0", "x0", "xDavg_target"],
        "outputs": ["D", "B", "xB", "xDavg"],
        "example_user_prompt": (
            "I have 1000 mol of ethanol-water at 5 mol% ethanol. "
            "I want the average distillate to be 20 mol% ethanol. "
            "How much distillate can I collect?"
        ),
    },
    "solve_batch_given_W0_x0_xB": {
        "label": "Choose a target final still composition",
        "problem_type": "solve_batch_given_W0_x0_xB",
        "description": (
            "Use this workflow when you know the initial charge amount (W0), the initial "
            "ethanol mole fraction (x0), and the final still ethanol mole fraction (xB) you want to reach."
        ),
        "required_inputs": ["W0", "x0", "xB"],
        "outputs": ["D", "B", "xB", "xDavg"],
        "example_user_prompt": (
            "I start with 1000 mol of ethanol-water at 5 mol% ethanol and distill "
            "until the still is 1 mol% ethanol. How much distillate do I collect?"
        ),
    },
    "check_batch_consistency": {
        "label": "Check an existing proposed result",
        "problem_type": "check_batch_consistency",
        "description": (
            "Use this workflow when you already have a proposed batch result and "
            "want to check total balance, ethanol component balance, and Rayleigh consistency."
        ),
        "required_inputs": ["W0", "B", "D", "x0", "xB", "xDavg"],
        "outputs": [
            "is_total_balance_consistent",
            "is_component_balance_consistent",
            "is_rayleigh_consistent",
            "is_fully_consistent",
        ],
        "example_user_prompt": (
            "Check whether this batch result is consistent: W0=1000 mol, "
            "B=763.986 mol, D=236.014 mol, x0=0.05, xB=0.003661, and xDavg=0.20."
        ),
    },
}

VARIABLE_DESCRIPTIONS = {
    "W0": "initial charge amount (W0)",
    "B": "final still amount (B)",
    "D": "distillate amount (D)",
    "x0": "initial ethanol mole fraction (x0)",
    "xB": "final still ethanol mole fraction (xB)",
    "xDavg": "average distillate ethanol mole fraction (xDavg)",
    "xDavg_target": "target average distillate ethanol mole fraction (xDavg_target)",
}


def analyze_knowns_against_workflows(
    knowns: dict,
    requested_outputs: list | None = None,
) -> dict:
    requested_outputs = requested_outputs or []
    workflow_results = {}

    for problem_type, workflow in SUPPORTED_WORKFLOWS.items():
        required_inputs = workflow["required_inputs"]
        present_inputs = [name for name in required_inputs if name in knowns]
        missing_inputs = [name for name in required_inputs if name not in knowns]
        output_overlap = [name for name in requested_outputs if name in workflow["outputs"]]

        workflow_results[problem_type] = {
            "problem_type": problem_type,
            "label": workflow["label"],
            "description": workflow["description"],
            "required_inputs": required_inputs,
            "present_inputs": present_inputs,
            "missing_inputs": missing_inputs,
            "is_ready": len(missing_inputs) == 0,
            "outputs": workflow["outputs"],
            "output_overlap": output_overlap,
            "missing_count": len(missing_inputs),
        }

    ready_workflows = [
        item for item in workflow_results.values() if item["is_ready"]
    ]

    closest_workflows = sorted(
        workflow_results.values(),
        key=lambda item: (
            item["missing_count"],
            -len(item["output_overlap"]),
            item["label"],
        ),
    )

    recommendation = recommend_next_design_basis(
        knowns=knowns,
        requested_outputs=requested_outputs,
    )

    return {
        "workflow_analysis": workflow_results,
        "ready_workflows": ready_workflows,
        "closest_workflows": closest_workflows[:3],
        "recommended_next_question": recommendation["recommended_next_question"],
        "recommendation": recommendation,
    }


def recommend_next_design_basis(
    knowns: dict,
    requested_outputs: list | None = None,
    user_goal: str | None = None,
) -> dict:
    requested_outputs = requested_outputs or []
    user_goal = (user_goal or "").lower()

    if not knowns:
        return {
            "status": "needs_starting_inputs",
            "recommended_next_question": (
                "Do you know the initial charge amount (W0) and the initial ethanol mole fraction (x0)?"
            ),
            "explanation": (
                "With no starting values yet, the most useful first design basis is the "
                "initial charge amount (W0) together with the initial ethanol mole fraction (x0)."
            ),
            "next_inputs_options": ["W0", "x0"],
        }

    if {"W0", "x0", "xDavg_target"}.issubset(knowns):
        return {
            "status": "ready_to_calculate",
            "recommended_next_question": (
                "I have enough information to run the target-average-distillate calculation. "
                "Would you like me to calculate the distillate amount (D), the final still amount (B), "
                "and the final still ethanol mole fraction (xB)?"
            ),
            "explanation": (
                "The supported target-average-distillate workflow is ready."
            ),
            "next_inputs_options": [],
        }

    if {"W0", "x0", "xB"}.issubset(knowns):
        return {
            "status": "ready_to_calculate",
            "recommended_next_question": (
                "I have enough information to run the target-final-still-composition calculation. "
                "Would you like me to calculate the distillate amount (D) and the average distillate ethanol mole fraction (xDavg)?"
            ),
            "explanation": "The supported target-final-still workflow is ready.",
            "next_inputs_options": [],
        }

    if {"W0", "B", "D", "x0", "xB", "xDavg"}.issubset(knowns):
        return {
            "status": "ready_to_calculate",
            "recommended_next_question": (
                "I have enough information to run the consistency check. "
                "Would you like me to verify the balances and Rayleigh consistency?"
            ),
            "explanation": "The supported consistency-check workflow is ready.",
            "next_inputs_options": [],
        }

    if "D" in knowns and "xDavg_target" in knowns and "W0" not in knowns and "x0" not in knowns:
        return {
            "status": "design_prototyping",
            "recommended_next_question": (
                "Do you know either the initial ethanol mole fraction (x0) of the feed or the final still ethanol mole fraction (xB) you want to reach?"
            ),
            "explanation": (
                "The distillate amount (D) and target average distillate ethanol mole fraction (xDavg_target) alone do not uniquely determine the required initial charge amount or feed composition, so one more design basis is needed."
            ),
            "next_inputs_options": ["x0", "xB"],
        }

    if {"W0", "x0"}.issubset(knowns) and "xDavg_target" not in knowns and "xB" not in knowns:
        return {
            "status": "choose_target",
            "recommended_next_question": (
                "Do you want to target the average distillate ethanol mole fraction (xDavg_target) or the final still ethanol mole fraction (xB)?"
            ),
            "explanation": (
                "With the initial charge amount (W0) and initial ethanol mole fraction (x0) known, the next useful design choice is which target to specify."
            ),
            "next_inputs_options": ["xDavg_target", "xB"],
        }

    if "W0" in knowns and "x0" not in knowns:
        return {
            "status": "needs_feed_composition",
            "recommended_next_question": (
                "Do you know the initial ethanol mole fraction (x0) of the feed, and do you want to target the average distillate ethanol mole fraction (xDavg_target) or the final still ethanol mole fraction (xB)?"
            ),
            "explanation": (
                "An initial charge amount alone is not enough to run a supported batch-distillation calculation."
            ),
            "next_inputs_options": ["x0", "xDavg_target", "xB"],
        }

    if "x0" in knowns and "W0" not in knowns:
        return {
            "status": "needs_charge_amount",
            "recommended_next_question": (
                "Do you know the initial charge amount (W0) in the still, and do you want to target the average distillate ethanol mole fraction (xDavg_target) or the final still ethanol mole fraction (xB)?"
            ),
            "explanation": (
                "The initial feed composition alone is not enough to run a supported batch-distillation calculation."
            ),
            "next_inputs_options": ["W0", "xDavg_target", "xB"],
        }

    if user_goal and "start" in user_goal:
        return {
            "status": "needs_starting_inputs",
            "recommended_next_question": (
                "Do you know the initial charge amount (W0) and the initial ethanol mole fraction (x0)?"
            ),
            "explanation": (
                "Those two inputs are the most useful starting basis for the supported workflows."
            ),
            "next_inputs_options": ["W0", "x0"],
        }

    closest_workflow = min(
        SUPPORTED_WORKFLOWS.values(),
        key=lambda workflow: len([name for name in workflow["required_inputs"] if name not in knowns]),
    )
    missing_inputs = [name for name in closest_workflow["required_inputs"] if name not in knowns]
    missing_text = ", ".join(VARIABLE_DESCRIPTIONS.get(name, name) for name in missing_inputs)

    return {
        "status": "needs_more_inputs",
        "recommended_next_question": (
            f"To use the {closest_workflow['label'].lower()} workflow, do you know {missing_text}?"
        ),
        "explanation": (
            "The closest supported workflow is not ready yet, but it can be completed with a few more inputs."
        ),
        "next_inputs_options": missing_inputs,
    }


def prototype_supported_scenarios(knowns: dict) -> dict:
    scenarios = {
        "avg_distillate_scenarios": [],
        "final_still_scenarios": [],
        "notes": [],
        "is_illustrative_only": True,
    }

    W0 = knowns.get("W0")
    x0 = knowns.get("x0")

    if W0 is not None and x0 is not None:
        for xDavg_target in [0.10, 0.15, 0.20]:
            if xDavg_target <= x0:
                continue
            try:
                result = solve_D_given_W0_x0_xDavg(
                    W0=W0,
                    x0=x0,
                    xDavg_target=xDavg_target,
                    n=100,
                )
                scenarios["avg_distillate_scenarios"].append(
                    {
                        "xDavg_target": xDavg_target,
                        "D": result["D"],
                        "B": result["B"],
                        "xB": result["xB"],
                    }
                )
            except Exception as exc:
                scenarios["notes"].append(
                    f"Skipped illustrative xDavg_target={xDavg_target:.4f}: {exc}"
                )

        for xB in [0.0025, 0.005, 0.01, 0.02]:
            if xB >= x0:
                continue
            try:
                result = solve_batch_given_W0_x0_xB(
                    W0=W0,
                    x0=x0,
                    xB=xB,
                    n=100,
                )
                scenarios["final_still_scenarios"].append(
                    {
                        "xB": xB,
                        "D": result["D"],
                        "B": result["B"],
                        "xDavg": result["xDavg"],
                    }
                )
            except Exception as exc:
                scenarios["notes"].append(
                    f"Skipped illustrative xB={xB:.4f}: {exc}"
                )

        return scenarios

    if "D" in knowns and "xDavg_target" in knowns and ("W0" not in knowns or "x0" not in knowns):
        scenarios["notes"].append(
            "A target distillate amount (D) together with a target average distillate ethanol mole fraction (xDavg_target) still needs another design basis such as the initial ethanol mole fraction (x0) or the final still ethanol mole fraction (xB) before reliable illustrative scenarios can be generated with the currently supported solvers."
        )
        return scenarios

    scenarios["notes"].append(
        "Illustrative scenarios are only generated when a supported solver can be run with the current known inputs."
    )
    return scenarios
