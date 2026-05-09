from typing import Any

from agents.design_experiments import (
    build_sample_possible_plan,
    run_planned_scenarios,
    shifted_sample_values,
)
from agents.error_handling import VARIABLE_DISPLAY_NAMES
from agents.workflows import VARIABLE_DESCRIPTIONS

CONFIRM_COMMIT_MESSAGES = {
    "yes",
    "yeah",
    "yep",
    "confirm",
    "use it",
    "use that",
    "go with it",
    "that works",
}

REJECT_COMMIT_MESSAGES = {
    "no",
    "cancel",
    "don't use it",
    "dont use it",
    "choose another",
    "not that one",
}


def _normalize_reply(message: str) -> str:
    return " ".join(message.strip().lower().split())


def is_pending_commit_confirmation(message: str) -> bool:
    return _normalize_reply(message) in CONFIRM_COMMIT_MESSAGES


def is_pending_commit_rejection(message: str) -> bool:
    return _normalize_reply(message) in REJECT_COMMIT_MESSAGES


def clear_pending_commit_state() -> dict[str, Any]:
    return {
        "pending_commit_variable": None,
        "pending_commit_value": None,
        "pending_commit_source": None,
    }


def format_scenario_row(row: dict[str, Any]) -> str:
    sampled_variable = row.get("sampled_variable")
    status = row.get("status", "unknown")
    sample_value = row.get("sampled_value")
    prefix = "custom " if row.get("custom") else ""
    if sampled_variable in {"xB", "x0"} and "W0" in row:
        return (
            f"{prefix}{sampled_variable}={sample_value:.4f} -> W0={row['W0']:.3f} mol, "
            f"B={row['B']:.3f} mol, check={status}"
        )
    if sampled_variable == "xDavg_target":
        line = (
            f"{prefix}xDavg_target={sample_value:.4f} -> D={row.get('D', 0):.3f} mol, "
            f"B={row.get('B', 0):.3f} mol, xB={row.get('xB', 0):.6f}, status={status}"
        )
        if row.get("note"):
            line += f" ({row['note']})"
        return line
    if sampled_variable == "xB":
        line = (
            f"{prefix}xB={sample_value:.4f} -> D={row.get('D', 0):.3f} mol, "
            f"B={row.get('B', 0):.3f} mol, xDavg={row.get('xDavg', 0):.6f}, status={status}"
        )
        if row.get("note"):
            line += f" ({row['note']})"
        return line
    return f"{prefix}{sampled_variable}={sample_value} -> status={status}"


def format_scenario_rows(rows: list[dict[str, Any]], max_rows: int = 5) -> str:
    return "\n".join("- " + format_scenario_row(row) for row in rows[:max_rows])


def build_active_experiment_state(
    knowns: dict[str, float] | None,
    sampled_variable: str | None,
    scenario_results: list[dict[str, Any]] | None,
    status: str = "awaiting_selection",
    plan: dict[str, Any] | None = None,
    active_experiment_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not scenario_results:
        return {
            "active_experiment": None,
            "experiment_results": None,
            "experiment_sampled_variable": None,
            "experiment_knowns": None,
            "experiment_status": None,
        }

    active_experiment = {
        "base_knowns": dict(knowns or {}),
        "sampled_variable": sampled_variable,
        "plan": plan,
    }
    if active_experiment_overrides:
        active_experiment.update(active_experiment_overrides)

    return {
        "active_experiment": active_experiment,
        "experiment_results": scenario_results,
        "experiment_sampled_variable": sampled_variable,
        "experiment_knowns": dict(knowns or {}),
        "experiment_status": status,
    }


def handle_pending_commit_response(
    state: dict[str, Any],
    user_message: str,
) -> dict[str, Any] | None:
    pending_variable = state.get("pending_commit_variable")
    pending_value = state.get("pending_commit_value")
    pending_source = state.get("pending_commit_source") or {}

    if pending_variable is None or pending_value is None:
        return None

    if is_pending_commit_confirmation(user_message):
        updated_knowns = dict(state.get("knowns") or state.get("prior_knowns") or {})
        updated_knowns[pending_variable] = pending_value
        active_experiment = state.get("active_experiment") or {}
        experiment_results = state.get("experiment_results")
        experiment_sampled_variable = state.get("experiment_sampled_variable")
        active_overrides = dict(active_experiment)
        active_overrides["committed_variable"] = pending_variable
        active_overrides["committed_value"] = pending_value
        active_overrides["pending_commit_source"] = pending_source
        final_answer = (
            f"Got it. I'll use {VARIABLE_DISPLAY_NAMES.get(pending_variable, pending_variable)} = {pending_value:.4f} going forward."
        )
        return {
            "knowns": updated_knowns,
            "guidance_response": final_answer,
            "final_answer": final_answer,
            **build_active_experiment_state(
                knowns=updated_knowns,
                sampled_variable=experiment_sampled_variable,
                scenario_results=experiment_results,
                status="selection_committed",
                plan=active_experiment.get("plan"),
                active_experiment_overrides=active_overrides,
            ),
            **clear_pending_commit_state(),
        }

    if is_pending_commit_rejection(user_message):
        final_answer = (
            "Okay, I won't use that value as the design basis. Choose another option, or give me a custom value to try."
        )
        return {
            "guidance_response": final_answer,
            "final_answer": final_answer,
            **clear_pending_commit_state(),
        }

    return None


def handle_experiment_followup(
    state: dict[str, Any],
    command: dict[str, Any],
) -> dict[str, Any] | None:
    active_experiment = state.get("active_experiment")
    if not active_experiment:
        return None

    experiment_results = state.get("experiment_results") or []
    experiment_sampled_variable = state.get("experiment_sampled_variable")
    experiment_knowns = state.get("experiment_knowns") or {}
    intent = command["intent"]
    if command.get("needs_clarification"):
        final_answer = command.get(
            "clarification_question",
            "I need one more detail before I can continue that experiment follow-up.",
        )
        return {
            "guidance_response": final_answer,
            "final_answer": final_answer,
        }

    if intent == "end_experiment":
        final_answer = (
            "Done with this experiment. I cleared the active scenario set and kept your remembered known values."
        )
        return {
            "guidance_response": final_answer,
            "final_answer": final_answer,
            **build_active_experiment_state(None, None, None),
            **clear_pending_commit_state(),
        }

    if intent == "select_option":
        option_index = command["option_index"] or 0
        if option_index == 0 and command.get("relative_choice"):
            relative_choice = command["relative_choice"]
            if relative_choice == "first" and experiment_results:
                option_index = 1
            elif relative_choice == "last" and experiment_results:
                option_index = len(experiment_results)
            elif relative_choice == "middle" and experiment_results:
                option_index = (len(experiment_results) + 1) // 2
        if option_index < 1 or option_index > len(experiment_results):
            final_answer = (
                f"I only have {len(experiment_results)} stored option(s) in the current experiment."
            )
            return {
                "guidance_response": final_answer,
                "final_answer": final_answer,
            }
        row = experiment_results[option_index - 1]
        sampled_variable = row.get("sampled_variable", experiment_sampled_variable)
        sampled_value = row.get("sampled_value")
        if sampled_variable == "xB":
            final_answer = (
                f"Option {option_index} uses final still ethanol mole fraction (xB) = {sampled_value:.4f}. "
                f"That gives initial charge amount (W0) = {row.get('W0', 0):.3f} mol and final still amount (B) = {row.get('B', 0):.3f} mol. "
                f"Should I use xB = {sampled_value:.4f} as the design basis going forward?"
            )
        elif sampled_variable == "x0":
            final_answer = (
                f"Option {option_index} uses initial ethanol mole fraction (x0) = {sampled_value:.4f}. "
                f"That gives initial charge amount (W0) = {row.get('W0', 0):.3f} mol and final still amount (B) = {row.get('B', 0):.3f} mol. "
                f"Should I use x0 = {sampled_value:.4f} as the design basis going forward?"
            )
        else:
            final_answer = (
                f"Option {option_index} uses {sampled_variable} = {sampled_value:.4f}. "
                "Should I use that as the design basis going forward?"
            )
        return {
            "guidance_response": final_answer,
            "final_answer": final_answer,
            **build_active_experiment_state(
                experiment_knowns,
                experiment_sampled_variable,
                experiment_results,
                status="awaiting_confirmation",
                plan=active_experiment.get("plan"),
                active_experiment_overrides={
                    "selected_option": option_index,
                    "selected_row": row,
                },
            ),
            "pending_commit_variable": sampled_variable,
            "pending_commit_value": sampled_value,
            "pending_commit_source": {
                "option_index": option_index,
                "row": row,
                "source": "selected_option",
            },
        }

    if intent == "try_custom_value":
        variable = command["target_variable"]
        value = command["value"]
        if variable is None or value is None:
            final_answer = "I could not parse that experiment value."
            return {
                "guidance_response": final_answer,
                "final_answer": final_answer,
            }
        custom_plan = build_sample_possible_plan(
            knowns=experiment_knowns,
            sampled_variable=variable,
            sample_values=[value],
            reason=f"I sampled {VARIABLE_DISPLAY_NAMES.get(variable, variable)} at the value you provided.",
            next_question=f"If you want, I can try more {VARIABLE_DISPLAY_NAMES.get(variable, variable)} values.",
        )
        scenario_result = run_planned_scenarios(experiment_knowns, custom_plan, n=100)
        if not scenario_result["rows"]:
            final_answer = (
                f"I could not generate a scenario for {VARIABLE_DISPLAY_NAMES.get(variable, variable)} = {value:.4f} with the current experiment context."
            )
            return {
                "guidance_response": final_answer,
                "final_answer": final_answer,
            }
        custom_rows = []
        for row in scenario_result["rows"]:
            custom_row = dict(row)
            custom_row["custom"] = True
            custom_row["source"] = "custom_value"
            custom_rows.append(custom_row)

        merged_rows = [
            row
            for row in experiment_results
            if not (
                row.get("sampled_variable") == variable
                and row.get("sampled_value") == value
            )
        ] + custom_rows

        final_answer = (
            f"I tried {VARIABLE_DISPLAY_NAMES.get(variable, variable)} = {value:.4f}.\n\n"
            + format_scenario_rows(custom_rows, max_rows=1)
            + f"\n\nShould I use {variable} = {value:.4f} as the design basis going forward?"
        )
        return {
            "guidance_response": final_answer,
            "final_answer": final_answer,
            **build_active_experiment_state(
                experiment_knowns,
                variable,
                merged_rows,
                plan=custom_plan,
            ),
            "pending_commit_variable": variable,
            "pending_commit_value": value,
            "pending_commit_source": {
                "row": custom_rows[0],
                "source": "custom_value",
            },
        }

    if intent == "explain_option":
        option_index = command["option_index"] or 0
        if option_index == 0 and command.get("relative_choice"):
            relative_choice = command["relative_choice"]
            if relative_choice == "first" and experiment_results:
                option_index = 1
            elif relative_choice == "last" and experiment_results:
                option_index = len(experiment_results)
            elif relative_choice == "middle" and experiment_results:
                option_index = (len(experiment_results) + 1) // 2
        if option_index < 1:
            final_answer = "Which option would you like me to explain?"
            return {
                "guidance_response": final_answer,
                "final_answer": final_answer,
            }
        if option_index > len(experiment_results):
            final_answer = (
                f"I only have {len(experiment_results)} stored option(s) in the current experiment."
            )
            return {
                "guidance_response": final_answer,
                "final_answer": final_answer,
            }
        row = experiment_results[option_index - 1]
        sampled_variable = row.get("sampled_variable", experiment_sampled_variable)
        sampled_value = row.get("sampled_value")
        variable_label = VARIABLE_DISPLAY_NAMES.get(sampled_variable, sampled_variable)
        explanation = (
            f"Option {option_index} samples {variable_label} = {sampled_value:.4f}. "
            + format_scenario_row(row).lstrip("- ")
            + ". "
            + "This row keeps the other experiment inputs fixed and shows the resulting scenario."
        )
        return {
            "guidance_response": explanation,
            "final_answer": explanation,
        }

    if intent in {"shift_samples_higher", "shift_samples_lower"}:
        variable = command["target_variable"] or experiment_sampled_variable
        if variable is None:
            final_answer = "I do not have an active sampled variable to adjust yet."
            return {
                "guidance_response": final_answer,
                "final_answer": final_answer,
            }
        current_values = [
            row["sampled_value"]
            for row in experiment_results
            if row.get("sampled_variable") == variable and row.get("sampled_value") is not None
        ]
        sample_values = shifted_sample_values(
            sampled_variable=variable,
            current_values=current_values,
            knowns=experiment_knowns,
            direction="higher" if intent == "shift_samples_higher" else "lower",
        )
        if not sample_values:
            final_answer = (
                f"I do not have any reasonable {'higher' if intent == 'shift_samples_higher' else 'lower'} {VARIABLE_DISPLAY_NAMES.get(variable, variable)} values to sample from the current experiment."
            )
            return {
                "guidance_response": final_answer,
                "final_answer": final_answer,
            }
        custom_plan = build_sample_possible_plan(
            knowns=experiment_knowns,
            sampled_variable=variable,
            sample_values=sample_values,
            reason=f"I sampled {'higher' if intent == 'shift_samples_higher' else 'lower'} {VARIABLE_DISPLAY_NAMES.get(variable, variable)} values from the current experiment.",
            next_question="Pick one of these values, or give your own.",
        )
        scenario_result = run_planned_scenarios(experiment_knowns, custom_plan, n=100)
        final_answer = (
            f"I sampled {'higher' if intent == 'shift_samples_higher' else 'lower'} {VARIABLE_DISPLAY_NAMES.get(variable, variable)} values.\n\n"
            + format_scenario_rows(scenario_result["rows"], max_rows=3)
        )
        return {
            "guidance_response": final_answer,
            "final_answer": final_answer,
            **build_active_experiment_state(
                experiment_knowns,
                variable,
                scenario_result["rows"],
                plan=custom_plan,
            ),
            **clear_pending_commit_state(),
        }

    if intent == "switch_sampling_axis":
        variable = command["target_variable"]
        if variable == "x0":
            selected_row = (active_experiment or {}).get("selected_row")
            if not selected_row or "xB" not in selected_row:
                available_xb_rows = [
                    row
                    for row in experiment_results
                    if row.get("sampled_variable") == "xB" and row.get("sampled_value") is not None
                ]
                option_text = ""
                if available_xb_rows:
                    option_bits = [
                        f"option {index + 1} (xB = {row['sampled_value']:.4f})"
                        for index, row in enumerate(available_xb_rows[:5])
                    ]
                    option_text = (
                        "\n\nAvailable xB options:\n- "
                        + "\n- ".join(option_bits)
                    )
                final_answer = (
                    "To compare initial ethanol mole fraction (x0), I need to hold one final still ethanol mole fraction (xB) fixed."
                    + option_text
                    + "\n\nPick one of the previous xB options, or provide your own xB value. Which xB should I hold fixed?"
                )
                return {
                    "guidance_response": final_answer,
                    "final_answer": final_answer,
                    **clear_pending_commit_state(),
                }
            comparison_knowns = {
                "D": selected_row["D"],
                "xDavg_target": selected_row.get(
                    "xDavg", experiment_knowns.get("xDavg_target")
                ),
                "xB": selected_row["xB"],
            }
            custom_plan = build_sample_possible_plan(
                knowns=comparison_knowns,
                sampled_variable="x0",
                sample_values=[0.03, 0.05, 0.10],
                reason="I switched the comparison axis to initial ethanol mole fraction (x0).",
                next_question="Pick one of these x0 values, or give your own.",
            )
            scenario_result = run_planned_scenarios(comparison_knowns, custom_plan, n=100)
            final_answer = (
                "I switched the comparison axis to initial ethanol mole fraction (x0).\n\n"
                + format_scenario_rows(scenario_result["rows"], max_rows=3)
            )
            return {
                "guidance_response": final_answer,
                "final_answer": final_answer,
                **build_active_experiment_state(
                    comparison_knowns,
                    "x0",
                    scenario_result["rows"],
                    plan=custom_plan,
                ),
                **clear_pending_commit_state(),
            }

        final_answer = (
            f"I can only switch to {VARIABLE_DISPLAY_NAMES.get(variable, variable)} if the current experiment has the right fixed inputs for that comparison."
        )
        return {
            "guidance_response": final_answer,
            "final_answer": final_answer,
            **clear_pending_commit_state(),
        }

    return None


def build_design_advisor_response(
    knowns: dict[str, float],
    analysis: dict[str, Any],
    experiment_plan: dict[str, Any],
    planned_scenarios: dict[str, Any],
    detailed: bool,
) -> dict[str, Any]:
    if knowns:
        known_lines = []
        short_known_bits = []
        for name, value in knowns.items():
            description = VARIABLE_DISPLAY_NAMES.get(name, VARIABLE_DESCRIPTIONS.get(name, name))
            if isinstance(value, float):
                if name in {"W0", "B", "D"}:
                    value_text = f"{value:.3f}"
                else:
                    value_text = f"{value:.6f}"
            else:
                value_text = str(value)
            known_lines.append(f"- {description} = {value_text}")
            short_known_bits.append(f"{description} = {value_text}")
        knowns_block = "Known inputs so far:\n" + "\n".join(known_lines)
        knowns_summary = "You gave " + "; ".join(short_known_bits) + "."
    else:
        knowns_block = "Known inputs so far:\n- none yet"
        knowns_summary = "You have not given any numeric inputs yet."

    plan_status = experiment_plan["status"]
    if plan_status == "ready_to_calculate":
        status_intro = "You already have enough information for a supported workflow."
    elif plan_status == "choose_sampling_axis":
        status_intro = "Your design is not unique yet."
    elif plan_status == "sample_possible":
        status_intro = "I can help you explore a useful sampling axis."
    else:
        status_intro = "I can suggest the next useful design basis."

    relevant_workflow_lines = []
    for workflow in analysis["closest_workflows"]:
        required_text = ", ".join(
            VARIABLE_DISPLAY_NAMES.get(name, name) for name in workflow["required_inputs"]
        )
        missing_text = (
            ", ".join(
                VARIABLE_DISPLAY_NAMES.get(name, name) for name in workflow["missing_inputs"]
            )
            if workflow["missing_inputs"]
            else "none"
        )
        relevant_workflow_lines.append(
            f"- {workflow['label']}: requires {required_text}; missing now: {missing_text}."
        )

    compact_example_lines = [
        "- " + format_scenario_row(row) for row in planned_scenarios["rows"][:5]
    ]
    scenario_block = ""
    if compact_example_lines:
        scenario_block = (
            "Illustrative example scenarios:\n" + "\n".join(compact_example_lines)
        )
        if planned_scenarios["notes"]:
            scenario_block += "\n\nNotes:\n" + "\n".join(
                f"- {note}" for note in planned_scenarios["notes"]
            )
        scenario_block += "\n\nThese are illustrative examples to compare design choices."
    elif planned_scenarios["notes"]:
        scenario_block = "Notes:\n" + "\n".join(
            f"- {note}" for note in planned_scenarios["notes"]
        )

    planning_lines = []
    if experiment_plan["candidate_sampling_variables"]:
        planning_lines.append(
            "Candidate sampling variables: "
            + ", ".join(
                VARIABLE_DISPLAY_NAMES.get(name, VARIABLE_DESCRIPTIONS.get(name, name))
                for name in experiment_plan["candidate_sampling_variables"]
            )
        )
    if experiment_plan["recommended_sampling_variable"]:
        recommended_variable = experiment_plan["recommended_sampling_variable"]
        if recommended_variable == "both":
            planning_lines.append(
                "Recommended sampling plan: compare both target average distillate composition (xDavg_target) and final still ethanol mole fraction (xB)."
            )
        else:
            planning_lines.append(
                "Recommended sampling variable: "
                + VARIABLE_DISPLAY_NAMES.get(
                    recommended_variable,
                    VARIABLE_DESCRIPTIONS.get(recommended_variable, recommended_variable),
                )
            )
    if experiment_plan["sample_values"]:
        sample_lines = []
        for variable_name, values in experiment_plan["sample_values"].items():
            if not values:
                continue
            sample_lines.append(
                f"- {VARIABLE_DISPLAY_NAMES.get(variable_name, VARIABLE_DESCRIPTIONS.get(variable_name, variable_name))}: "
                + ", ".join(f"{value:.4f}" for value in values[:5])
            )
        if sample_lines:
            planning_lines.append("Illustrative sample values:\n" + "\n".join(sample_lines))

    if detailed:
        final_answer = (
            status_intro
            + "\n\n"
            + knowns_block
            + "\n\nRelevant supported workflows:\n"
            + "\n".join(relevant_workflow_lines)
            + "\n\n"
            + experiment_plan["reason"]
        )

        if planning_lines:
            final_answer += "\n\n" + "\n\n".join(planning_lines)

        if scenario_block:
            final_answer += "\n\n" + scenario_block

        final_answer += "\n\n" + experiment_plan["next_question"]
    else:
        brief_parts = [knowns_summary, experiment_plan["reason"], experiment_plan["next_question"]]
        final_answer = "\n\n".join(part for part in brief_parts if part)

        if plan_status == "sample_possible" and compact_example_lines:
            final_answer += (
                "\n\nIllustrative example scenarios:\n"
                + "\n".join(compact_example_lines[:3])
                + "\n\nThese are illustrative examples to compare design choices."
            )
        elif plan_status == "sample_possible" and experiment_plan["sample_values"]:
            sample_text_lines = []
            for variable_name, values in experiment_plan["sample_values"].items():
                if not values:
                    continue
                sample_text_lines.append(
                    f"- {VARIABLE_DISPLAY_NAMES.get(variable_name, VARIABLE_DESCRIPTIONS.get(variable_name, variable_name))}: "
                    + ", ".join(f"{value:.4f}" for value in values[:3])
                )
            if sample_text_lines:
                final_answer += "\n\nIllustrative sample values:\n" + "\n".join(sample_text_lines)

    return {
        "guidance_response": final_answer,
        "final_answer": final_answer,
        **build_active_experiment_state(
            knowns=knowns,
            sampled_variable=experiment_plan.get("recommended_sampling_variable"),
            scenario_results=planned_scenarios["rows"],
            status="awaiting_selection",
            plan=experiment_plan,
        ),
        **clear_pending_commit_state(),
    }
