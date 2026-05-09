import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from agents.design_experiments import (
    build_sample_possible_plan,
    plan_experiment_from_knowns,
    run_planned_scenarios,
    shifted_sample_values,
)
from agents.error_handling import VARIABLE_DISPLAY_NAMES, normalize_error_for_user
from agents.experiment_commands import parse_experiment_command
from agents.prompts import build_problem_structurer_prompt
from agents.response_style import wants_detailed_explanation
from agents.schemas import LLMProblemRequest, ProblemRequest
from agents.state import BatchDistillationState
from agents.workflows import (
    SUPPORTED_WORKFLOWS,
    VARIABLE_DESCRIPTIONS,
    analyze_knowns_against_workflows,
)
from engineering.tools import (
    check_batch_consistency,
    solve_batch_given_W0_x0_xB,
    solve_D_given_W0_x0_xDavg,
)

load_dotenv()


def _format_scenario_row(row: dict) -> str:
    sampled_variable = row.get("sampled_variable")
    status = row.get("status", "unknown")
    sample_value = row.get("sampled_value")
    if sampled_variable in {"xB", "x0"} and "W0" in row:
        line = (
            f"{sampled_variable}={sample_value:.4f} -> W0={row['W0']:.3f} mol, "
            f"B={row['B']:.3f} mol, check={status}"
        )
        return line
    if sampled_variable == "xDavg_target":
        line = (
            f"xDavg_target={sample_value:.4f} -> D={row.get('D', 0):.3f} mol, "
            f"B={row.get('B', 0):.3f} mol, xB={row.get('xB', 0):.6f}, status={status}"
        )
        if row.get("note"):
            line += f" ({row['note']})"
        return line
    if sampled_variable == "xB":
        line = (
            f"xB={sample_value:.4f} -> D={row.get('D', 0):.3f} mol, "
            f"B={row.get('B', 0):.3f} mol, xDavg={row.get('xDavg', 0):.6f}, status={status}"
        )
        if row.get("note"):
            line += f" ({row['note']})"
        return line
    return f"{sampled_variable}={sample_value} -> status={status}"


def _format_compact_scenario_rows(rows: list[dict]) -> str:
    return "\n".join("- " + _format_scenario_row(row) for row in rows)


def face_node(state: BatchDistillationState) -> BatchDistillationState:
    """
    First node.

    For now, this just passes the user's message forward.
    Later, this can become a conversational LLM node.
    """
    user_message = state["user_message"]

    return {
        "user_goal": user_message,
    }


def problem_structurer_node(state: BatchDistillationState) -> BatchDistillationState:
    """
    Second node.

    Uses an LLM plus a Pydantic schema to turn the user's request into a
    structured calculation request.
    """
    user_message = state.get("user_goal", state["user_message"])
    prior_knowns = state.get("prior_knowns", {})
    prior_needs_clarification = state.get("prior_needs_clarification", False)
    prior_clarification_question = state.get("prior_clarification_question")

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is missing. Add it to .env before running the LLM ProblemStructurer."
        )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    structured_llm = llm.with_structured_output(
        LLMProblemRequest,
        method="function_calling",
    )

    structured = structured_llm.invoke(
        build_problem_structurer_prompt(
            user_message=user_message,
            prior_knowns=prior_knowns,
            prior_needs_clarification=prior_needs_clarification,
            prior_clarification_question=prior_clarification_question,
        )
    )
    merged_knowns = {**prior_knowns, **structured.knowns.model_dump(exclude_none=True)}
    structured_dict = ProblemRequest(
        intent_type=structured.intent_type,
        problem_type=structured.problem_type,
        knowns=merged_knowns,
        unknowns=structured.unknowns,
        needs_clarification=structured.needs_clarification,
        clarification_question=structured.clarification_question,
    ).model_dump()

    return structured_dict


def guidance_responder_node(state: BatchDistillationState) -> BatchDistillationState:
    """
    Handles broad orientation, conceptual help, and unsupported requests.
    """
    user_message = state.get("user_message", "")
    wants_detail = wants_detailed_explanation(user_message)
    wants_options = any(
        phrase in user_message.lower()
        for phrase in ("how do i start", "what can you do", "show me my options", "options")
    )

    workflow_lines = []
    for workflow in SUPPORTED_WORKFLOWS.values():
        workflow_lines.append(
            f"- {workflow['label']}: {workflow['description']} "
            f"Required inputs: {', '.join(workflow['required_inputs'])}."
        )

    if state.get("needs_clarification", False):
        question = state.get(
            "clarification_question",
            "What additional information do you want to provide?",
        )
        if wants_detail:
            final_answer = (
                "I need one more piece of information before I can run that calculation.\n\n"
                + "\n".join(workflow_lines)
                + "\n\n"
                + question
            )
        else:
            final_answer = (
                "I need one more piece of information before I can run that calculation.\n\n"
                + question
            )
    elif state.get("intent_type") == "conceptual_question":
        if wants_detail:
            final_answer = (
                "I can explain the variables and supported workflows.\n\n"
                + "\n".join(workflow_lines)
                + "\n\nWhich variable or workflow would you like to clarify first?"
            )
        else:
            final_answer = "I can explain the variables or the workflow options. What would you like to clarify?"
    elif state.get("intent_type") == "open_ended_guidance":
        if wants_detail or wants_options:
            final_answer = (
                "I can help you choose a calculation path.\n\nSupported workflows:\n"
                + "\n".join(workflow_lines)
                + "\n\nTo start, do you know the initial charge amount (W0) and the initial ethanol mole fraction (x0)?"
            )
        else:
            final_answer = (
                "I can help you choose a calculation path, such as targeting the average distillate composition or the final still composition. "
                "To start, do you know the initial charge amount (W0) and the initial ethanol mole fraction (x0)?"
            )
    else:
        if wants_detail:
            final_answer = (
                "I can help with a few supported batch-distillation tasks.\n\nSupported workflows:\n"
                + "\n".join(workflow_lines)
                + "\n\nWhich workflow would you like to try?"
            )
        else:
            final_answer = "I can help with a few supported batch-distillation tasks. Which workflow would you like to try?"

    return {
        "guidance_response": final_answer,
        "final_answer": final_answer,
        }


def design_advisor_node(state: BatchDistillationState) -> BatchDistillationState:
    """
    Handles partial-knowns design guidance and illustrative scenario exploration.
    """
    knowns = state.get("knowns", {})
    unknowns = state.get("unknowns", [])
    user_message = state.get("user_message", "")
    user_goal = state.get("user_goal", user_message)
    wants_detail = wants_detailed_explanation(user_message)
    active_experiment = state.get("active_experiment")
    experiment_results = state.get("experiment_results") or []
    experiment_sampled_variable = state.get("experiment_sampled_variable")
    experiment_knowns = state.get("experiment_knowns") or {}
    experiment_command = parse_experiment_command(user_message)

    if experiment_command["is_experiment_command"] and active_experiment:
        action = experiment_command["action"]

        if action == "clear_experiment":
            final_answer = (
                "Done with this experiment. I cleared the active scenario set and kept your remembered known values."
            )
            return {
                "guidance_response": final_answer,
                "final_answer": final_answer,
                "active_experiment": None,
                "experiment_results": None,
                "experiment_sampled_variable": None,
                "experiment_knowns": None,
                "experiment_status": None,
            }

        if action == "use_option":
            option_index = experiment_command["option_index"] or 0
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
            updated_experiment = dict(active_experiment)
            updated_experiment["selected_option"] = option_index
            updated_experiment["selected_row"] = row
            return {
                "guidance_response": final_answer,
                "final_answer": final_answer,
                "active_experiment": updated_experiment,
                "experiment_results": experiment_results,
                "experiment_sampled_variable": experiment_sampled_variable,
                "experiment_knowns": experiment_knowns,
                "experiment_status": "awaiting_confirmation",
            }

        if action == "try_value":
            variable = experiment_command["variable"]
            value = experiment_command["value"]
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
            row = scenario_result["rows"][0]
            final_answer = (
                f"I tried {VARIABLE_DISPLAY_NAMES.get(variable, variable)} = {value:.4f}.\n\n"
                + _format_compact_scenario_rows(scenario_result["rows"][:1])
            )
            return {
                "guidance_response": final_answer,
                "final_answer": final_answer,
                "active_experiment": {
                    "base_knowns": experiment_knowns,
                    "sampled_variable": variable,
                    "plan": custom_plan,
                },
                "experiment_results": scenario_result["rows"],
                "experiment_sampled_variable": variable,
                "experiment_knowns": experiment_knowns,
                "experiment_status": "awaiting_selection",
            }

        if action in {"show_higher", "show_lower"}:
            variable = experiment_command["variable"] or experiment_sampled_variable
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
                direction="higher" if action == "show_higher" else "lower",
            )
            if not sample_values:
                final_answer = (
                    f"I do not have any reasonable {action.split('_')[1]} {VARIABLE_DISPLAY_NAMES.get(variable, variable)} values to sample from the current experiment."
                )
                return {
                    "guidance_response": final_answer,
                    "final_answer": final_answer,
                }
            custom_plan = build_sample_possible_plan(
                knowns=experiment_knowns,
                sampled_variable=variable,
                sample_values=sample_values,
                reason=f"I sampled {action.split('_')[1]} {VARIABLE_DISPLAY_NAMES.get(variable, variable)} values from the current experiment.",
                next_question="Pick one of these values, or give your own.",
            )
            scenario_result = run_planned_scenarios(experiment_knowns, custom_plan, n=100)
            final_answer = (
                f"I sampled {action.split('_')[1]} {VARIABLE_DISPLAY_NAMES.get(variable, variable)} values.\n\n"
                + _format_compact_scenario_rows(scenario_result["rows"][:3])
            )
            return {
                "guidance_response": final_answer,
                "final_answer": final_answer,
                "active_experiment": {
                    "base_knowns": experiment_knowns,
                    "sampled_variable": variable,
                    "plan": custom_plan,
                },
                "experiment_results": scenario_result["rows"],
                "experiment_sampled_variable": variable,
                "experiment_knowns": experiment_knowns,
                "experiment_status": "awaiting_selection",
            }

        if action == "compare_variable":
            variable = experiment_command["variable"]
            if variable == "x0":
                selected_row = (active_experiment or {}).get("selected_row")
                if not selected_row or "xB" not in selected_row:
                    final_answer = (
                        "To compare initial ethanol mole fraction (x0) instead, I need a current final still ethanol mole fraction (xB) to hold fixed."
                    )
                    return {
                        "guidance_response": final_answer,
                        "final_answer": final_answer,
                    }
                comparison_knowns = {
                    "D": selected_row["D"],
                    "xDavg_target": selected_row.get("xDavg", experiment_knowns.get("xDavg_target")),
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
                    + _format_compact_scenario_rows(scenario_result["rows"][:3])
                )
                return {
                    "guidance_response": final_answer,
                    "final_answer": final_answer,
                    "active_experiment": {
                        "base_knowns": comparison_knowns,
                        "sampled_variable": "x0",
                        "plan": custom_plan,
                    },
                    "experiment_results": scenario_result["rows"],
                    "experiment_sampled_variable": "x0",
                    "experiment_knowns": comparison_knowns,
                    "experiment_status": "awaiting_selection",
                }

            final_answer = (
                f"I can only switch to {VARIABLE_DISPLAY_NAMES.get(variable, variable)} if the current experiment has the right fixed inputs for that comparison."
            )
            return {
                "guidance_response": final_answer,
                "final_answer": final_answer,
            }

    analysis = analyze_knowns_against_workflows(
        knowns=knowns,
        requested_outputs=unknowns,
    )
    experiment_plan = plan_experiment_from_knowns(
        knowns=knowns,
        requested_outputs=unknowns,
        user_goal=user_goal,
    )
    planned_scenarios = run_planned_scenarios(knowns=knowns, plan=experiment_plan, n=100)

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
        required_text = ", ".join(VARIABLE_DISPLAY_NAMES.get(name, name) for name in workflow["required_inputs"])
        missing_text = (
            ", ".join(VARIABLE_DISPLAY_NAMES.get(name, name) for name in workflow["missing_inputs"])
            if workflow["missing_inputs"]
            else "none"
        )
        relevant_workflow_lines.append(
            f"- {workflow['label']}: requires {required_text}; missing now: {missing_text}."
        )

    def build_active_experiment_dict() -> dict | None:
        if not planned_scenarios["rows"]:
            return None
        return {
            "base_knowns": dict(knowns),
            "sampled_variable": experiment_plan.get("recommended_sampling_variable"),
            "plan": experiment_plan,
        }

    def _scenario_rows_to_state() -> tuple[dict | None, list[dict] | None, str | None, dict | None, str | None]:
        if not planned_scenarios["rows"]:
            return None, None, None, None, None
        return (
            build_active_experiment_dict(),
            planned_scenarios["rows"],
            experiment_plan.get("recommended_sampling_variable"),
            dict(knowns),
            "awaiting_selection",
        )

    compact_example_lines = [
        "- " + _format_scenario_row(row) for row in planned_scenarios["rows"][:5]
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

    if wants_detail:
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
        "active_experiment": _scenario_rows_to_state()[0],
        "experiment_results": _scenario_rows_to_state()[1],
        "experiment_sampled_variable": _scenario_rows_to_state()[2],
        "experiment_knowns": _scenario_rows_to_state()[3],
        "experiment_status": _scenario_rows_to_state()[4],
    }


def validation_calculation_node(state: BatchDistillationState) -> BatchDistillationState:
    """
    Third node.

    Calls deterministic Python tools.
    """
    problem_type = state["problem_type"]
    knowns = state["knowns"]

    try:
        if problem_type == "solve_D_given_W0_x0_xDavg":
            required_keys = ("W0", "x0", "xDavg_target")
            missing_keys = [key for key in required_keys if key not in knowns]
            if missing_keys:
                return {
                    "calculation_success": False,
                    "result": {},
                    "consistency_check": {},
                    "errors": [
                        normalize_error_for_user(
                            "missing required inputs",
                            context={"missing_inputs": missing_keys},
                        )
                    ],
                    "warnings": [],
                }

            result = solve_D_given_W0_x0_xDavg(
                W0=knowns["W0"],
                x0=knowns["x0"],
                xDavg_target=knowns["xDavg_target"],
                n=100,
            )

            check = check_batch_consistency(
                W0=result["W0"],
                B=result["B"],
                D=result["D"],
                x0=result["x0"],
                xB=result["xB"],
                xDavg=result["xDavg"],
                n=result["n"],
            )

            return {
                "calculation_success": True,
                "result": result,
                "consistency_check": check,
                "errors": [],
                "warnings": [],
            }

        if problem_type == "solve_batch_given_W0_x0_xB":
            required_keys = ("W0", "x0", "xB")
            missing_keys = [key for key in required_keys if key not in knowns]
            if missing_keys:
                return {
                    "calculation_success": False,
                    "result": {},
                    "consistency_check": {},
                    "errors": [
                        normalize_error_for_user(
                            "missing required inputs",
                            context={"missing_inputs": missing_keys},
                        )
                    ],
                    "warnings": [],
                }

            result = solve_batch_given_W0_x0_xB(
                W0=knowns["W0"],
                x0=knowns["x0"],
                xB=knowns["xB"],
                n=100,
            )

            check = check_batch_consistency(
                W0=result["W0"],
                B=result["B"],
                D=result["D"],
                x0=result["x0"],
                xB=result["xB"],
                xDavg=result["xDavg"],
                n=100,
            )

            return {
                "calculation_success": True,
                "result": result,
                "consistency_check": check,
                "errors": [],
                "warnings": [],
            }

        if problem_type == "check_batch_consistency":
            required_keys = ("W0", "B", "D", "x0", "xB", "xDavg")
            missing_keys = [key for key in required_keys if key not in knowns]
            if missing_keys:
                return {
                    "calculation_success": False,
                    "result": {},
                    "consistency_check": {},
                    "errors": [
                        normalize_error_for_user(
                            "missing required inputs",
                            context={"missing_inputs": missing_keys},
                        )
                    ],
                    "warnings": [],
                }

            check = check_batch_consistency(
                W0=knowns["W0"],
                B=knowns["B"],
                D=knowns["D"],
                x0=knowns["x0"],
                xB=knowns["xB"],
                xDavg=knowns["xDavg"],
                n=100,
            )

            return {
                "calculation_success": True,
                "result": dict(knowns),
                "consistency_check": check,
                "errors": [],
                "warnings": [],
            }

        return {
            "calculation_success": False,
            "result": {},
            "consistency_check": {},
            "errors": [
                normalize_error_for_user(
                    "unsupported problem_type",
                    context={"problem_type": "unsupported"},
                )
            ],
            "warnings": [],
        }

    except Exception as exc:
        return {
            "calculation_success": False,
            "result": {},
            "consistency_check": {},
            "errors": [
                normalize_error_for_user(
                    exc,
                    context={"problem_type": problem_type},
                )
            ],
            "warnings": [],
        }


def result_explainer_node(state: BatchDistillationState) -> BatchDistillationState:
    """
    Fourth node.

    For now, this is deterministic formatting.
    Later, this can become an LLM explanation node.
    """

    if state.get("needs_clarification", False):
        return {
            "final_answer": state.get(
                "clarification_question",
                "I need more information before I can calculate the result.",
            )
        }

    if not state.get("calculation_success", False):
        errors = state.get("errors", [])
        error_lines = "\n".join(f"- {error}" for error in errors) if errors else "- I could not complete the calculation."
        return {
            "final_answer": (
                "I could not complete the calculation.\n\n"
                + error_lines
            )
        }

    result = state["result"]
    check = state["consistency_check"]
    problem_type = state["problem_type"]

    if problem_type == "check_batch_consistency":
        final_answer = (
            "Consistency check result:\n\n"
            + f"- Fully consistent: {check.get('is_fully_consistent')}\n"
            + f"- Total balance consistent: {check.get('is_total_balance_consistent')}\n"
            + f"- Component balance consistent: {check.get('is_component_balance_consistent')}\n"
            + f"- Rayleigh equation consistent: {check.get('is_rayleigh_consistent')}"
        )
        return {
            "final_answer": final_answer
        }

    D = result["D"]
    B = result["B"]
    xB = result["xB"]
    xDavg = result["xDavg"]
    W0 = result["W0"]
    x0 = result["x0"]

    input_summary_lines = [
        f"- Initial charge amount (W0) = {W0:.3f} mol",
        f"- Initial ethanol mole fraction (x0) = {x0:.6f}",
    ]

    if "xDavg_target" in result:
        input_summary_lines.append(
            f"- Target average distillate ethanol mole fraction (xDavg_target) = {result['xDavg_target']:.6f}"
        )
    else:
        input_summary_lines.append(
            f"- Target final still ethanol mole fraction (xB) = {xB:.6f}"
        )

    final_answer = (
        "For the batch distillation problem:\n\n"
        + "\n".join(input_summary_lines)
        + "\n\nCalculated result:\n\n"
        + f"- Distillate amount (D) = {D:.3f} mol\n"
        + f"- Final still amount (B) = {B:.3f} mol\n"
        + f"- Final still ethanol mole fraction (xB) = {xB:.6f}\n"
        + f"- Average distillate ethanol mole fraction (xDavg) = {xDavg:.6f}\n\n"
        + "Consistency check:\n\n"
        + f"- Fully consistent: {check.get('is_fully_consistent')}\n"
        + f"- Total balance consistent: {check.get('is_total_balance_consistent')}\n"
        + f"- Component balance consistent: {check.get('is_component_balance_consistent')}\n"
        + f"- Rayleigh equation consistent: {check.get('is_rayleigh_consistent')}"
    )

    return {
        "final_answer": final_answer
    }


def route_after_problem_structurer(state: BatchDistillationState) -> str:
    intent_type = state.get("intent_type")
    user_message = state.get("user_message", "")

    if state.get("active_experiment") and parse_experiment_command(user_message)["is_experiment_command"]:
        return "design_advisor"

    # Partial-knowns and underdetermined design requests go to the design advisor.
    if intent_type == "design_prototyping":
        return "design_advisor"

    # Explanation requests with remembered design knowns are often best answered
    # by the design advisor in the current problem context.
    if wants_detailed_explanation(user_message) and state.get("knowns"):
        return "design_advisor"

    # Broad orientation, conceptual help, and unsupported requests go to guidance.
    if intent_type in {"open_ended_guidance", "conceptual_question", "unknown"}:
        return "guidance_responder"

    # Clarification requests still need a user-facing guidance response.
    if state.get("needs_clarification", False):
        return "guidance_responder"

    # Ready calculation requests continue to deterministic validation.
    if intent_type in {"calculation_request", "clarification_answer"}:
        return "validation_calculation"

    if state.get("problem_type") == "unknown":
        return "guidance_responder"

    return "validation_calculation"
