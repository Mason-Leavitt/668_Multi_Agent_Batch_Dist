import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from agents.design_advisor_helpers import (
    build_design_advisor_response,
    handle_pending_commit_response,
    handle_experiment_followup,
    is_pending_commit_confirmation,
    is_pending_commit_rejection,
)
from agents.design_experiments import plan_experiment_from_knowns, run_planned_scenarios
from agents.error_handling import normalize_error_for_user
from agents.experiment_commands import parse_experiment_command
from agents.experiment_followup_interpreter import (
    interpret_experiment_followup_with_llm,
    is_plausible_experiment_followup,
)
from agents.prompts import build_problem_structurer_prompt
from agents.response_style import wants_detailed_explanation
from agents.schemas import LLMProblemRequest, ProblemRequest
from agents.state import BatchDistillationState
from agents.workflows import SUPPORTED_WORKFLOWS, analyze_knowns_against_workflows
from engineering.tools import (
    check_batch_consistency,
    solve_batch_given_W0_x0_xB,
    solve_D_given_W0_x0_xDavg,
)

load_dotenv()


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
    pending_commit_response = handle_pending_commit_response(state, user_message)
    if pending_commit_response is not None:
        return pending_commit_response
    experiment_followup = parse_experiment_command(user_message)
    if (
        state.get("active_experiment")
        and (
            not experiment_followup["is_experiment_followup"]
            or experiment_followup["intent"] == "unknown"
        )
        and is_plausible_experiment_followup(user_message)
    ):
        experiment_followup = interpret_experiment_followup_with_llm(
            user_message=user_message,
            active_experiment=state.get("active_experiment") or {},
            experiment_results=state.get("experiment_results"),
            experiment_sampled_variable=state.get("experiment_sampled_variable"),
            experiment_knowns=state.get("experiment_knowns"),
        )

    if (
        (experiment_followup["is_experiment_followup"] and experiment_followup["intent"] != "unknown")
        or experiment_followup.get("needs_clarification", False)
    ):
        followup_response = handle_experiment_followup(state, experiment_followup)
        if followup_response is not None:
            return followup_response

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
    return build_design_advisor_response(
        knowns=knowns,
        analysis=analysis,
        experiment_plan=experiment_plan,
        planned_scenarios=planned_scenarios,
        detailed=wants_detail,
    )


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

    experiment_followup = parse_experiment_command(user_message)
    if state.get("pending_commit_variable") is not None and (
        is_pending_commit_confirmation(user_message)
        or is_pending_commit_rejection(user_message)
    ):
        return "design_advisor"
    if (
        state.get("active_experiment")
        and (
            (experiment_followup["is_experiment_followup"] and experiment_followup["intent"] != "unknown")
            or is_plausible_experiment_followup(user_message)
        )
    ):
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
