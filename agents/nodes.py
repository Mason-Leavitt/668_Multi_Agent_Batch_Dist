import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from agents.prompts import build_problem_structurer_prompt
from agents.schemas import LLMProblemRequest, ProblemRequest
from agents.state import BatchDistillationState
from agents.workflows import (
    SUPPORTED_WORKFLOWS,
    VARIABLE_DESCRIPTIONS,
    analyze_knowns_against_workflows,
    prototype_supported_scenarios,
)
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
    workflow_lines = []
    for workflow in SUPPORTED_WORKFLOWS.values():
        workflow_lines.append(
            f"- {workflow['label']}: {workflow['description']} "
            f"Required inputs: {', '.join(workflow['required_inputs'])}."
        )

    if state.get("needs_clarification", False):
        intro = (
            "I need one more piece of information before I can run a deterministic calculation."
        )
        question = state.get(
            "clarification_question",
            "What additional information do you want to provide?",
        )
    elif state.get("intent_type") == "conceptual_question":
        intro = (
            "I can help explain the supported batch distillation workflows and the inputs they use."
        )
        question = "Which variable or workflow would you like to clarify first?"
    elif state.get("intent_type") == "open_ended_guidance":
        intro = "I can help you approach the batch distillation calculation in a few ways."
        question = (
            "To start, do you know your initial charge W0 and initial ethanol mole fraction x0?"
        )
    else:
        intro = "I can help with a few supported batch distillation tasks."
        question = "Which workflow would you like to try?"

    final_answer = (
        intro
        + "\n\nSupported workflows:\n"
        + "\n".join(workflow_lines)
        + "\n\n"
        + question
    )

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

    analysis = analyze_knowns_against_workflows(
        knowns=knowns,
        requested_outputs=unknowns,
    )
    recommendation = analysis["recommendation"]
    prototype = prototype_supported_scenarios(knowns)

    if knowns:
        known_lines = []
        for name, value in knowns.items():
            description = VARIABLE_DESCRIPTIONS.get(name, name)
            if isinstance(value, float):
                if name in {"W0", "B", "D"}:
                    value_text = f"{value:.3f}"
                else:
                    value_text = f"{value:.6f}"
            else:
                value_text = str(value)
            known_lines.append(f"- {name} = {value_text} ({description})")
        knowns_block = "Known inputs so far:\n" + "\n".join(known_lines)
    else:
        knowns_block = "Known inputs so far:\n- none yet"

    if analysis["ready_workflows"]:
        status_intro = (
            "You already have enough information for at least one supported deterministic workflow."
        )
    elif "D" in knowns and "xDavg_target" in knowns and "W0" not in knowns and "x0" not in knowns:
        status_intro = (
            "Your request is still underdetermined: D and xDavg_target alone do not uniquely determine W0 and x0."
        )
    else:
        status_intro = (
            "I can compare your current knowns against the supported workflows and suggest the next useful design basis."
        )

    relevant_workflow_lines = []
    for workflow in analysis["closest_workflows"]:
        missing_text = ", ".join(workflow["missing_inputs"]) if workflow["missing_inputs"] else "none"
        relevant_workflow_lines.append(
            f"- {workflow['label']}: requires {', '.join(workflow['required_inputs'])}; missing now: {missing_text}."
        )

    scenario_sections = []
    if prototype["avg_distillate_scenarios"]:
        lines = ["Illustrative target-average-distillate scenarios:"]
        for scenario in prototype["avg_distillate_scenarios"]:
            lines.append(
                "- xDavg_target={xDavg_target:.4f} -> D={D:.3f} mol, B={B:.3f} mol, xB={xB:.6f}".format(
                    **scenario
                )
            )
        scenario_sections.append("\n".join(lines))

    if prototype["final_still_scenarios"]:
        lines = ["Illustrative target-final-still scenarios:"]
        for scenario in prototype["final_still_scenarios"]:
            lines.append(
                "- xB={xB:.4f} -> D={D:.3f} mol, B={B:.3f} mol, xDavg={xDavg:.6f}".format(
                    **scenario
                )
            )
        scenario_sections.append("\n".join(lines))

    if prototype["notes"]:
        scenario_sections.append("Notes:\n" + "\n".join(f"- {note}" for note in prototype["notes"]))

    has_scenario_results = bool(
        prototype["avg_distillate_scenarios"] or prototype["final_still_scenarios"]
    )

    if scenario_sections:
        scenario_block = "\n\n".join(scenario_sections)
        if has_scenario_results:
            scenario_block += "\n\nThese scenario results are illustrative, not final design recommendations."
    else:
        scenario_block = ""

    final_answer = (
        status_intro
        + "\n\n"
        + knowns_block
        + "\n\nRelevant supported workflows:\n"
        + "\n".join(relevant_workflow_lines)
        + "\n\n"
        + recommendation["explanation"]
    )

    if scenario_block:
        final_answer += "\n\n" + scenario_block

    final_answer += "\n\n" + recommendation["recommended_next_question"]

    return {
        "guidance_response": final_answer,
        "final_answer": final_answer,
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
                        "Missing required inputs for check_batch_consistency: "
                        + ", ".join(missing_keys)
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
            "errors": [f"Unsupported problem_type: {problem_type}"],
            "warnings": [],
        }

    except Exception as exc:
        return {
            "calculation_success": False,
            "result": {},
            "consistency_check": {},
            "errors": [str(exc)],
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
        return {
            "final_answer": (
                "I could not complete the calculation.\n\n"
                f"Errors: {errors}"
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
        f"- Initial charge, W0 = {W0:.3f} mol",
        f"- Initial ethanol mole fraction, x0 = {x0:.6f}",
    ]

    if "xDavg_target" in result:
        input_summary_lines.append(
            f"- Target average distillate ethanol mole fraction = {result['xDavg_target']:.6f}"
        )
    else:
        input_summary_lines.append(
            f"- Final still ethanol mole fraction target, xB = {xB:.6f}"
        )

    final_answer = (
        "For the batch distillation problem:\n\n"
        + "\n".join(input_summary_lines)
        + "\n\nCalculated result:\n\n"
        + f"- Distillate collected, D = {D:.3f} mol\n"
        + f"- Final still amount, B = {B:.3f} mol\n"
        + f"- Final still ethanol mole fraction, xB = {xB:.6f}\n"
        + f"- Average distillate ethanol mole fraction, xDavg = {xDavg:.6f}\n\n"
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

    # Partial-knowns and underdetermined design requests go to the design advisor.
    if intent_type == "design_prototyping":
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
