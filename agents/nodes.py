import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from agents.prompts import build_problem_structurer_prompt
from agents.schemas import LLMProblemRequest, ProblemRequest
from agents.state import BatchDistillationState
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

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is missing. Add it to .env before running the LLM ProblemStructurer."
        )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    structured_llm = llm.with_structured_output(
        LLMProblemRequest,
        method="function_calling",
    )

    structured = structured_llm.invoke(build_problem_structurer_prompt(user_message))
    structured_dict = ProblemRequest(
        problem_type=structured.problem_type,
        knowns=structured.knowns.model_dump(exclude_none=True),
        unknowns=structured.unknowns,
        needs_clarification=structured.needs_clarification,
        clarification_question=structured.clarification_question,
    ).model_dump()
    print(
        "ProblemStructurer:",
        {
            "problem_type": structured_dict["problem_type"],
            "knowns": structured_dict["knowns"],
            "unknowns": structured_dict["unknowns"],
            "needs_clarification": structured_dict["needs_clarification"],
        },
    )

    return structured_dict


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
    if state.get("needs_clarification", False):
        return "result_explainer"

    return "validation_calculation"
