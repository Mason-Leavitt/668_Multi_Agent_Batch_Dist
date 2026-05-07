from typing import Any, TypedDict, Literal

from langgraph.graph import StateGraph, START, END

# Import your deterministic tool.
# Adjust this import depending on your file names.
from tools import solve_D_given_W0_x0_xDavg, check_batch_consistency


class BatchDistillationState(TypedDict, total=False):
    """
    Shared state passed through the LangGraph workflow.

    Each node receives this state and returns a partial update.
    """

    # Original user input
    user_message: str

    # Face node output
    user_goal: str

    # ProblemStructurer output
    problem_type: Literal[
        "solve_D_given_W0_x0_xDavg",
        "solve_batch_given_W0_x0_xB",
        "unknown",
    ]
    knowns: dict[str, float]
    unknowns: list[str]
    needs_clarification: bool
    clarification_question: str | None

    # ValidationCalculation output
    calculation_success: bool
    result: dict[str, Any]
    consistency_check: dict[str, Any]
    errors: list[str]
    warnings: list[str]

    # ResultExplainer output
    final_answer: str


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

    For now, this is hardcoded for your test example.
    Later, this becomes an LLM structured-output node.
    """
    return {
        "problem_type": "solve_D_given_W0_x0_xDavg",
        "knowns": {
            "W0": 1000.0,
            "x0": 0.05,
            "xDavg_target": 0.20,
        },
        "unknowns": ["D", "B", "xB"],
        "needs_clarification": False,
        "clarification_question": None,
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
                "I need more information before I can calculate the result."
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

    final_answer = f"""
For the batch distillation problem:

- Initial charge, W0 = {W0:.3f} mol
- Initial ethanol mole fraction, x0 = {x0:.6f}
- Target average distillate ethanol mole fraction = {result["xDavg_target"]:.6f}

Calculated result:

- Distillate collected, D = {D:.3f} mol
- Final still amount, B = {B:.3f} mol
- Final still ethanol mole fraction, xB = {xB:.6f}
- Average distillate ethanol mole fraction, xDavg = {xDavg:.6f}

Consistency check:

- Fully consistent: {check.get("is_fully_consistent")}
- Total balance consistent: {check.get("is_total_balance_consistent")}
- Component balance consistent: {check.get("is_component_balance_consistent")}
- Rayleigh equation consistent: {check.get("is_rayleigh_consistent")}
""".strip()

    return {
        "final_answer": final_answer
    }


def route_after_problem_structurer(state: BatchDistillationState) -> str:
    if state.get("needs_clarification", False):
        return "result_explainer"

    return "validation_calculation"


def build_graph():
    """
    Build and compile the LangGraph workflow.
    """
    graph = StateGraph(BatchDistillationState)

    graph.add_node("face", face_node)
    graph.add_node("problem_structurer", problem_structurer_node)
    graph.add_node("validation_calculation", validation_calculation_node)
    graph.add_node("result_explainer", result_explainer_node)

    graph.add_edge(START, "face")
    graph.add_edge("face", "problem_structurer")

    graph.add_conditional_edges(
        "problem_structurer",
        route_after_problem_structurer,
        {
            "validation_calculation": "validation_calculation",
            "result_explainer": "result_explainer",
        },
    )

    graph.add_edge("validation_calculation", "result_explainer")
    graph.add_edge("result_explainer", END)

    return graph.compile()


if __name__ == "__main__":
    app = build_graph()

    final_state = app.invoke({
        "user_message": (
            "I have 1000 mol of ethanol-water at 5 mol% ethanol. "
            "I want the average distillate to be 20 mol% ethanol. "
            "How much distillate can I collect?"
        )
    })

    print(final_state["final_answer"])
