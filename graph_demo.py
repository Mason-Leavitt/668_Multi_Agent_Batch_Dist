import os
from textwrap import dedent
from typing import Any, TypedDict, Literal

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field

# Import your deterministic tool.
# Adjust this import depending on your file names.
from tools import solve_D_given_W0_x0_xDavg, check_batch_consistency

load_dotenv()

ProblemType = Literal[
    "solve_D_given_W0_x0_xDavg",
    "solve_batch_given_W0_x0_xB",
    "check_batch_consistency",
    "unknown",
]


class StructuredProblemRequest(BaseModel):
    problem_type: ProblemType
    knowns: dict[str, float] = Field(default_factory=dict)
    unknowns: list[str] = Field(default_factory=list)
    needs_clarification: bool
    clarification_question: str | None = None


class StructuredKnowns(BaseModel):
    W0: float | None = None
    x0: float | None = None
    xB: float | None = None
    xDavg_target: float | None = None
    xDavg: float | None = None
    B: float | None = None
    D: float | None = None


class StructuredProblemRequestLLM(BaseModel):
    problem_type: ProblemType
    knowns: StructuredKnowns = Field(default_factory=StructuredKnowns)
    unknowns: list[str] = Field(default_factory=list)
    needs_clarification: bool
    clarification_question: str | None = None


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
    problem_type: ProblemType
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
        StructuredProblemRequestLLM,
        method="function_calling",
    )

    prompt = dedent(
        f"""
        You are the ProblemStructurer node for a simple educational batch
        distillation assistant.

        Your job is to convert the user's request into structured data for the
        deterministic calculation tools. Do not perform any engineering
        calculations.

        Supported problem_type values:
        - solve_D_given_W0_x0_xDavg
        - solve_batch_given_W0_x0_xB
        - check_batch_consistency
        - unknown

        Interpretation rules:
        - "5 mol%" means 0.05 mole fraction.
        - "20 mol%" means 0.20 mole fraction.
        - "5 mole percent" means 0.05 mole fraction.
        - "20 mole percent" means 0.20 mole fraction.
        - If the user says only "percent" without specifying mole percent,
          weight percent, or volume percent, treat it as ambiguous and ask for
          clarification.
        - Do not treat ABV, volume percent, or weight percent as mole fraction.
        - If enough information is missing to select and populate a supported
          calculation, return:
          problem_type="unknown"
          needs_clarification=True
          clarification_question=<concise question>

        Mapping guidance:
        - If the user gives W0, x0, and a target average distillate composition
          and asks how much distillate can be collected, use
          solve_D_given_W0_x0_xDavg.
        - If the user gives W0, x0, and xB, use solve_batch_given_W0_x0_xB.
        - If the user asks to verify consistency and provides W0, B, D, x0, xB,
          and xDavg, use check_batch_consistency.

        Keep knowns numeric. Keep unknowns as variable names. Keep the
        clarification question concise.

        Include known numeric values inside the knowns object using these field
        names when applicable:
        - W0
        - x0
        - xB
        - xDavg_target
        - xDavg
        - B
        - D

        For this kind of request:
        "I have 1000 mol of ethanol-water at 5 mol% ethanol. I want the average
        distillate to be 20 mol% ethanol. How much distillate can I collect?"
        the correct mapping is:
        - problem_type = solve_D_given_W0_x0_xDavg
        - knowns.W0 = 1000.0
        - knowns.x0 = 0.05
        - knowns.xDavg_target = 0.20
        - unknowns includes D, B, and xB
        - needs_clarification = False

        User message:
        {user_message}
        """
    ).strip()

    structured = structured_llm.invoke(prompt)
    structured_dict = StructuredProblemRequest(
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
