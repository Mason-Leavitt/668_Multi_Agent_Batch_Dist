from langgraph.graph import END, START, StateGraph

from agents.nodes import (
    face_node,
    guidance_responder_node,
    problem_structurer_node,
    result_explainer_node,
    route_after_problem_structurer,
    validation_calculation_node,
)
from agents.state import BatchDistillationState


def build_graph():
    """
    Build and compile the LangGraph workflow.
    """
    graph = StateGraph(BatchDistillationState)

    graph.add_node("face", face_node)
    graph.add_node("problem_structurer", problem_structurer_node)
    graph.add_node("guidance_responder", guidance_responder_node)
    graph.add_node("validation_calculation", validation_calculation_node)
    graph.add_node("result_explainer", result_explainer_node)

    graph.add_edge(START, "face")
    graph.add_edge("face", "problem_structurer")

    graph.add_conditional_edges(
        "problem_structurer",
        route_after_problem_structurer,
        {
            "validation_calculation": "validation_calculation",
            "guidance_responder": "guidance_responder",
        },
    )

    graph.add_edge("guidance_responder", END)
    graph.add_edge("validation_calculation", "result_explainer")
    graph.add_edge("result_explainer", END)

    return graph.compile()
