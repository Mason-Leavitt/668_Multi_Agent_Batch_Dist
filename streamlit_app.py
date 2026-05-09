import streamlit as st

from agents.error_handling import normalize_error_for_user
from agents.graph import build_graph
from agents.session_commands import apply_session_command, parse_session_command


def initialize_session_state() -> None:
    if "graph_app" not in st.session_state:
        st.session_state.graph_app = build_graph()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "prior_knowns" not in st.session_state:
        st.session_state.prior_knowns = {}

    if "prior_needs_clarification" not in st.session_state:
        st.session_state.prior_needs_clarification = False

    if "prior_clarification_question" not in st.session_state:
        st.session_state.prior_clarification_question = None

    if "active_experiment" not in st.session_state:
        st.session_state.active_experiment = None

    if "experiment_results" not in st.session_state:
        st.session_state.experiment_results = None

    if "experiment_sampled_variable" not in st.session_state:
        st.session_state.experiment_sampled_variable = None

    if "experiment_knowns" not in st.session_state:
        st.session_state.experiment_knowns = None

    if "experiment_status" not in st.session_state:
        st.session_state.experiment_status = None


def reset_conversation() -> None:
    # Reset clears the remembered session state and visible chat history together.
    st.session_state.messages = []
    st.session_state.prior_knowns = {}
    st.session_state.prior_needs_clarification = False
    st.session_state.prior_clarification_question = None
    st.session_state.active_experiment = None
    st.session_state.experiment_results = None
    st.session_state.experiment_sampled_variable = None
    st.session_state.experiment_knowns = None
    st.session_state.experiment_status = None


def main() -> None:
    st.set_page_config(page_title="Batch Distillation Assistant")
    initialize_session_state()

    st.title("Batch Distillation Assistant")
    st.write(
        "A simple LangGraph multi-agent assistant using deterministic batch distillation tools."
    )

    with st.sidebar:
        st.button("Reset conversation", on_click=reset_conversation, use_container_width=True)
        st.subheader("Workflow")
        st.markdown(
            "- `Face`\n"
            "- `ProblemStructurer`\n"
            "- `GuidanceResponder`\n"
            "- `DesignAdvisor`\n"
            "- `ValidationCalculation`\n"
            "- `ResultExplainer`"
        )
        st.caption(
            "The LLM structures the problem, but deterministic Python tools perform the calculations."
        )
        st.caption(
            "Scenario results are illustrative examples for comparing design choices."
        )
        with st.expander("Remembered knowns", expanded=True):
            st.json(st.session_state.prior_knowns or {})
        if st.session_state.prior_needs_clarification:
            st.caption(
                f"Waiting for clarification: {st.session_state.prior_clarification_question}"
            )
        if st.session_state.active_experiment and st.session_state.experiment_sampled_variable:
            st.caption(
                "Active experiment: "
                f"sampling {st.session_state.experiment_sampled_variable} "
                f"with {len(st.session_state.experiment_results or [])} stored row(s)"
            )
        with st.expander("Example prompts", expanded=False):
            st.markdown(
                '- "I have 1000 mol of ethanol-water at 5 mol% ethanol. I want the average distillate to be 20 mol% ethanol. How much distillate can I collect?"'
            )
            st.markdown(
                '- "I start with 1000 mol of ethanol-water at 5 mol% ethanol and distill until the still is 1 mol% ethanol. How much distillate do I collect?"'
            )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_message = st.chat_input("Ask a batch distillation question")
    if not user_message:
        return

    st.session_state.messages.append({"role": "user", "content": user_message})
    with st.chat_message("user"):
        st.markdown(user_message)

    try:
        session_command = parse_session_command(user_message)
        if session_command["is_session_command"]:
            session_update = apply_session_command(
                command=session_command,
                prior_knowns=st.session_state.prior_knowns,
                prior_needs_clarification=st.session_state.prior_needs_clarification,
                prior_clarification_question=st.session_state.prior_clarification_question,
                active_experiment=st.session_state.active_experiment,
                experiment_results=st.session_state.experiment_results,
                experiment_sampled_variable=st.session_state.experiment_sampled_variable,
                experiment_knowns=st.session_state.experiment_knowns,
                experiment_status=st.session_state.experiment_status,
            )
            st.session_state.prior_knowns = session_update["prior_knowns"]
            st.session_state.prior_needs_clarification = session_update[
                "prior_needs_clarification"
            ]
            st.session_state.prior_clarification_question = session_update[
                "prior_clarification_question"
            ]
            st.session_state.active_experiment = session_update["active_experiment"]
            st.session_state.experiment_results = session_update["experiment_results"]
            st.session_state.experiment_sampled_variable = session_update[
                "experiment_sampled_variable"
            ]
            st.session_state.experiment_knowns = session_update["experiment_knowns"]
            st.session_state.experiment_status = session_update["experiment_status"]
            assistant_message = session_update["message"]
        else:
            graph_input = {"user_message": user_message}
            if (
                st.session_state.prior_knowns
                or st.session_state.prior_needs_clarification
                or st.session_state.prior_clarification_question
            ):
                graph_input["prior_knowns"] = st.session_state.prior_knowns
                graph_input["prior_needs_clarification"] = (
                    st.session_state.prior_needs_clarification
                )
                graph_input["prior_clarification_question"] = (
                    st.session_state.prior_clarification_question
                )
            if (
                st.session_state.active_experiment
                or st.session_state.experiment_results
                or st.session_state.experiment_knowns
                or st.session_state.experiment_sampled_variable
            ):
                graph_input["active_experiment"] = st.session_state.active_experiment
                graph_input["experiment_results"] = st.session_state.experiment_results
                graph_input["experiment_sampled_variable"] = (
                    st.session_state.experiment_sampled_variable
                )
                graph_input["experiment_knowns"] = st.session_state.experiment_knowns
                graph_input["experiment_status"] = st.session_state.experiment_status

            final_state = st.session_state.graph_app.invoke(graph_input)
            assistant_message = final_state["final_answer"]
            st.session_state.prior_knowns = final_state.get("knowns", {})
            st.session_state.prior_needs_clarification = final_state.get(
                "needs_clarification", False
            )
            st.session_state.prior_clarification_question = final_state.get(
                "clarification_question"
            )
            st.session_state.active_experiment = final_state.get("active_experiment")
            st.session_state.experiment_results = final_state.get("experiment_results")
            st.session_state.experiment_sampled_variable = final_state.get(
                "experiment_sampled_variable"
            )
            st.session_state.experiment_knowns = final_state.get("experiment_knowns")
            st.session_state.experiment_status = final_state.get("experiment_status")
            has_active_experiment_context = bool(
                st.session_state.active_experiment
                or st.session_state.experiment_results
                or st.session_state.experiment_knowns
                or st.session_state.experiment_sampled_variable
            )

            if (
                not st.session_state.prior_needs_clarification
                and not has_active_experiment_context
                and final_state.get("intent_type")
                not in {
                    "design_prototyping",
                    "open_ended_guidance",
                    "conceptual_question",
                }
            ):
                st.session_state.prior_knowns = {}
                st.session_state.prior_clarification_question = None
                st.session_state.active_experiment = None
                st.session_state.experiment_results = None
                st.session_state.experiment_sampled_variable = None
                st.session_state.experiment_knowns = None
                st.session_state.experiment_status = None

            if final_state.get("calculation_success") is True:
                st.session_state.active_experiment = None
                st.session_state.experiment_results = None
                st.session_state.experiment_sampled_variable = None
                st.session_state.experiment_knowns = None
                st.session_state.experiment_status = None
    except Exception as exc:
        assistant_message = normalize_error_for_user(exc)

    st.session_state.messages.append({"role": "assistant", "content": assistant_message})
    with st.chat_message("assistant"):
        st.markdown(assistant_message)


if __name__ == "__main__":
    main()
