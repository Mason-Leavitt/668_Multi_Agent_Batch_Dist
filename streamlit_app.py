import streamlit as st

from agents.graph import build_graph


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


def reset_conversation() -> None:
    st.session_state.messages = []
    st.session_state.prior_knowns = {}
    st.session_state.prior_needs_clarification = False
    st.session_state.prior_clarification_question = None


def main() -> None:
    st.set_page_config(page_title="Batch Distillation Assistant")
    initialize_session_state()

    st.title("Batch Distillation Assistant")
    st.write(
        "A simple LangGraph multi-agent assistant using deterministic engineering tools."
    )

    with st.sidebar:
        st.button("Reset conversation", on_click=reset_conversation, use_container_width=True)
        with st.expander("Remembered knowns", expanded=True):
            st.json(st.session_state.prior_knowns or {})
        if st.session_state.prior_needs_clarification:
            st.caption(
                f"Waiting for clarification: {st.session_state.prior_clarification_question}"
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
        final_state = st.session_state.graph_app.invoke(
            {
                "user_message": user_message,
                "prior_knowns": st.session_state.prior_knowns,
                "prior_needs_clarification": st.session_state.prior_needs_clarification,
                "prior_clarification_question": st.session_state.prior_clarification_question,
            }
        )
        assistant_message = final_state["final_answer"]
        st.session_state.prior_knowns = final_state.get("knowns", {})
        st.session_state.prior_needs_clarification = final_state.get(
            "needs_clarification", False
        )
        st.session_state.prior_clarification_question = final_state.get(
            "clarification_question"
        )

        if not st.session_state.prior_needs_clarification:
            st.session_state.prior_knowns = {}
            st.session_state.prior_clarification_question = None
    except Exception as exc:
        assistant_message = f"Assistant error: {exc}"

    st.session_state.messages.append({"role": "assistant", "content": assistant_message})
    with st.chat_message("assistant"):
        st.markdown(assistant_message)


if __name__ == "__main__":
    main()
