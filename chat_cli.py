from agents.graph import build_graph


def main() -> None:
    app = build_graph()
    session_knowns: dict[str, float] = {}
    awaiting_clarification = False
    clarification_question: str | None = None

    print("Batch distillation assistant")
    print("Ask a simple batch distillation question, or type exit, quit, or q to stop.")

    while True:
        user_message = input("\nYou: ").strip()

        if user_message.lower() in {"exit", "quit", "q"}:
            print("Goodbye.")
            break

        if not user_message:
            continue

        try:
            graph_input = {"user_message": user_message}
            if session_knowns or awaiting_clarification or clarification_question:
                graph_input["prior_knowns"] = session_knowns
                graph_input["prior_needs_clarification"] = awaiting_clarification
                graph_input["prior_clarification_question"] = clarification_question

            final_state = app.invoke(graph_input)
            session_knowns = final_state.get("knowns", {})
            awaiting_clarification = final_state.get("needs_clarification", False)
            clarification_question = final_state.get("clarification_question")

            if not awaiting_clarification and final_state.get("intent_type") not in {
                "design_prototyping",
                "open_ended_guidance",
                "underdetermined_design",
                "conceptual_question",
            }:
                session_knowns = {}
                clarification_question = None

            print(f"\nAssistant: {final_state['final_answer']}")
        except Exception as exc:
            print(f"\nAssistant error: {exc}")


if __name__ == "__main__":
    main()
