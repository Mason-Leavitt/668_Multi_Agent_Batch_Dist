from agents.graph import build_graph


def main() -> None:
    app = build_graph()

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
            final_state = app.invoke({"user_message": user_message})
            print(f"\nAssistant: {final_state['final_answer']}")
        except Exception as exc:
            print(f"\nAssistant error: {exc}")


if __name__ == "__main__":
    main()
