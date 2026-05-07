from agents.graph import build_graph


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
