from graph_demo import build_graph


USER_MESSAGE = (
    "I have 1000 mol of ethanol-water at 5 mol% ethanol. "
    "I want the average distillate to be 20 mol% ethanol. "
    "How much distillate can I collect?"
)


def main() -> None:
    app = build_graph()
    final_state = app.invoke({"user_message": USER_MESSAGE})

    assert final_state["calculation_success"] is True

    result = final_state["result"]
    for key in ("W0", "B", "D", "x0", "xB", "xDavg"):
        assert key in result, f"Missing result key: {key}"

    consistency_check = final_state["consistency_check"]
    assert consistency_check["is_fully_consistent"] is True

    print("Smoke test passed: LangGraph demo completed successfully.")
    print(
        "D={D:.3f}, B={B:.3f}, xB={xB:.6f}, xDavg={xDavg:.6f}".format(
            D=result["D"],
            B=result["B"],
            xB=result["xB"],
            xDavg=result["xDavg"],
        )
    )


if __name__ == "__main__":
    main()
