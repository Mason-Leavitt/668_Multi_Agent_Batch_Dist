from graph_demo import build_graph


HAPPY_PATH_1_MESSAGE = (
    "I have 1000 mol of ethanol-water at 5 mol% ethanol. "
    "I want the average distillate to be 20 mol% ethanol. "
    "How much distillate can I collect?"
)

HAPPY_PATH_2_MESSAGE = (
    "I start with 1000 mol of ethanol-water at 5 mol% ethanol and distill "
    "until the still is 1 mol% ethanol. How much distillate do I collect?"
)

CLARIFICATION_PATH_MESSAGE = (
    "I have 1000 mol of ethanol-water at 5 mol% ethanol. "
    "How much distillate can I collect?"
)


def assert_result_keys(state: dict) -> None:
    result = state["result"]
    for key in ("W0", "B", "D", "x0", "xB", "xDavg"):
        assert key in result, f"Missing result key: {key}"

    consistency_check = state["consistency_check"]
    assert consistency_check["is_fully_consistent"] is True


def main() -> None:
    app = build_graph()

    happy_path_1 = app.invoke({"user_message": HAPPY_PATH_1_MESSAGE})
    assert happy_path_1["problem_type"] == "solve_D_given_W0_x0_xDavg"
    assert happy_path_1["calculation_success"] is True
    assert_result_keys(happy_path_1)

    happy_path_2 = app.invoke({"user_message": HAPPY_PATH_2_MESSAGE})
    assert happy_path_2["problem_type"] == "solve_batch_given_W0_x0_xB"
    assert happy_path_2["calculation_success"] is True
    assert_result_keys(happy_path_2)

    clarification_path = app.invoke({"user_message": CLARIFICATION_PATH_MESSAGE})
    assert clarification_path["needs_clarification"] is True
    assert "final_answer" in clarification_path
    assert clarification_path["final_answer"].strip()

    print("Smoke test passed: multiple graph paths completed successfully.")
    print(
        "Happy path 1: D={D:.3f}, B={B:.3f}, xB={xB:.6f}, xDavg={xDavg:.6f}".format(
            D=happy_path_1["result"]["D"],
            B=happy_path_1["result"]["B"],
            xB=happy_path_1["result"]["xB"],
            xDavg=happy_path_1["result"]["xDavg"],
        )
    )
    print(
        "Happy path 2: D={D:.3f}, B={B:.3f}, xB={xB:.6f}, xDavg={xDavg:.6f}".format(
            D=happy_path_2["result"]["D"],
            B=happy_path_2["result"]["B"],
            xB=happy_path_2["result"]["xB"],
            xDavg=happy_path_2["result"]["xDavg"],
        )
    )
    print(f"Clarification path: {clarification_path['final_answer']}")


if __name__ == "__main__":
    main()
