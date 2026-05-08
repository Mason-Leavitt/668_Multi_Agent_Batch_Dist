from agents.graph import build_graph

# This is an LLM-integration smoke test.
# It requires OpenAI access and may fail if the API is unavailable or the
# model's structured output behavior changes.

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

CLARIFICATION_FOLLOWUP_MESSAGE = "20 mol% average distillate."


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

    clarification_followup = app.invoke(
        {
            "user_message": CLARIFICATION_FOLLOWUP_MESSAGE,
            "prior_knowns": clarification_path["knowns"],
            "prior_needs_clarification": clarification_path["needs_clarification"],
            "prior_clarification_question": clarification_path["clarification_question"],
        }
    )
    assert clarification_followup["problem_type"] == "solve_D_given_W0_x0_xDavg"
    assert clarification_followup["calculation_success"] is True
    assert_result_keys(clarification_followup)

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
    print(
        "Clarification follow-up: D={D:.3f}, B={B:.3f}, xB={xB:.6f}, xDavg={xDavg:.6f}".format(
            D=clarification_followup["result"]["D"],
            B=clarification_followup["result"]["B"],
            xB=clarification_followup["result"]["xB"],
            xDavg=clarification_followup["result"]["xDavg"],
        )
    )


if __name__ == "__main__":
    main()
