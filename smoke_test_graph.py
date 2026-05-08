"""
LLM-integration smoke test for the current LangGraph demo workflows.

This is not a deterministic unit test. It requires OpenAI access and may fail
if the API is unavailable or the model's structured-output behavior changes.
"""

from agents.graph import build_graph

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
OPEN_ENDED_GUIDANCE_MESSAGE = "I don't know where to start but I want to conduct a distillation."
UNDERDETERMINED_DESIGN_MESSAGE = (
    "I want to produce about 50 moles of ethanol-water mixture distillate "
    "at a 0.2 ethanol mole fraction. How do I set up the still?"
)
UNDERDETERMINED_DESIGN_FOLLOWUP_MESSAGE = "I don't know, how much would I need?"


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

    open_ended_guidance = app.invoke({"user_message": OPEN_ENDED_GUIDANCE_MESSAGE})
    assert "final_answer" in open_ended_guidance
    assert open_ended_guidance["final_answer"].strip()
    assert (
        open_ended_guidance.get("intent_type") == "open_ended_guidance"
        or "supported workflows" in open_ended_guidance["final_answer"].lower()
    )
    guidance_text = open_ended_guidance["final_answer"].lower()
    assert "average distillate" in guidance_text
    assert "final still" in guidance_text

    underdetermined_design = app.invoke({"user_message": UNDERDETERMINED_DESIGN_MESSAGE})
    assert "final_answer" in underdetermined_design
    assert underdetermined_design["final_answer"].strip()
    underdetermined_text = underdetermined_design["final_answer"].lower()
    assert "50" in underdetermined_design["final_answer"]
    assert "0.2" in underdetermined_design["final_answer"]
    assert "x0" in underdetermined_text or "initial ethanol mole fraction" in underdetermined_text
    assert "what is w0" not in underdetermined_text

    underdetermined_followup = app.invoke(
        {
            "user_message": UNDERDETERMINED_DESIGN_FOLLOWUP_MESSAGE,
            "prior_knowns": underdetermined_design["knowns"],
            "prior_needs_clarification": underdetermined_design["needs_clarification"],
            "prior_clarification_question": underdetermined_design["clarification_question"],
        }
    )
    assert "final_answer" in underdetermined_followup
    assert underdetermined_followup["final_answer"].strip()
    followup_text = underdetermined_followup["final_answer"].lower()
    assert "cannot" in followup_text or "need one more design basis" in followup_text
    assert "x0" in followup_text or "initial ethanol mole fraction" in followup_text
    assert "xb" in followup_text or "final still composition" in followup_text
    assert "what is w0" not in followup_text

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
    print(f"Open-ended guidance: {open_ended_guidance['final_answer']}")
    print(f"Underdetermined design: {underdetermined_design['final_answer']}")
    print(f"Underdetermined follow-up: {underdetermined_followup['final_answer']}")


if __name__ == "__main__":
    main()
