"""
LLM-integration smoke test for the current LangGraph demo workflows.

This is not a deterministic unit test. It requires OpenAI access and may fail
if the API is unavailable or the model's structured-output behavior changes.
"""

from agents.graph import build_graph
from agents.nodes import validation_calculation_node

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
HOW_DO_I_START_MESSAGE = "How do I start?"
PARTIAL_KNOWNS_TARGETS_MESSAGE = "I have 1000 mol at 5 mol% ethanol, help me choose targets."
UNDERDETERMINED_DESIGN_MESSAGE = (
    "I want to produce about 50 moles of ethanol-water mixture distillate "
    "at a 0.2 ethanol mole fraction. How do I set up the still?"
)
UNDERDETERMINED_DESIGN_FOLLOWUP_MESSAGE = "I don't know, how much would I need?"
DESIGN_PROTOTYPING_MESSAGE = (
    "I want a distillate of 50 moles at a 0.2 mole fraction of ethanol. "
    "How much initial mole mixture do I need and at what mole fraction?"
)
EXPLAIN_FOLLOWUP_MESSAGE = "I don't understand. Explain the options."


def assert_result_keys(state: dict) -> None:
    result = state["result"]
    for key in ("W0", "B", "D", "x0", "xB", "xDavg"):
        assert key in result, f"Missing result key: {key}"

    consistency_check = state["consistency_check"]
    assert consistency_check["is_fully_consistent"] is True


def main() -> None:
    app = build_graph()

    incomplete_direct_calc = validation_calculation_node(
        {
            "problem_type": "solve_D_given_W0_x0_xDavg",
            "knowns": {"xDavg_target": 0.20},
        }
    )
    assert incomplete_direct_calc["calculation_success"] is False
    assert incomplete_direct_calc["errors"]
    direct_error_text = incomplete_direct_calc["errors"][0].lower()
    assert "'w0'" not in direct_error_text
    assert "keyerror" not in direct_error_text
    assert "initial charge amount (w0)" in direct_error_text
    assert "initial ethanol mole fraction (x0)" in direct_error_text

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

    how_do_i_start = app.invoke({"user_message": HOW_DO_I_START_MESSAGE})
    assert "final_answer" in how_do_i_start
    assert how_do_i_start["final_answer"].strip()
    how_to_start_text = how_do_i_start["final_answer"].lower()
    assert "workflow" in how_to_start_text or "target average distillate" in how_to_start_text
    assert "w0" in how_to_start_text
    assert "x0" in how_to_start_text

    partial_knowns_targets = app.invoke({"user_message": PARTIAL_KNOWNS_TARGETS_MESSAGE})
    assert "final_answer" in partial_knowns_targets
    assert partial_knowns_targets["final_answer"].strip()
    partial_text = partial_knowns_targets["final_answer"].lower()
    assert "w0" in partial_text and "1000.000" in partial_knowns_targets["final_answer"]
    assert "x0" in partial_text and "0.050000" in partial_knowns_targets["final_answer"]
    assert "xdavg_target" in partial_text or "xb" in partial_text
    assert "if you want, i can also show example scenarios" in partial_text or "illustrative" in partial_text

    underdetermined_design = app.invoke({"user_message": UNDERDETERMINED_DESIGN_MESSAGE})
    assert "final_answer" in underdetermined_design
    assert underdetermined_design["final_answer"].strip()
    assert underdetermined_design.get("intent_type") == "design_prototyping"
    underdetermined_text = underdetermined_design["final_answer"].lower()
    assert "50" in underdetermined_design["final_answer"]
    assert "0.2" in underdetermined_design["final_answer"]
    assert "x0" in underdetermined_text or "initial ethanol mole fraction" in underdetermined_text
    assert "what is w0" not in underdetermined_text

    explain_followup = app.invoke(
        {
            "user_message": EXPLAIN_FOLLOWUP_MESSAGE,
            "prior_knowns": underdetermined_design["knowns"],
            "prior_needs_clarification": underdetermined_design["needs_clarification"],
            "prior_clarification_question": underdetermined_design["clarification_question"],
        }
    )
    assert "final_answer" in explain_followup
    assert explain_followup["final_answer"].strip()
    explain_text = explain_followup["final_answer"].lower()
    assert "workflow" in explain_text or "supported workflows" in explain_text or "known inputs so far" in explain_text
    assert "x0" in explain_text

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
    assert (
        "cannot" in followup_text
        or "need one more design basis" in followup_text
        or "underdetermined" in followup_text
        or "do not uniquely determine" in followup_text
    )
    assert "x0" in followup_text or "initial ethanol mole fraction" in followup_text
    assert "xb" in followup_text or "final still composition" in followup_text
    assert "what is w0" not in followup_text

    design_prototyping = app.invoke({"user_message": DESIGN_PROTOTYPING_MESSAGE})
    assert "final_answer" in design_prototyping
    assert design_prototyping["final_answer"].strip()
    design_text = design_prototyping["final_answer"].lower()
    assert design_prototyping.get("intent_type") == "design_prototyping"
    assert (
        "underdetermined" in design_text
        or "not enough" in design_text
        or "do not uniquely determine" in design_text
    )
    assert "x0" in design_text
    assert "xb" in design_text or "final still composition" in design_text
    assert "example scenarios" in design_text or "additional design basis" in design_text or "design basis" in design_text

    print("Smoke test passed: multiple graph paths completed successfully.")
    print(f"Incomplete direct calculation handling: {incomplete_direct_calc['errors'][0]}")
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
    print(f"How do I start: {how_do_i_start['final_answer']}")
    print(f"Partial knowns guidance: {partial_knowns_targets['final_answer']}")
    print(f"Underdetermined design: {underdetermined_design['final_answer']}")
    print(f"Explain follow-up: {explain_followup['final_answer']}")
    print(f"Underdetermined follow-up: {underdetermined_followup['final_answer']}")
    print(f"Design prototyping: {design_prototyping['final_answer']}")


if __name__ == "__main__":
    main()
