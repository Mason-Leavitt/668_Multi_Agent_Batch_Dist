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
PLANNING_D_XDAVG_ONLY_MESSAGE = "I want 20 mol of distillate at xDavg 0.2."
PLANNING_D_XDAVG_X0_MESSAGE = (
    "I want 20 mol of distillate at xDavg 0.2 and my feed x0 is 0.05."
)
PLANNING_W0_X0_OPTIONS_MESSAGE = "I have 1000 mol at x0 0.05 and want to compare options."
EXPLAIN_FOLLOWUP_MESSAGE = "I don't understand. Explain the options."


def assert_result_keys(state: dict) -> None:
    result = state["result"]
    for key in ("W0", "B", "D", "x0", "xB", "xDavg"):
        assert key in result, f"Missing result key: {key}"

    consistency_check = state["consistency_check"]
    assert consistency_check["is_fully_consistent"] is True


def pass_check(label: str) -> None:
    print(f"[PASS] {label}")


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
    pass_check("user-safe incomplete calculation handling")

    happy_path_1 = app.invoke({"user_message": HAPPY_PATH_1_MESSAGE})
    assert happy_path_1["problem_type"] == "solve_D_given_W0_x0_xDavg"
    assert happy_path_1["calculation_success"] is True
    assert_result_keys(happy_path_1)
    pass_check("happy path: target average distillate")

    happy_path_2 = app.invoke({"user_message": HAPPY_PATH_2_MESSAGE})
    assert happy_path_2["problem_type"] == "solve_batch_given_W0_x0_xB"
    assert happy_path_2["calculation_success"] is True
    assert_result_keys(happy_path_2)
    pass_check("happy path: target final still composition")

    clarification_path = app.invoke({"user_message": CLARIFICATION_PATH_MESSAGE})
    assert clarification_path["needs_clarification"] is True
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
    pass_check("clarification follow-up")

    open_ended_guidance = app.invoke({"user_message": OPEN_ENDED_GUIDANCE_MESSAGE})
    assert open_ended_guidance["final_answer"].strip()
    guidance_text = open_ended_guidance["final_answer"].lower()
    assert (
        open_ended_guidance.get("intent_type") == "open_ended_guidance"
        or "supported workflows" in guidance_text
    )
    assert "average distillate" in guidance_text
    assert "final still" in guidance_text
    pass_check("open-ended guidance")

    how_do_i_start = app.invoke({"user_message": HOW_DO_I_START_MESSAGE})
    how_to_start_text = how_do_i_start["final_answer"].lower()
    assert "workflow" in how_to_start_text or "target average distillate" in how_to_start_text
    assert "w0" in how_to_start_text
    assert "x0" in how_to_start_text
    pass_check("broad start guidance")

    partial_knowns_targets = app.invoke({"user_message": PARTIAL_KNOWNS_TARGETS_MESSAGE})
    partial_text = partial_knowns_targets["final_answer"].lower()
    assert "1000.000" in partial_knowns_targets["final_answer"]
    assert "0.050000" in partial_knowns_targets["final_answer"]
    assert "xdavg_target" in partial_text or "xb" in partial_text
    assert "illustrative" in partial_text or "compare design choices" in partial_text
    pass_check("partial-known design planning")

    underdetermined_design = app.invoke({"user_message": UNDERDETERMINED_DESIGN_MESSAGE})
    underdetermined_text = underdetermined_design["final_answer"].lower()
    assert underdetermined_design.get("intent_type") == "design_prototyping"
    assert "50" in underdetermined_design["final_answer"]
    assert "0.2" in underdetermined_design["final_answer"]
    assert "x0" in underdetermined_text or "feed composition" in underdetermined_text
    assert "xb" in underdetermined_text or "final still" in underdetermined_text
    assert "what is w0" not in underdetermined_text
    pass_check("underdetermined design guidance")

    explain_followup = app.invoke(
        {
            "user_message": EXPLAIN_FOLLOWUP_MESSAGE,
            "prior_knowns": underdetermined_design["knowns"],
            "prior_needs_clarification": underdetermined_design["needs_clarification"],
            "prior_clarification_question": underdetermined_design["clarification_question"],
        }
    )
    explain_text = explain_followup["final_answer"].lower()
    assert "known inputs so far" in explain_text or "relevant supported workflows" in explain_text
    assert "x0" in explain_text
    pass_check("explanation follow-up")

    underdetermined_followup = app.invoke(
        {
            "user_message": UNDERDETERMINED_DESIGN_FOLLOWUP_MESSAGE,
            "prior_knowns": underdetermined_design["knowns"],
            "prior_needs_clarification": underdetermined_design["needs_clarification"],
            "prior_clarification_question": underdetermined_design["clarification_question"],
        }
    )
    followup_text = underdetermined_followup["final_answer"].lower()
    assert "do not uniquely determine" in followup_text or "underdetermined" in followup_text
    assert "x0" in followup_text or "feed composition" in followup_text
    assert "xb" in followup_text or "final still" in followup_text
    assert "what is w0" not in followup_text
    pass_check("underdetermined follow-up guidance")

    design_prototyping = app.invoke({"user_message": DESIGN_PROTOTYPING_MESSAGE})
    design_text = design_prototyping["final_answer"].lower()
    assert design_prototyping.get("intent_type") == "design_prototyping"
    assert "do not uniquely determine" in design_text or "underdetermined" in design_text
    assert "sample" in design_text or "design basis" in design_text
    pass_check("design prototyping guidance")

    planning_d_xdavg_only = app.invoke({"user_message": PLANNING_D_XDAVG_ONLY_MESSAGE})
    planning_only_text = planning_d_xdavg_only["final_answer"].lower()
    assert "20" in planning_d_xdavg_only["final_answer"]
    assert "0.2" in planning_d_xdavg_only["final_answer"]
    assert "x0" in planning_only_text or "feed composition" in planning_only_text
    assert "xb" in planning_only_text or "final still" in planning_only_text
    assert "what is w0" not in planning_only_text
    pass_check("planning axis choice: D + xDavg_target")

    planning_d_xdavg_x0 = app.invoke({"user_message": PLANNING_D_XDAVG_X0_MESSAGE})
    planning_x0_text = planning_d_xdavg_x0["final_answer"].lower()
    assert "20" in planning_d_xdavg_x0["final_answer"]
    assert "0.2" in planning_d_xdavg_x0["final_answer"]
    assert "0.05" in planning_d_xdavg_x0["final_answer"]
    assert "xb=" in planning_d_xdavg_x0["final_answer"].lower() or "final still ethanol mole fraction (xb)" in planning_x0_text
    assert "w0=" in planning_d_xdavg_x0["final_answer"].lower() or "w0" in planning_x0_text
    assert "keyerror" not in planning_x0_text
    assert "traceback" not in planning_x0_text
    pass_check("scenario sampling: D + xDavg_target + x0")

    planning_w0_x0_options = app.invoke({"user_message": PLANNING_W0_X0_OPTIONS_MESSAGE})
    planning_options_text = planning_w0_x0_options["final_answer"].lower()
    assert "1000.000" in planning_w0_x0_options["final_answer"]
    assert "0.050000" in planning_w0_x0_options["final_answer"]
    assert "xdavg_target" in planning_options_text or "average distillate" in planning_options_text
    assert "xb" in planning_options_text or "final still" in planning_options_text
    assert "illustrative" in planning_options_text or "compare design choices" in planning_options_text
    pass_check("scenario sampling: W0 + x0 options")

    print("Smoke test passed.")


if __name__ == "__main__":
    main()
