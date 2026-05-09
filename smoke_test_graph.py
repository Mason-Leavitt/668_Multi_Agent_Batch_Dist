"""
LLM-integration smoke test for the current LangGraph demo workflows.

This is not a deterministic unit test. It requires OpenAI access and may fail
if the API is unavailable or the model's structured-output behavior changes.
"""

from agents.graph import build_graph
from agents.experiment_followup_interpreter import is_plausible_experiment_followup
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
USE_OPTION_2_MESSAGE = "use option 2"
TRY_XB_MESSAGE = "try xB = 0.007"
SHOW_HIGHER_XB_MESSAGE = "show higher xB values"
SHOW_LOWER_XB_MESSAGE = "show lower xB values"
COMPARE_X0_INSTEAD_MESSAGE = "compare x0 instead"
DONE_WITH_EXPERIMENT_MESSAGE = "done with this experiment"
CHOOSE_SECOND_ONE_MESSAGE = "choose the second one"
WHAT_IF_XB_MESSAGE = "what if xB is 0.007?"
VARY_FEED_COMPOSITION_MESSAGE = "vary feed composition instead"
EXPLAIN_OPTION_2_MESSAGE = "explain option 2"
CONFIRM_SELECTION_MESSAGE = "yes"
REJECT_SELECTION_MESSAGE = "no"


def assert_result_keys(state: dict) -> None:
    result = state["result"]
    for key in ("W0", "B", "D", "x0", "xB", "xDavg"):
        assert key in result, f"Missing result key: {key}"

    consistency_check = state["consistency_check"]
    assert consistency_check["is_fully_consistent"] is True


def pass_check(label: str) -> None:
    print(f"[PASS] {label}")


def experiment_context_from_state(state: dict) -> dict:
    return {
        "prior_knowns": state["knowns"],
        "active_experiment": state.get("active_experiment"),
        "experiment_results": state.get("experiment_results"),
        "experiment_sampled_variable": state.get("experiment_sampled_variable"),
        "experiment_knowns": state.get("experiment_knowns"),
        "experiment_status": state.get("experiment_status"),
        "pending_commit_variable": state.get("pending_commit_variable"),
        "pending_commit_value": state.get("pending_commit_value"),
        "pending_commit_source": state.get("pending_commit_source"),
    }


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

    use_option_2 = app.invoke(
        {
            "user_message": USE_OPTION_2_MESSAGE,
            "prior_knowns": planning_d_xdavg_x0["knowns"],
            "active_experiment": planning_d_xdavg_x0.get("active_experiment"),
            "experiment_results": planning_d_xdavg_x0.get("experiment_results"),
            "experiment_sampled_variable": planning_d_xdavg_x0.get(
                "experiment_sampled_variable"
            ),
            "experiment_knowns": planning_d_xdavg_x0.get("experiment_knowns"),
            "experiment_status": planning_d_xdavg_x0.get("experiment_status"),
        }
    )
    use_option_text = use_option_2["final_answer"].lower()
    assert "option 2" in use_option_text
    assert "xb" in use_option_text
    assert "should i use" in use_option_text
    assert use_option_2.get("pending_commit_variable") == "xB"
    assert abs((use_option_2.get("pending_commit_value") or 0.0) - 0.005) < 1e-12
    pass_check("experiment follow-up: use option 2")

    confirm_option_2 = app.invoke(
        {
            "user_message": CONFIRM_SELECTION_MESSAGE,
            **experiment_context_from_state(use_option_2),
        }
    )
    confirm_text = confirm_option_2["final_answer"].lower()
    assert "i'll use" in confirm_text or "going forward" in confirm_text
    assert abs(confirm_option_2["knowns"].get("xB", 0.0) - 0.005) < 1e-12
    assert confirm_option_2.get("pending_commit_variable") is None
    assert confirm_option_2.get("pending_commit_value") is None
    pass_check("experiment follow-up: confirm option 2")

    reject_option_2 = app.invoke(
        {
            "user_message": REJECT_SELECTION_MESSAGE,
            **experiment_context_from_state(use_option_2),
        }
    )
    reject_text = reject_option_2["final_answer"].lower()
    assert "won't use" in reject_text or "choose another" in reject_text
    assert reject_option_2.get("pending_commit_variable") is None
    assert reject_option_2.get("pending_commit_value") is None
    assert "xB" not in reject_option_2.get("knowns", {})
    pass_check("experiment follow-up: reject option 2")

    try_xb = app.invoke(
        {
            "user_message": TRY_XB_MESSAGE,
            **experiment_context_from_state(planning_d_xdavg_x0),
        }
    )
    try_xb_text = try_xb["final_answer"].lower()
    assert "0.007" in try_xb["final_answer"]
    assert "w0=" in try_xb_text or "w0" in try_xb_text
    assert "keyerror" not in try_xb_text
    assert "traceback" not in try_xb_text
    assert try_xb.get("experiment_results")
    assert any(
        row.get("custom") and abs(row.get("sampled_value", 0.0) - 0.007) < 1e-12
        for row in (try_xb.get("experiment_results") or [])
    )
    pass_check("experiment follow-up: try xB")

    explain_option_2_after_custom = app.invoke(
        {
            "user_message": EXPLAIN_OPTION_2_MESSAGE,
            **experiment_context_from_state(try_xb),
        }
    )
    explain_after_custom_text = explain_option_2_after_custom["final_answer"].lower()
    assert "option 2" in explain_after_custom_text
    assert "0.0050" in explain_option_2_after_custom["final_answer"] or "0.005" in explain_option_2_after_custom["final_answer"]
    assert "keyerror" not in explain_after_custom_text
    assert "traceback" not in explain_after_custom_text
    pass_check("experiment follow-up: explain option 2 after custom")

    show_higher_xb = app.invoke(
        {
            "user_message": SHOW_HIGHER_XB_MESSAGE,
            **experiment_context_from_state(planning_d_xdavg_x0),
        }
    )
    show_higher_text = show_higher_xb["final_answer"].lower()
    assert "higher" in show_higher_text
    assert "xb" in show_higher_text
    assert show_higher_xb.get("active_experiment")
    assert show_higher_xb.get("experiment_results")
    assert "keyerror" not in show_higher_text
    assert "traceback" not in show_higher_text
    pass_check("experiment follow-up: show higher xB")

    show_lower_xb = app.invoke(
        {
            "user_message": SHOW_LOWER_XB_MESSAGE,
            **experiment_context_from_state(planning_d_xdavg_x0),
        }
    )
    show_lower_text = show_lower_xb["final_answer"].lower()
    assert "lower" in show_lower_text
    assert "xb" in show_lower_text
    assert show_lower_xb.get("active_experiment")
    assert show_lower_xb.get("experiment_results")
    assert "keyerror" not in show_lower_text
    assert "traceback" not in show_lower_text
    pass_check("experiment follow-up: show lower xB")

    compare_x0_without_selection = app.invoke(
        {
            "user_message": COMPARE_X0_INSTEAD_MESSAGE,
            **experiment_context_from_state(planning_d_xdavg_x0),
        }
    )
    compare_x0_without_selection_text = compare_x0_without_selection["final_answer"].lower()
    assert "x0" in compare_x0_without_selection_text or "initial ethanol mole fraction" in compare_x0_without_selection_text
    assert "xb" in compare_x0_without_selection_text
    assert "which xb should i hold fixed" in compare_x0_without_selection_text or "pick one of the previous xb options" in compare_x0_without_selection_text
    assert compare_x0_without_selection.get("active_experiment")
    assert compare_x0_without_selection.get("experiment_results")
    assert "keyerror" not in compare_x0_without_selection_text
    assert "traceback" not in compare_x0_without_selection_text
    pass_check("experiment follow-up: compare x0 without selection")

    compare_x0_instead = app.invoke(
        {
            "user_message": COMPARE_X0_INSTEAD_MESSAGE,
            **experiment_context_from_state(use_option_2),
        }
    )
    compare_x0_text = compare_x0_instead["final_answer"].lower()
    assert "x0" in compare_x0_text or "initial ethanol mole fraction" in compare_x0_text
    assert "keyerror" not in compare_x0_text
    assert "traceback" not in compare_x0_text
    if compare_x0_instead.get("experiment_results"):
        assert compare_x0_instead.get("active_experiment")
    pass_check("experiment follow-up: compare x0 instead")

    done_with_experiment = app.invoke(
        {
            "user_message": DONE_WITH_EXPERIMENT_MESSAGE,
            **experiment_context_from_state(planning_d_xdavg_x0),
        }
    )
    done_text = done_with_experiment["final_answer"].lower()
    assert "done with this experiment" in done_text or "cleared" in done_text
    assert done_with_experiment.get("active_experiment") is None
    assert not done_with_experiment.get("experiment_results")
    assert done_with_experiment.get("experiment_status") is None
    pass_check("experiment follow-up: done with experiment")

    choose_second_one = app.invoke(
        {
            "user_message": CHOOSE_SECOND_ONE_MESSAGE,
            **experiment_context_from_state(planning_d_xdavg_x0),
        }
    )
    choose_second_text = choose_second_one["final_answer"].lower()
    assert "keyerror" not in choose_second_text
    assert "traceback" not in choose_second_text
    assert (
        "option 2" in choose_second_text
        or "second" in choose_second_text
        or "which option" in choose_second_text
    )
    pass_check("natural follow-up: choose second one")

    what_if_xb = app.invoke(
        {
            "user_message": WHAT_IF_XB_MESSAGE,
            **experiment_context_from_state(planning_d_xdavg_x0),
        }
    )
    what_if_xb_text = what_if_xb["final_answer"].lower()
    assert "keyerror" not in what_if_xb_text
    assert "traceback" not in what_if_xb_text
    assert "0.007" in what_if_xb["final_answer"]
    assert "xb" in what_if_xb_text
    pass_check("natural follow-up: what if xB")

    vary_feed_composition = app.invoke(
        {
            "user_message": VARY_FEED_COMPOSITION_MESSAGE,
            **experiment_context_from_state(planning_d_xdavg_x0),
        }
    )
    vary_feed_text = vary_feed_composition["final_answer"].lower()
    assert "keyerror" not in vary_feed_text
    assert "traceback" not in vary_feed_text
    assert "x0" in vary_feed_text or "initial ethanol mole fraction" in vary_feed_text
    assert "xb" in vary_feed_text or "final still" in vary_feed_text
    pass_check("natural follow-up: vary feed composition")

    explain_option_2 = app.invoke(
        {
            "user_message": EXPLAIN_OPTION_2_MESSAGE,
            **experiment_context_from_state(planning_d_xdavg_x0),
        }
    )
    explain_option_2_text = explain_option_2["final_answer"].lower()
    assert "keyerror" not in explain_option_2_text
    assert "traceback" not in explain_option_2_text
    assert "option 2" in explain_option_2_text or "which option" in explain_option_2_text
    pass_check("natural follow-up: explain option 2")

    assert is_plausible_experiment_followup("okay") is False
    pass_check("plausibility gate: generic short message")

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
