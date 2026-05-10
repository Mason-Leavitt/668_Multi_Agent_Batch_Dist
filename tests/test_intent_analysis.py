import unittest

from app.ui_metadata import get_core_batch_variable_keys, get_variable_metadata, get_workflow_metadata
from agents.intent_analysis import (
    analyze_batch_capabilities,
    analyze_intent_capabilities,
    build_pending_incomplete_followup_response,
    is_incomplete_workflow_followup,
)
from agents.interface_agent import _determine_missing_inputs
from agents.schemas import GoalClassification
from agents.workflow_schemas import WorkflowPlan


class IntentAnalysisTests(unittest.TestCase):
    def test_variable_registry_contains_core_batch_variables(self) -> None:
        core_keys = get_core_batch_variable_keys()
        self.assertIn("W0", core_keys)
        self.assertIn("x0", core_keys)
        self.assertIn("xDavg", core_keys)
        self.assertIn("xB", core_keys)
        self.assertEqual(core_keys[:4], ["W0", "x0", "D", "xDavg"])
        self.assertEqual(get_variable_metadata("W")["key"], "B")

    def test_workflow_registry_contains_sweep_workflows(self) -> None:
        self.assertEqual(get_workflow_metadata("product_to_feed_sweep")["workflow_id"], "product_to_feed_sweep")
        self.assertEqual(get_workflow_metadata("feed_to_product_sweep")["workflow_id"], "feed_to_product_sweep")

    def test_rayleigh_w0_x0_xdavg_is_domain_relevant_but_not_ready(self) -> None:
        classification = GoalClassification(
            goal="solve_rayleigh_batch_variables",
            output_mode="numeric_answer",
            known_inputs=["W0", "x0", "xDavg"],
            requested_outputs=["xB", "B", "D"],
            missing_inputs=["xB"],
            variable_assignments={
                "W0": "100 L",
                "x0": "10% ABV",
                "xDavg": "40% ABV",
            },
            input_format="mixed_units",
            output_format="user_friendly",
            requires_input_conversion=True,
            requires_output_conversion=True,
            confidence=0.84,
            reasoning_summary="test",
            user_facing_summary="test",
        )
        plan = WorkflowPlan(
            workflow_name="solve_rayleigh_batch_variables",
            ready_to_execute=False,
            required_inputs=[],
            available_inputs=["W0", "x0", "xDavg"],
            missing_inputs=["xB"],
            normalization_steps=[],
            calculation_steps=[],
            result_steps=[],
            warnings=[],
            suggested_next_message="test",
        )

        analysis = analyze_intent_capabilities(
            "given W0 = 100 L, x0 = 10% ABV, and xDavg = 40% ABV, what can you calculate?",
            classification=classification,
            plan=plan,
        )

        self.assertTrue(analysis.domain_relevant)
        self.assertFalse(analysis.can_calculate_now)
        self.assertIn("W0", analysis.known_quantities)
        self.assertIn("x0", analysis.known_quantities)
        self.assertIn("xDavg", analysis.known_quantities)
        self.assertTrue(any("xB" in item for item in analysis.missing_but_needed))
        self.assertNotIn("unsupported_or_unclear", analysis.candidate_goals)
        self.assertIsNotNone(analysis.best_next_question)
        self.assertIsNotNone(analysis.capability_analysis)
        assert analysis.capability_analysis is not None
        self.assertGreaterEqual(len(analysis.capability_analysis.nearly_possible_calculations), 1)

    def test_pending_incomplete_followup_names_missing_values(self) -> None:
        classification = GoalClassification(
            goal="solve_rayleigh_batch_variables",
            output_mode="numeric_answer",
            known_inputs=["W0", "x0", "xDavg"],
            requested_outputs=["xB", "B", "D"],
            missing_inputs=["xB"],
            variable_assignments={
                "W0": "100 L",
                "x0": "10% ABV",
                "xDavg": "40% ABV",
            },
            input_format="mixed_units",
            output_format="user_friendly",
            requires_input_conversion=True,
            requires_output_conversion=True,
            confidence=0.84,
            reasoning_summary="test",
            user_facing_summary="test",
        )
        plan = WorkflowPlan(
            workflow_name="solve_rayleigh_batch_variables",
            ready_to_execute=False,
            required_inputs=[],
            available_inputs=["W0", "x0", "xDavg"],
            missing_inputs=["xB"],
            normalization_steps=[],
            calculation_steps=[],
            result_steps=[],
            warnings=[],
            suggested_next_message="test",
        )
        analysis = analyze_intent_capabilities(
            "given W0 = 100 L, x0 = 10% ABV, and xDavg = 40% ABV, what can you calculate?",
            classification=classification,
            plan=plan,
        )

        self.assertTrue(is_incomplete_workflow_followup("what additional values do you need?"))
        response = build_pending_incomplete_followup_response(
            "what additional values do you need?",
            analysis,
        )
        self.assertIsNotNone(response)
        assert response is not None
        self.assertIn("xB", response)
        self.assertNotIn("I need values for before I can run it", response)

    def test_pending_incomplete_followup_lists_options(self) -> None:
        classification = GoalClassification(
            goal="solve_rayleigh_batch_variables",
            output_mode="numeric_answer",
            known_inputs=["W0", "x0", "xDavg"],
            requested_outputs=["xB", "B", "D"],
            missing_inputs=["xB"],
            variable_assignments={
                "W0": "100 L",
                "x0": "10% ABV",
                "xDavg": "40% ABV",
            },
            input_format="mixed_units",
            output_format="user_friendly",
            requires_input_conversion=True,
            requires_output_conversion=True,
            confidence=0.84,
            reasoning_summary="test",
            user_facing_summary="test",
        )
        plan = WorkflowPlan(
            workflow_name="solve_rayleigh_batch_variables",
            ready_to_execute=False,
            required_inputs=[],
            available_inputs=["W0", "x0", "xDavg"],
            missing_inputs=["xB"],
            normalization_steps=[],
            calculation_steps=[],
            result_steps=[],
            warnings=[],
            suggested_next_message="test",
        )
        analysis = analyze_intent_capabilities(
            "given W0 = 100 L, x0 = 10% ABV, and xDavg = 40% ABV, what can you calculate?",
            classification=classification,
            plan=plan,
        )

        response = build_pending_incomplete_followup_response(
            "what are my options?",
            analysis,
        )
        self.assertIsNotNone(response)
        assert response is not None
        self.assertIn("Rayleigh", response)

    def test_pending_incomplete_followup_handles_sweep_option(self) -> None:
        capability = analyze_batch_capabilities(
            known_quantities={
                "W0": "100 L",
                "x0": "10% ABV",
                "xDavg": "40% ABV",
            },
            requested_outputs=["xB", "B", "D"],
            candidate_goals=["solve_rayleigh_batch_variables"],
        )
        analysis = analyze_intent_capabilities(
            "given W0 = 100 L, x0 = 10% ABV, and xDavg = 40% ABV, what can you calculate?",
            classification=GoalClassification(
                goal="solve_rayleigh_batch_variables",
                output_mode="numeric_answer",
                known_inputs=["W0", "x0", "xDavg"],
                requested_outputs=["xB", "B", "D"],
                missing_inputs=["xB"],
                variable_assignments={
                    "W0": "100 L",
                    "x0": "10% ABV",
                    "xDavg": "40% ABV",
                },
                input_format="mixed_units",
                output_format="user_friendly",
                requires_input_conversion=True,
                requires_output_conversion=True,
                confidence=0.84,
                reasoning_summary="test",
                user_facing_summary="test",
            ),
            plan=WorkflowPlan(
                workflow_name="solve_rayleigh_batch_variables",
                ready_to_execute=False,
                required_inputs=[],
                available_inputs=["W0", "x0", "xDavg"],
                missing_inputs=["xB"],
                normalization_steps=[],
                calculation_steps=[],
                result_steps=[],
                warnings=[],
                suggested_next_message="test",
            ),
        )
        analysis.capability_analysis = capability

        response = build_pending_incomplete_followup_response(
            "can you sweep it instead?",
            analysis,
        )
        self.assertIsNotNone(response)
        assert response is not None
        self.assertIn("sweep", response.lower())

    def test_out_of_domain_request_stays_non_domain_relevant(self) -> None:
        analysis = analyze_intent_capabilities(
            "write me a sonnet about rain",
            classification=None,
            plan=None,
        )
        self.assertFalse(analysis.domain_relevant)
        self.assertFalse(analysis.can_calculate_now)

    def test_existing_ready_workflow_can_calculate_now(self) -> None:
        classification = GoalClassification(
            goal="feed_to_product_sweep",
            output_mode="table",
            known_inputs=["feed_volume", "feed_abv"],
            requested_outputs=["D", "xDavg"],
            missing_inputs=[],
            variable_assignments={
                "feed_volume": "100 L",
                "feed_abv": "10% ABV",
            },
            input_format="volume_abv",
            output_format="user_friendly",
            requires_input_conversion=True,
            requires_output_conversion=True,
            confidence=0.95,
            reasoning_summary="test",
            user_facing_summary="test",
        )
        plan = WorkflowPlan(
            workflow_name="feed_to_product_sweep",
            ready_to_execute=True,
            required_inputs=["feed_volume", "feed_abv"],
            available_inputs=["feed_volume", "feed_abv"],
            missing_inputs=[],
            normalization_steps=["Convert user-friendly feed volume and ABV to internal W0 and x0."],
            calculation_steps=["Sweep xB across feasible stopping compositions."],
            result_steps=["Return requested D and xDavg combinations."],
            warnings=[],
            suggested_next_message="test",
        )

        analysis = analyze_intent_capabilities(
            "I have 100 L at 10% ABV. What product outcomes can I get?",
            classification=classification,
            plan=plan,
        )
        self.assertTrue(analysis.domain_relevant)
        self.assertTrue(analysis.can_calculate_now)

    def test_interface_agent_missing_inputs_use_shared_capability_analysis(self) -> None:
        missing = _determine_missing_inputs(
            "solve_rayleigh_batch_variables",
            ["W0", "x0", "xDavg"],
            sweep_variable=None,
            variable_assignments={
                "W0": "100 L",
                "x0": "10% ABV",
                "xDavg": "40% ABV",
            },
            requested_outputs=["xB", "B", "D"],
        )

        self.assertTrue(any("xB" in item or "D" in item or "B/W" in item for item in missing))


if __name__ == "__main__":
    unittest.main()
