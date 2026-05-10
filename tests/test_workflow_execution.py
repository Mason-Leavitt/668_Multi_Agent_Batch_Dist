import unittest

from agents.execution_schemas import WorkflowExecutionResult
from agents.interface_agent import analyze_message_features
from agents.schemas import GoalClassification
from agents.workflow_executor import execute_solve_mole_balance, execute_solve_rayleigh_batch_variables
from agents.workflow_planner import create_workflow_plan
from agents.workflow_schemas import WorkflowPlan


class WorkflowExecutionTests(unittest.TestCase):
    def test_mole_balance_execution_allows_equations_list_metadata(self) -> None:
        classification = GoalClassification(
            goal="solve_mole_balance",
            output_mode="numeric_answer",
            known_inputs=["W0", "x0", "xB"],
            requested_outputs=["D", "B", "xDavg"],
            missing_inputs=["D"],
            variable_assignments={
                "W0": "100 mole",
                "x0": "0.05 mole fraction",
                "xB": "0.001 mole fraction",
                "B": "80 mole",
            },
            input_format="model_units",
            output_format="model_units",
            requires_input_conversion=False,
            requires_output_conversion=False,
            confidence=0.9,
            reasoning_summary="test",
            user_facing_summary="test",
        )
        plan = WorkflowPlan(
            workflow_name="solve_mole_balance",
            ready_to_execute=True,
            required_inputs=["W0", "x0", "B", "xB"],
            available_inputs=["W0", "x0", "B", "xB"],
            missing_inputs=[],
            normalization_steps=[],
            calculation_steps=[],
            result_steps=[],
            warnings=[],
            suggested_next_message="test",
        )

        result = execute_solve_mole_balance(classification, plan)

        self.assertTrue(result.success)
        self.assertIsInstance(result, WorkflowExecutionResult)
        self.assertIn("equations", result.execution_parameters)
        self.assertIsInstance(result.execution_parameters["equations"], list)
        self.assertEqual(
            result.execution_parameters["equations"],
            ["W0 = D + B", "W0*x0 = D*xDavg + B*xB"],
        )

    def test_workflow_execution_result_allows_nested_execution_metadata(self) -> None:
        result = WorkflowExecutionResult(
            workflow_name="test",
            success=True,
            message="ok",
            execution_parameters={
                "equations": ["W0 = D + B", "W0*x0 = D*xDavg + B*xB"],
                "metadata": {"solve_case": "A", "used_root_finding": False},
            },
        )

        self.assertEqual(result.execution_parameters["equations"][0], "W0 = D + B")
        self.assertEqual(result.execution_parameters["metadata"]["solve_case"], "A")

    def test_rayleigh_w0_x0_xb_succeeds_in_model_units(self) -> None:
        classification = GoalClassification(
            goal="solve_rayleigh_batch_variables",
            output_mode="numeric_answer",
            known_inputs=["W0", "x0", "xB"],
            requested_outputs=["B", "D"],
            missing_inputs=[],
            variable_assignments={
                "W0": "100",
                "x0": "0.05",
                "xB": "0.001",
            },
            input_format="model_units",
            output_format="model_units",
            requires_input_conversion=False,
            requires_output_conversion=False,
            confidence=0.95,
            reasoning_summary="test",
            user_facing_summary="test",
        )
        plan = WorkflowPlan(
            workflow_name="solve_rayleigh_batch_variables",
            ready_to_execute=True,
            required_inputs=["W0", "x0", "xB"],
            available_inputs=["W0", "x0", "xB"],
            missing_inputs=[],
            normalization_steps=[],
            calculation_steps=[],
            result_steps=[],
            warnings=[],
            suggested_next_message="test",
        )

        result = execute_solve_rayleigh_batch_variables(classification, plan)

        self.assertTrue(result.success)
        row = result.rows[0]
        self.assertIn("B", row)
        self.assertIn("D", row)
        self.assertIn("xDavg", row)
        self.assertIn("rayleigh_integral", result.execution_parameters)
        self.assertAlmostEqual(float(row["D"]), float(row["W0"]) - float(row["B"]), places=5)

    def test_rayleigh_w0_x0_xb_xdavg_succeeds_with_consistency_check(self) -> None:
        classification = GoalClassification(
            goal="solve_rayleigh_batch_variables",
            output_mode="numeric_answer",
            known_inputs=["W0", "x0", "xB", "xDavg"],
            requested_outputs=["B", "D"],
            missing_inputs=[],
            variable_assignments={
                "W0": "100",
                "x0": "0.05",
                "xB": "0.001",
                "xDavg": "0.2",
            },
            input_format="model_units",
            output_format="model_units",
            requires_input_conversion=False,
            requires_output_conversion=False,
            confidence=0.95,
            reasoning_summary="test",
            user_facing_summary="test",
        )
        plan = WorkflowPlan(
            workflow_name="solve_rayleigh_batch_variables",
            ready_to_execute=True,
            required_inputs=["W0", "x0", "xB"],
            available_inputs=["W0", "x0", "xB", "xDavg"],
            missing_inputs=[],
            normalization_steps=[],
            calculation_steps=[],
            result_steps=[],
            warnings=[],
            suggested_next_message="test",
        )

        result = execute_solve_rayleigh_batch_variables(classification, plan)

        self.assertTrue(result.success)
        row = result.rows[0]
        self.assertIn("provided_xDavg", row)
        self.assertIn("xDavg_implied", row)
        self.assertIn("xDavg_error", row)
        self.assertIn("provided_xDavg", result.execution_parameters)
        self.assertIn("implied_xDavg", result.execution_parameters)

    def test_rayleigh_w0_volume_with_mole_fraction_compositions_is_normalized(self) -> None:
        classification = GoalClassification(
            goal="solve_rayleigh_batch_variables",
            output_mode="numeric_answer",
            known_inputs=["W0", "x0", "xB"],
            requested_outputs=["B", "D"],
            missing_inputs=[],
            variable_assignments={
                "W0": "100 L",
                "x0": "0.05 mole fraction",
                "xB": "0.001 mole fraction",
            },
            input_format="mixed_units",
            output_format="mixed_units",
            requires_input_conversion=True,
            requires_output_conversion=True,
            confidence=0.95,
            reasoning_summary="test",
            user_facing_summary="test",
        )
        plan = WorkflowPlan(
            workflow_name="solve_rayleigh_batch_variables",
            ready_to_execute=True,
            required_inputs=["W0", "x0", "xB"],
            available_inputs=["W0", "x0", "xB"],
            missing_inputs=[],
            normalization_steps=[],
            calculation_steps=[],
            result_steps=[],
            warnings=[],
            suggested_next_message="test",
        )

        result = execute_solve_rayleigh_batch_variables(classification, plan)

        self.assertTrue(result.success)
        self.assertIn("W0", result.normalized_inputs)
        self.assertGreater(float(result.normalized_inputs["W0"]), 0.0)

    def test_rayleigh_planner_and_executor_stay_consistent_for_w0_x0_xb(self) -> None:
        classification = GoalClassification(
            goal="solve_rayleigh_batch_variables",
            output_mode="numeric_answer",
            known_inputs=["W0", "x0", "xB"],
            requested_outputs=["B", "D"],
            missing_inputs=[],
            variable_assignments={
                "W0": "100 L",
                "x0": "10% ABV",
                "xB": "0.001",
            },
            input_format="mixed_units",
            output_format="mixed_units",
            requires_input_conversion=True,
            requires_output_conversion=True,
            confidence=0.95,
            reasoning_summary="test",
            user_facing_summary="test",
        )

        plan = create_workflow_plan(classification)
        result = execute_solve_rayleigh_batch_variables(classification, plan)

        self.assertTrue(plan.ready_to_execute)
        self.assertTrue(result.success)

    def test_interface_hints_preserve_w0_liter_units(self) -> None:
        hints = analyze_message_features(
            "given W0 = 100 L, x0 = 10% abv, xB = 0.001, xDavg = 40% abv, solve for B and D."
        )

        self.assertEqual(hints["input_format_hint"], "mixed_units")
        self.assertEqual(hints["variable_assignments_hint"]["W0"], "100 L")
        self.assertEqual(hints["variable_assignments_hint"]["x0"], "10% abv")


if __name__ == "__main__":
    unittest.main()
