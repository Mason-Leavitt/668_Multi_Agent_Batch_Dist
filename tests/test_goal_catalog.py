import unittest

from agents.goal_catalog import GOAL_DESCRIPTIONS, SUPPORTED_GOALS
from app.ui_metadata import get_workflow_reference


class GoalCatalogTests(unittest.TestCase):
    def test_supported_goals_include_registry_workflows_and_local_categories(self) -> None:
        workflow_ids = [str(workflow["workflow_id"]) for workflow in get_workflow_reference()]

        for workflow_id in workflow_ids:
            self.assertIn(workflow_id, SUPPORTED_GOALS)

        for goal_name in [
            "consistency_check",
            "explain_variable_or_workflow",
            "unsupported_or_unclear",
        ]:
            self.assertIn(goal_name, SUPPORTED_GOALS)

    def test_executable_goal_descriptions_match_workflow_registry(self) -> None:
        for workflow in get_workflow_reference():
            workflow_id = str(workflow["workflow_id"])
            self.assertEqual(GOAL_DESCRIPTIONS[workflow_id], workflow["description"])

    def test_conversational_goal_descriptions_are_still_present(self) -> None:
        self.assertIn("consistency_check", GOAL_DESCRIPTIONS)
        self.assertIn("explain_variable_or_workflow", GOAL_DESCRIPTIONS)
        self.assertIn("unsupported_or_unclear", GOAL_DESCRIPTIONS)


if __name__ == "__main__":
    unittest.main()
