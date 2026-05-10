import unittest

from agents.conversation_router import (
    ROUTER_SYSTEM_PROMPT,
    WORKFLOW_SUMMARY_SECTION,
    build_router_workflow_summary_section,
)


class ConversationRouterPromptTests(unittest.TestCase):
    def test_router_workflow_summary_includes_registry_workflows(self) -> None:
        section = build_router_workflow_summary_section()

        self.assertIn("feed_to_product_sweep", section)
        self.assertIn("product_to_feed_sweep", section)
        self.assertIn("solve_mole_balance", section)
        self.assertIn("solve_rayleigh_batch_variables", section)

    def test_router_prompt_contains_generated_workflow_summary(self) -> None:
        self.assertIn("Implemented executable workflows:", ROUTER_SYSTEM_PROMPT)
        self.assertIn(WORKFLOW_SUMMARY_SECTION, ROUTER_SYSTEM_PROMPT)
        self.assertIn("feed_to_product_sweep", ROUTER_SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()
