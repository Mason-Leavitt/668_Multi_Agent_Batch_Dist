import unittest

from app.ui_metadata import (
    format_core_variable_reference,
    format_user_facing_batch_input_reference,
    format_variable_and_input_reference,
)
from agents.classification_prompt import (
    CLASSIFICATION_SYSTEM_PROMPT,
    build_variable_reference_prompt_section,
    build_workflow_reference_prompt_section,
)


class ClassificationPromptTests(unittest.TestCase):
    def test_variable_reference_section_includes_core_registry_variables(self) -> None:
        section = build_variable_reference_prompt_section()

        self.assertIn("W0", section)
        self.assertIn("x0", section)
        self.assertIn("xDavg", section)
        self.assertIn("xB", section)
        self.assertIn("feed_volume", section)
        self.assertIn("product_abv", section)

    def test_workflow_reference_section_includes_registry_workflows(self) -> None:
        section = build_workflow_reference_prompt_section()

        self.assertIn("feed_to_product_sweep:", section)
        self.assertIn("product_to_feed_sweep:", section)
        self.assertIn("solve_rayleigh_batch_variables:", section)

    def test_system_prompt_contains_registry_backed_sections(self) -> None:
        self.assertIn("Core variable glossary:", CLASSIFICATION_SYSTEM_PROMPT)
        self.assertIn("User-facing batch input forms:", CLASSIFICATION_SYSTEM_PROMPT)
        self.assertIn("Supported top-level goals:", CLASSIFICATION_SYSTEM_PROMPT)
        self.assertIn("W0", CLASSIFICATION_SYSTEM_PROMPT)
        self.assertIn("feed_volume", CLASSIFICATION_SYSTEM_PROMPT)
        self.assertIn("product_to_feed_sweep:", CLASSIFICATION_SYSTEM_PROMPT)

    def test_shared_metadata_formatters_cover_core_and_user_facing_inputs(self) -> None:
        core = format_core_variable_reference()
        user_facing = format_user_facing_batch_input_reference()
        combined = format_variable_and_input_reference()

        self.assertIn("W0", core)
        self.assertIn("xDavg", core)
        self.assertIn("feed_volume", user_facing)
        self.assertIn("product_abv", user_facing)
        self.assertIn("Core variable glossary:", combined)
        self.assertIn("User-facing batch input forms:", combined)


if __name__ == "__main__":
    unittest.main()
