import unittest

from agents.interface_agent import DISPLAY_VARIABLE_ORDER
from app.ui_metadata import (
    get_core_batch_variable_keys,
    get_user_facing_batch_input_keys,
    get_user_facing_batch_input_metadata,
)


class InterfaceAgentTests(unittest.TestCase):
    def test_display_variable_order_uses_registry_core_order(self) -> None:
        expected = (
            get_core_batch_variable_keys() + get_user_facing_batch_input_keys()
        )
        self.assertEqual(DISPLAY_VARIABLE_ORDER, expected)

    def test_user_facing_batch_input_keys_match_existing_tail_order(self) -> None:
        self.assertEqual(
            get_user_facing_batch_input_keys(),
            [
                "feed_volume",
                "feed_abv",
                "product_volume",
                "product_abv",
                "bottoms_volume",
                "bottoms_abv",
            ],
        )

    def test_user_facing_batch_input_metadata_maps_to_internal_variables(self) -> None:
        expected_mappings = {
            "feed_volume": "W0",
            "feed_abv": "x0",
            "product_volume": "D",
            "product_abv": "xDavg",
            "bottoms_volume": "B",
            "bottoms_abv": "xB",
        }

        for key, related_internal in expected_mappings.items():
            metadata = get_user_facing_batch_input_metadata(key)
            self.assertIsNotNone(metadata)
            self.assertEqual(metadata["related_internal_variable"], related_internal)


if __name__ == "__main__":
    unittest.main()
