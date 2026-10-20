import unittest

from openrouter_models import (
    filter_free_vision_models,
    parse_vision_models,
)


class OpenRouterVisionModelTests(unittest.TestCase):
    def test_free_only_filter_excludes_paid_and_unknown_pricing(self):
        payload = {
            "data": [
                {
                    "id": "qwen/qwen3.8-27b:free",
                    "architecture": {
                        "input_modalities": ["text", "image"],
                        "output_modalities": ["text"],
                    },
                    "pricing": {"prompt": "0", "completion": "0"},
                },
                {
                    "id": "qwen/qwen3.8-27b",
                    "architecture": {
                        "input_modalities": ["text", "image"],
                        "output_modalities": ["text"],
                    },
                    "pricing": {"prompt": "0.00000024", "completion": "0.0000022"},
                },
                {
                    "id": "unknown/price",
                    "architecture": {
                        "input_modalities": ["text", "image"],
                        "output_modalities": ["text"],
                    },
                },
            ]
        }

        self.assertEqual(
            filter_free_vision_models(payload), ["qwen/qwen3.8-27b:free"]
        )

    def test_paid_models_are_included_and_marked_with_rates(self):
        payload = {
            "data": [
                {
                    "id": "qwen/qwen3.8-27b",
                    "architecture": {
                        "input_modalities": ["text", "image"],
                        "output_modalities": ["text"],
                    },
                    "pricing": {"prompt": "0.00000024", "completion": "0.0000022"},
                }
            ]
        }

        [model] = parse_vision_models(payload, include_paid=True)

        self.assertEqual(model.model_id, "qwen/qwen3.8-27b")
        self.assertFalse(model.is_free)
        self.assertIn("$0.24/M input", model.display_name)
        self.assertIn("$2.2/M output", model.display_name)

if __name__ == "__main__":
    unittest.main()
