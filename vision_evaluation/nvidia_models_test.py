import unittest
from unittest.mock import patch

from nvidia_models import (
    CATALOG_MODEL_PATHS,
    fetch_vision_models,
)


class NVIDIAVisionModelTests(unittest.TestCase):
    @patch("nvidia_models._register_mistral_large_4")
    @patch("nvidia_models._register_glm_53_flash")
    @patch("nvidia_models.has_hosted_endpoint", return_value=True)
    @patch("langchain_nvidia_ai_endpoints.ChatNVIDIA")
    def test_mistral_large_is_listed_for_evaluation(
        self, chat_nvidia, _has_hosted_endpoint, _register_glm, _register_mistral
    ):
        chat_nvidia.return_value._client.available_models = []
        chat_nvidia.return_value.available_models = []

        models = fetch_vision_models(api_key="test-key")

        mistral = next(model for model in models if model.model_id == "mistral-large-4")
        self.assertEqual(mistral.display_name, "Mistral Large 4 [FREE]")
        self.assertEqual(
            CATALOG_MODEL_PATHS[mistral.model_id],
            "mistralai/mistral-large-4-1t-a52b",
        )

    @patch("nvidia_models._register_step_5_preview")
    @patch("nvidia_models._register_mistral_large_4")
    @patch("nvidia_models._register_glm_53_flash")
    @patch("nvidia_models.has_hosted_endpoint", return_value=True)
    @patch("langchain_nvidia_ai_endpoints.ChatNVIDIA")
    def test_step_5_preview_is_listed_for_evaluation(
        self, chat_nvidia, _has_hosted_endpoint, _register_glm,
        _register_mistral, _register_step_5
    ):
        chat_nvidia.return_value._client.available_models = []
        chat_nvidia.return_value.available_models = []

        models = fetch_vision_models(api_key="test-key")

        step_5 = next(
            model for model in models if model.model_id == "stepfun-ai/step-5-preview"
        )
        self.assertEqual(step_5.display_name, "Step 5 Preview [FREE]")
        self.assertEqual(
            CATALOG_MODEL_PATHS[step_5.model_id],
            "stepfun-ai/step-5-preview",
        )


if __name__ == "__main__":
    unittest.main()
