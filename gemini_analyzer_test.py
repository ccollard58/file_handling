import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from PIL import Image

from llm_analyzer import LLMAnalyzer


class GeminiLLMAnalyzerTests(unittest.TestCase):
    @patch("llm_analyzer.genai.Client")
    @patch("vision_evaluation.gemini_models.create_chat_model")
    def test_initializes_clients_with_shared_gemini_factory(
        self, create_chat_model, genai_client
    ):
        text_client = SimpleNamespace()
        vision_client = SimpleNamespace()
        create_chat_model.side_effect = [text_client, vision_client]

        analyzer = LLMAnalyzer(
            model="gemini-text",
            vision_model="gemini-vision",
            provider="google",
            google_api_key="test-key",
        )

        self.assertIs(analyzer.llm, text_client)
        self.assertIs(analyzer.vision_llm, vision_client)
        self.assertEqual(create_chat_model.call_count, 2)
        self.assertEqual(
            create_chat_model.call_args_list[0].kwargs["api_key"],
            "test-key",
        )
        self.assertEqual(
            create_chat_model.call_args_list[1].kwargs["temperature"],
            0.0,
        )
        self.assertTrue(genai_client.called)

    @patch("vision_evaluation.gemini_models.fetch_model_ids")
    def test_model_discovery_uses_shared_gemini_catalog(self, fetch_model_ids):
        fetch_model_ids.return_value = ["gemini-text", "gemini-vision"]

        text_models, vision_models = LLMAnalyzer.get_google_models("test-key")

        self.assertEqual(text_models, ["gemini-text", "gemini-vision"])
        self.assertEqual(vision_models, text_models)
        self.assertIsNot(vision_models, text_models)
        fetch_model_ids.assert_called_once_with(api_key="test-key")

    @patch("llm_analyzer.count_input_tokens", return_value=7)
    def test_text_request_counts_and_reserves_prompt(self, count_tokens):
        analyzer = LLMAnalyzer.__new__(LLMAnalyzer)
        analyzer.provider = "google"
        analyzer.google_client = object()
        analyzer._gemini_rate_limiter = Mock()
        analyzer._gemini_rate_limiter.admit.return_value = nullcontext()
        analyzer.model = "gemini-text"
        analyzer.llm = Mock()
        analyzer.llm.invoke.return_value = SimpleNamespace(content="analysis")

        result = analyzer._invoke_llm_once("Classify this document")

        self.assertEqual(result, "analysis")
        count_tokens.assert_called_once_with(
            analyzer.google_client,
            "gemini-text",
            "Classify this document",
        )
        analyzer._gemini_rate_limiter.admit.assert_called_once_with(
            "gemini-text", 7
        )

    @patch("llm_analyzer.count_input_tokens", return_value=11)
    def test_vision_request_counts_and_reserves_image_input(self, count_tokens):
        analyzer = LLMAnalyzer.__new__(LLMAnalyzer)
        analyzer.provider = "google"
        analyzer.google_client = object()
        analyzer._gemini_rate_limiter = Mock()
        analyzer._gemini_rate_limiter.admit.return_value = nullcontext()
        analyzer.vision_model = "gemini-vision"
        analyzer.vision_llm = Mock()
        analyzer.vision_llm.invoke.return_value = SimpleNamespace(
            content="recognized text"
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "sample.png"
            Image.new("RGB", (2, 2), color="white").save(image_path)
            result = analyzer._invoke_vision_llm_once(
                "Read this image", str(image_path)
            )

        self.assertEqual(result, "recognized text")
        contents = count_tokens.call_args.args[2]
        self.assertEqual(contents[1], "Read this image")
        analyzer._gemini_rate_limiter.admit.assert_called_once_with(
            "gemini-vision", 11
        )


if __name__ == "__main__":
    unittest.main()
