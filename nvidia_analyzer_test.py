import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from PIL import Image

from llm_analyzer import LLMAnalyzer


class NvidiaLLMAnalyzerTests(unittest.TestCase):
    @patch("vision_evaluation.nvidia_models.create_chat_model")
    def test_initializes_text_and_vision_clients(self, create_chat_model):
        text_client = SimpleNamespace()
        vision_client = SimpleNamespace()
        create_chat_model.side_effect = [text_client, vision_client]

        analyzer = LLMAnalyzer(
            model="nvidia/text-model",
            vision_model="nvidia/vision-model",
            provider="nvidia",
            nvidia_api_key="test-key",
        )

        self.assertIs(analyzer.llm, text_client)
        self.assertIs(analyzer.vision_llm, vision_client)
        self.assertEqual(create_chat_model.call_count, 2)
        first_call = create_chat_model.call_args_list[0]
        second_call = create_chat_model.call_args_list[1]
        self.assertEqual(first_call.kwargs["api_key"], "test-key")
        self.assertEqual(first_call.kwargs["temperature"], 0.6)
        self.assertEqual(second_call.kwargs["temperature"], 0.0)

    @patch("vision_evaluation.nvidia_models.create_chat_model")
    def test_vision_invocation_embeds_image(self, create_chat_model):
        vision_client = Mock()
        vision_client.invoke.return_value = SimpleNamespace(
            content="recognized handwriting"
        )
        create_chat_model.return_value = vision_client
        analyzer = LLMAnalyzer(
            provider="nvidia",
            nvidia_api_key="test-key",
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "sample.png"
            Image.new("RGB", (2, 2), color="white").save(image_path)
            result = analyzer._invoke_vision_llm_once(
                "Read this image", str(image_path)
            )

        self.assertEqual(result, "recognized handwriting")
        message = vision_client.invoke.call_args.args[0][0]
        self.assertEqual(message.content[1]["type"], "image")
        self.assertEqual(message.content[1]["mime_type"], "image/png")
        self.assertTrue(message.content[1]["base64"])


if __name__ == "__main__":
    unittest.main()
