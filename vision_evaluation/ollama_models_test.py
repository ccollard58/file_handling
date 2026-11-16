import unittest
from unittest.mock import Mock, patch

import requests

from ollama_cloud_models import fetch_local_vision_models


class OllamaLocalVisionModelTests(unittest.TestCase):
    @patch("ollama_cloud_models.requests.post")
    @patch("ollama_cloud_models.requests.get")
    def test_lists_installed_models_with_vision_capability(
        self, get, post
    ):
        get.return_value.json.return_value = {
            "models": [
                {"name": "llava:latest"},
                {"name": "llama3:latest"},
            ]
        }
        post.side_effect = [
            Mock(json=Mock(return_value={"capabilities": ["completion", "vision"]})),
            Mock(json=Mock(return_value={"capabilities": ["completion"]})),
        ]

        models = fetch_local_vision_models("http://ollama.test", api_key="test-key")

        self.assertEqual(models, ["llava:latest"])
        get.assert_called_once_with(
            "http://ollama.test/api/tags",
            headers={"Authorization": "Bearer test-key"},
            timeout=20,
        )
        self.assertEqual(post.call_count, 2)

    @patch("ollama_cloud_models.requests.post")
    @patch("ollama_cloud_models.requests.get")
    def test_uses_known_vision_name_when_capabilities_are_unavailable(
        self, get, post
    ):
        get.return_value.json.return_value = {
            "models": [{"name": "qwen2.5vl:latest"}]
        }
        post.side_effect = ValueError("capabilities unavailable")

        self.assertEqual(
            fetch_local_vision_models("http://ollama.test"),
            ["qwen2.5vl:latest"],
        )

    @patch(
        "ollama_cloud_models.fetch_text_models",
        return_value=["local-text:latest"],
    )
    @patch(
        "evaluation_gui.fetch_local_vision_models",
        return_value=["local-vl:latest"],
    )
    @patch(
        "evaluation_gui.fetch_free_cloud_vision_models",
        return_value=["cloud-vl:cloud"],
    )
    def test_gui_includes_local_models_in_evaluation_and_judge_lists(
        self, _cloud, _local_vision, _text
    ):
        from evaluation_gui import make_fetch_task

        models, judges, judge_error, vision_error = make_fetch_task(
            "http://localhost:11434", "ollama"
        )(None)

        self.assertEqual(models, ["cloud-vl:cloud", "local-vl:latest"])
        self.assertEqual(judges, ["local-text:latest", "local-vl:latest"])
        self.assertIsNone(judge_error)
        self.assertIsNone(vision_error)

    @patch(
        "ollama_cloud_models.fetch_text_models",
        return_value=["local-text:latest"],
    )
    @patch(
        "evaluation_gui.fetch_local_vision_models",
        return_value=["local-vl:latest"],
    )
    @patch(
        "evaluation_gui.fetch_free_cloud_vision_models",
        side_effect=requests.RequestException("offline"),
    )
    def test_local_models_remain_available_when_cloud_refresh_fails(
        self, _cloud, _local_vision, _text
    ):
        from evaluation_gui import make_fetch_task

        models, judges, judge_error, vision_error = make_fetch_task(
            "http://localhost:11434", "ollama"
        )(None)

        self.assertEqual(models, ["local-vl:latest"])
        self.assertEqual(judges, ["local-text:latest", "local-vl:latest"])
        self.assertIsNone(judge_error)
        self.assertIn("Could not refresh Ollama cloud models", vision_error)


if __name__ == "__main__":
    unittest.main()
