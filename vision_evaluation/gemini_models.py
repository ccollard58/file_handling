"""Discover Gemini API models and build chat clients for them.

Models are listed through the native Generative Language API and called through
Google's OpenAI-compatible endpoint, so only langchain-openai is required.
Docs: https://ai.google.dev/gemini-api/docs
"""

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests

GEMINI_API_ROOT = "https://generativelanguage.googleapis.com/v1beta"
GEMINI_BASE_URL = f"{GEMINI_API_ROOT}/openai/"
API_KEY_ENVS = ("GEMINI_API_KEY", "GOOGLE_API_KEY")
# Gemini models that cannot transcribe an image into text or act as a text judge.
_EXCLUDED_MARKERS = (
    "tts", "embedding", "image", "audio", "live", "robotics",
    "computer-use", "aqa", "imagen", "veo",
)
# Thinking tokens count against max_tokens, so leave room for them.
MIN_MAX_TOKENS = 4096
REASONING_EFFORT = "low"


@dataclass(frozen=True)
class GeminiModel:
    model_id: str

    @property
    def is_free(self) -> bool:
        # The Gemini API free tier covers prototyping within rate limits.
        return True

    @property
    def display_name(self) -> str:
        return f"{self.model_id} [FREE TIER]"


def get_api_key(explicit: Optional[str] = None) -> Optional[str]:
    if explicit:
        return explicit
    return next((os.environ[k] for k in API_KEY_ENVS if os.environ.get(k)), None)


def _require_key(explicit: Optional[str]) -> str:
    key = get_api_key(explicit)
    if not key:
        raise RuntimeError(
            f"Gemini API key missing: set {API_KEY_ENVS[0]}, pass --gemini-api-key, "
            "or enter it in the GUI"
        )
    return key


def parse_model_ids(models: List[Dict[str, Any]]) -> List[str]:
    """Pick multimodal Gemini chat models from a native models.list payload."""
    ids = set()
    for model in models:
        model_id = (model.get("name") or "").removeprefix("models/")
        methods = model.get("supportedGenerationMethods") or []
        if (
            model_id.startswith("gemini")
            and "generateContent" in methods
            and not any(marker in model_id for marker in _EXCLUDED_MARKERS)
        ):
            ids.add(model_id)
    return sorted(ids)


def fetch_model_ids(api_key: Optional[str] = None, timeout: int = 20) -> List[str]:
    """List callable Gemini model ids. Raises requests.RequestException on HTTP errors."""
    key = _require_key(api_key)
    models: List[Dict[str, Any]] = []
    page_token = None
    while True:
        params = {"pageSize": 1000}
        if page_token:
            params["pageToken"] = page_token
        response = requests.get(
            f"{GEMINI_API_ROOT}/models",
            params=params,
            headers={"x-goog-api-key": key},
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
        models.extend(payload.get("models", []))
        page_token = payload.get("nextPageToken")
        if not page_token:
            return parse_model_ids(models)


def fetch_vision_models(api_key: Optional[str] = None, timeout: int = 20) -> List[GeminiModel]:
    """Gemini models are natively multimodal, so every chat model can read images."""
    return [GeminiModel(m) for m in fetch_model_ids(api_key, timeout)]


def fetch_text_models(api_key: Optional[str] = None, timeout: int = 20) -> List[str]:
    """Gemini models usable as text judges."""
    return fetch_model_ids(api_key, timeout)


def create_chat_model(
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    timeout: int = 120,
    api_key: Optional[str] = None,
):
    """Create a LangChain ChatOpenAI client pointed at the Gemini API."""
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model,
        temperature=temperature,
        max_tokens=max(max_tokens, MIN_MAX_TOKENS),
        timeout=timeout,
        base_url=GEMINI_BASE_URL,
        api_key=_require_key(api_key),
        # Quota admission and retry pacing are managed by the evaluator so a
        # client retry cannot send unreserved requests behind its back.
        max_retries=0,
        reasoning_effort=REASONING_EFFORT,
    )


if __name__ == "__main__":
    print("\n".join(m.display_name for m in fetch_vision_models()))
