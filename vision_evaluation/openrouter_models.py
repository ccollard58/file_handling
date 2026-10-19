"""Discover free, vision-capable models on OpenRouter and build chat clients for them."""

import os
from typing import Any, Dict, List, Optional

import requests

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
API_KEY_ENV = "OPENROUTER_API_KEY"


def get_api_key(explicit: Optional[str] = None) -> Optional[str]:
    return explicit or os.environ.get(API_KEY_ENV)


def is_free(model: Dict[str, Any]) -> bool:
    pricing = model.get("pricing") or {}
    try:
        return all(float(pricing.get(k, 0) or 0) == 0 for k in ("prompt", "completion"))
    except (TypeError, ValueError):
        return False


def accepts_images(model: Dict[str, Any]) -> bool:
    arch = model.get("architecture") or {}
    outputs = arch.get("output_modalities") or []
    return "image" in (arch.get("input_modalities") or []) and "text" in outputs and "audio" not in outputs


def filter_free_vision_models(payload: Dict[str, Any]) -> List[str]:
    """Return ids of free models that accept image input from a /models payload."""
    return sorted(
        m["id"] for m in payload.get("data", [])
        if m.get("id") and m["id"] != "openrouter/free" and is_free(m) and accepts_images(m)
    )


def fetch_free_vision_models(
    base_url: str = OPENROUTER_BASE_URL, timeout: int = 20, api_key: Optional[str] = None
) -> List[str]:
    """List free vision models. The models endpoint is public; a key is optional.

    Raises requests.RequestException on network or HTTP errors.
    """
    headers = {"Authorization": f"Bearer {key}"} if (key := get_api_key(api_key)) else {}
    response = requests.get(f"{base_url.rstrip('/')}/models", headers=headers, timeout=timeout)
    response.raise_for_status()
    return filter_free_vision_models(response.json())


def create_chat_model(
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    timeout: int = 120,
    base_url: str = OPENROUTER_BASE_URL,
    api_key: Optional[str] = None,
):
    """Create a LangChain ChatOpenAI client for an OpenRouter model."""
    from langchain_openai import ChatOpenAI

    key = get_api_key(api_key)
    if not key:
        raise RuntimeError(f"OpenRouter API key missing: set {API_KEY_ENV} or pass --openrouter-api-key")

    kwargs: Dict[str, Any] = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "timeout": timeout,
        "base_url": base_url,
        "api_key": key,
        "max_retries": 3,
    }
    if "qwen" in model.lower() or "qwq" in model.lower():
        kwargs["top_p"] = 0.8
        kwargs["extra_body"] = {"top_k": 20, "min_p": 0.0}
    return ChatOpenAI(**kwargs)


if __name__ == "__main__":
    print("\n".join(fetch_free_vision_models()))
