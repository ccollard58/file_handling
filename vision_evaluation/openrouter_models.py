"""Discover free, vision-capable models on OpenRouter and build chat clients for them."""

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
API_KEY_ENV = "OPENROUTER_API_KEY"


@dataclass(frozen=True)
class OpenRouterVisionModel:
    model_id: str
    prompt_price: Optional[float]
    completion_price: Optional[float]

    @property
    def is_free(self) -> bool:
        return self.prompt_price == 0 and self.completion_price == 0

    @property
    def display_name(self) -> str:
        if self.is_free:
            return f"{self.model_id} [FREE]"
        if self.prompt_price is None or self.completion_price is None:
            return f"{self.model_id} [PRICE UNKNOWN]"
        return (
            f"{self.model_id} [PAID: "
            f"${self.prompt_price * 1_000_000:g}/M input, "
            f"${self.completion_price * 1_000_000:g}/M output]"
        )


def get_api_key(explicit: Optional[str] = None) -> Optional[str]:
    return explicit or os.environ.get(API_KEY_ENV)


def _get_price(pricing: Dict[str, Any], name: str) -> Optional[float]:
    if name not in pricing:
        return None
    try:
        return float(pricing[name])
    except (TypeError, ValueError):
        return None


def _is_free(model: Dict[str, Any]) -> bool:
    pricing = model.get("pricing") or {}
    prompt = _get_price(pricing, "prompt")
    completion = _get_price(pricing, "completion")
    return prompt == 0 and completion == 0


def accepts_images(model: Dict[str, Any]) -> bool:
    arch = model.get("architecture") or {}
    outputs = arch.get("output_modalities") or []
    return "image" in (arch.get("input_modalities") or []) and "text" in outputs and "audio" not in outputs


def parse_vision_models(
    payload: Dict[str, Any], include_paid: bool = False
) -> List[OpenRouterVisionModel]:
    """Parse vision models, excluding paid or unknown-price models by default."""
    models = []
    for model in payload.get("data", []):
        model_id = model.get("id")
        if not model_id or model_id == "openrouter/free" or not accepts_images(model):
            continue
        pricing = model.get("pricing") or {}
        prompt_price = _get_price(pricing, "prompt")
        completion_price = _get_price(pricing, "completion")
        entry = OpenRouterVisionModel(model_id, prompt_price, completion_price)
        if entry.is_free or (
            include_paid and prompt_price is not None and completion_price is not None
        ):
            models.append(entry)
    return sorted(models, key=lambda model: (not model.is_free, model.model_id))


def filter_free_vision_models(payload: Dict[str, Any]) -> List[str]:
    """Return ids of free models that accept image input from a /models payload."""
    return [model.model_id for model in parse_vision_models(payload)]


def fetch_vision_models(
    base_url: str = OPENROUTER_BASE_URL,
    timeout: int = 20,
    api_key: Optional[str] = None,
    include_paid: bool = False,
) -> List[OpenRouterVisionModel]:
    """List vision models from OpenRouter, optionally including priced models.

    Raises requests.RequestException on network or HTTP errors.
    """
    headers = {"Authorization": f"Bearer {key}"} if (key := get_api_key(api_key)) else {}
    response = requests.get(f"{base_url.rstrip('/')}/models", headers=headers, timeout=timeout)
    response.raise_for_status()
    return parse_vision_models(response.json(), include_paid=include_paid)


def fetch_free_vision_models(
    base_url: str = OPENROUTER_BASE_URL, timeout: int = 20, api_key: Optional[str] = None
) -> List[str]:
    """List only free vision models from OpenRouter."""
    return [
        model.model_id
        for model in fetch_vision_models(base_url, timeout, api_key, include_paid=False)
    ]


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
