"""Discover Token Harbor vision models (free ones in particular) and build chat clients.

Token Harbor exposes an OpenAI-compatible API at https://tokenharbor.ai/v1, but its
authenticated /v1/models endpoint only lists callable ids. Modalities and pricing are
published on the public https://tokenharbor.ai/models page, so that page is parsed and,
when a key is available, intersected with the ids the API says are callable.
"""

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests

TOKENHARBOR_BASE_URL = "https://tokenharbor.ai/v1"
TOKENHARBOR_MODELS_URL = "https://tokenharbor.ai/models"
API_KEY_ENV = "TOKENHARBOR_API_KEY"
_HEADERS = {"User-Agent": "Mozilla/5.0 (vision-model-evaluator)"}


@dataclass(frozen=True)
class TokenHarborModel:
    model_id: str
    price_in: Optional[float]
    price_out: Optional[float]
    is_free: bool
    input_modalities: tuple = ()

    @property
    def is_vision(self) -> bool:
        return "image" in self.input_modalities

    @property
    def display_name(self) -> str:
        if self.is_free:
            return f"{self.model_id} [FREE]"
        if self.price_in is None or self.price_out is None:
            return f"{self.model_id} [PRICE UNKNOWN]"
        return f"{self.model_id} [PAID: ${self.price_in:g}/M input, ${self.price_out:g}/M output]"


def get_api_key(explicit: Optional[str] = None) -> Optional[str]:
    return explicit or os.environ.get(API_KEY_ENV)


_ROW_KEYS = ("rows", "freeRows")


def parse_models_page(html: str) -> List[Dict[str, Any]]:
    """Extract model rows (paid and free lists) embedded in the Next.js payload of /models."""
    rows: Dict[str, Dict[str, Any]] = {}
    for key in _ROW_KEYS:
        marker = f'\\"{key}\\":['
        start = html.find(marker)
        if start < 0:
            continue
        text = html[start + len(marker) - 1:].replace('\\"', '"').replace("\\\\", "\\")
        try:
            parsed, _ = json.JSONDecoder().raw_decode(text)
        except json.JSONDecodeError:
            continue
        for row in parsed:
            if isinstance(row, dict) and row.get("surface"):
                rows.setdefault(row["surface"], row)
    if not rows:
        raise ValueError("Token Harbor models page format not recognized (no model rows found)")
    return list(rows.values())


def parse_vision_models(rows: List[Dict[str, Any]], include_paid: bool = False) -> List[TokenHarborModel]:
    """Convert page rows into vision models, free ones first. Paid ones need include_paid."""
    models = []
    for row in rows:
        model_id = row.get("surface")
        modalities = tuple(row.get("inputModalities") or ())
        outputs = row.get("outputModalities")
        # The page payload may replace repeated lists with string back-references.
        if not isinstance(outputs, list):
            outputs = ["text"]
        if not model_id or "image" not in modalities or "text" not in outputs:
            continue
        model = TokenHarborModel(
            model_id=model_id,
            price_in=row.get("priceIn"),
            price_out=row.get("priceOut"),
            is_free=bool(row.get("isFree")),
            input_modalities=modalities,
        )
        if model.is_free or (include_paid and model.price_in is not None and model.price_out is not None):
            models.append(model)
    return sorted(models, key=lambda m: (not m.is_free, m.model_id))


def fetch_callable_ids(api_key: str, base_url: str = TOKENHARBOR_BASE_URL, timeout: int = 20) -> List[str]:
    """Ids the API accepts for this key. Raises requests.RequestException (e.g. 401)."""
    response = requests.get(
        f"{base_url.rstrip('/')}/models", headers={"Authorization": f"Bearer {api_key}"}, timeout=timeout
    )
    response.raise_for_status()
    return [m["id"] for m in response.json().get("data", []) if m.get("id")]


def fetch_vision_models(
    timeout: int = 20, api_key: Optional[str] = None, include_paid: bool = False,
    base_url: str = TOKENHARBOR_BASE_URL,
) -> List[TokenHarborModel]:
    """List Token Harbor vision models. With a key, drops ids the API does not list as callable.

    Raises requests.RequestException on network/HTTP errors (including a rejected key)
    and ValueError if the models page layout changes.
    """
    page = requests.get(TOKENHARBOR_MODELS_URL, headers=_HEADERS, timeout=timeout)
    page.raise_for_status()
    models = parse_vision_models(parse_models_page(page.text), include_paid=include_paid)
    if key := get_api_key(api_key):
        callable_ids = set(fetch_callable_ids(key, base_url, timeout))
        models = [m for m in models if m.model_id in callable_ids]
    return models


def create_chat_model(
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    timeout: int = 120,
    base_url: str = TOKENHARBOR_BASE_URL,
    api_key: Optional[str] = None,
):
    """Create a LangChain ChatOpenAI client for a Token Harbor model."""
    from langchain_openai import ChatOpenAI

    key = get_api_key(api_key)
    if not key:
        raise RuntimeError(f"Token Harbor API key missing: set {API_KEY_ENV} or pass --tokenharbor-api-key")

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
    print("\n".join(m.display_name for m in fetch_vision_models(include_paid=True)))
