"""Discover free, vision-capable Ollama cloud models listed on ollama.com."""

import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

import requests

OLLAMA_SEARCH_URL = "https://ollama.com/search?c=cloud&c=vision"
OLLAMA_TAGS_URL = "https://ollama.com/library/{name}/tags"
_HEADERS = {"User-Agent": "Mozilla/5.0 (vision-model-evaluator)"}
_VISION_NAME_MARKERS = (
    "llava",
    "vision",
    "minicpm-v",
    "qwen2.5vl",
    "qwen3-vl",
    "qwen3.5",
    "moondream",
    "bakllava",
    "mistral-small3",
    "cogvlm",
    "pixtral",
    "gemma3",
    "gemma4",
    "llama4",
    "glm-ocr",
    "translategemma",
)


def _get(url: str, timeout: int = 20) -> str:
    response = requests.get(url, headers=_HEADERS, timeout=timeout)
    response.raise_for_status()
    return response.text


def parse_library_names(search_html: str) -> List[str]:
    """Extract unique model family names from an ollama.com search page."""
    names = re.findall(r'href="/library/([^"/:?]+)"', search_html)
    return list(dict.fromkeys(names))


def parse_cloud_tags(name: str, tags_html: str) -> List[str]:
    """Extract full `name:tag` identifiers whose tag marks a cloud variant."""
    pattern = rf'/library/{re.escape(name)}:([^"\s<?]*cloud[^"\s<?]*)'
    tags = dict.fromkeys(re.findall(pattern, tags_html))
    return [f"{name}:{tag}" for tag in tags]


def fetch_cloud_vision_models(timeout: int = 20) -> List[str]:
    """Return all cloud vision model identifiers listed on ollama.com.

    Raises requests.RequestException if the search page cannot be fetched.
    Families whose tag page fails to load are skipped.
    """
    models: List[str] = []
    for name in parse_library_names(_get(OLLAMA_SEARCH_URL, timeout)):
        try:
            tags_html = _get(OLLAMA_TAGS_URL.format(name=name), timeout)
        except requests.RequestException:
            continue
        models.extend(parse_cloud_tags(name, tags_html))
    return models


def is_model_usable(
    model: str,
    ollama_url: str,
    timeout: int = 60,
    api_key: Optional[str] = None,
) -> bool:
    """Probe a model with a one-token request through the Ollama server.

    ollama.com does not publish which plan a model belongs to, but the server
    answers 402 for models outside the Free plan (and 400/403 for models that
    cannot be used), so only a 200 response counts as free and usable.
    """
    api_key = api_key or os.environ.get("OLLAMA_API_KEY")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        response = requests.post(
            f"{ollama_url.rstrip('/')}/api/chat",
            json={
                "model": model,
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
                "options": {"num_predict": 1},
            },
            headers=headers,
            timeout=timeout,
        )
    except requests.RequestException:
        return False
    return response.status_code == 200


def fetch_free_cloud_vision_models(
    ollama_url: str = "http://localhost:11434",
    timeout: int = 20,
    api_key: Optional[str] = None,
) -> List[str]:
    """Return cloud vision models that the Ollama server accepts on the Free plan."""
    candidates = fetch_cloud_vision_models(timeout)
    if not candidates:
        return []
    with ThreadPoolExecutor(max_workers=4) as pool:
        usable = list(
            pool.map(
                lambda m: is_model_usable(m, ollama_url, api_key=api_key),
                candidates,
            )
        )
    return [model for model, ok in zip(candidates, usable) if ok]


def _is_vision_model(
    model: str,
    ollama_url: str,
    timeout: int,
    headers: dict,
) -> bool:
    """Check model capabilities, falling back to known vision model names."""
    try:
        response = requests.post(
            f"{ollama_url.rstrip('/')}/api/show",
            json={"model": model},
            headers=headers,
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
        if not payload.get("capabilities"):
            raise ValueError("model capabilities are unavailable")
        capabilities = {
            capability.lower() for capability in payload["capabilities"]
        }
        return "vision" in capabilities
    except (requests.RequestException, ValueError):
        lowered = model.lower()
        return any(marker in lowered for marker in _VISION_NAME_MARKERS)


def fetch_local_vision_models(
    ollama_url: str = "http://localhost:11434",
    timeout: int = 20,
    api_key: Optional[str] = None,
) -> List[str]:
    """List installed vision models at an Ollama-compatible endpoint."""
    api_key = api_key or os.environ.get("OLLAMA_API_KEY")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    response = requests.get(
        f"{ollama_url.rstrip('/')}/api/tags", headers=headers, timeout=timeout
    )
    response.raise_for_status()
    names = [
        model.get("name") or model.get("model")
        for model in response.json().get("models", [])
    ]
    names = [name for name in names if name]
    with ThreadPoolExecutor(max_workers=4) as pool:
        usable = list(
            pool.map(
                lambda name: _is_vision_model(name, ollama_url, timeout, headers),
                names,
            )
        )
    return sorted(name for name, ok in zip(names, usable) if ok)


def _is_text_model(
    model: str,
    ollama_url: str,
    timeout: int,
    headers: dict,
) -> bool:
    """Check model capabilities and keep generative models without vision input."""
    try:
        response = requests.post(
            f"{ollama_url.rstrip('/')}/api/show",
            json={"model": model},
            headers=headers,
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
        if not payload.get("capabilities"):
            raise ValueError("model capabilities are unavailable")
        capabilities = {capability.lower() for capability in payload["capabilities"]}
    except (requests.RequestException, ValueError):
        # Older Ollama-compatible servers may not return capabilities. In that
        # case, use the model name to avoid obvious vision and embedding models.
        lowered = model.lower()
        return not any(token in lowered for token in ("vision", "-vl", "vl:", "embed", "rerank"))

    return "vision" not in capabilities and bool(
        capabilities.intersection({"completion", "thinking", "tools"})
    )


def fetch_text_models(
    ollama_url: str = "http://localhost:11434",
    timeout: int = 20,
    api_key: Optional[str] = None,
) -> List[str]:
    """List installed/callable text models at an Ollama-compatible endpoint."""
    api_key = api_key or os.environ.get("OLLAMA_API_KEY")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    response = requests.get(
        f"{ollama_url.rstrip('/')}/api/tags", headers=headers, timeout=timeout
    )
    response.raise_for_status()
    names = [
        model.get("name") or model.get("model")
        for model in response.json().get("models", [])
    ]
    names = [name for name in names if name]
    with ThreadPoolExecutor(max_workers=4) as pool:
        usable = list(
            pool.map(
                lambda name: _is_text_model(name, ollama_url, timeout, headers),
                names,
            )
        )
    return sorted(name for name, ok in zip(names, usable) if ok)


if __name__ == "__main__":
    print("\n".join(fetch_free_cloud_vision_models()))
