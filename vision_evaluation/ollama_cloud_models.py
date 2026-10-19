"""Discover free, vision-capable Ollama cloud models listed on ollama.com."""

import re
from concurrent.futures import ThreadPoolExecutor
from typing import List

import requests

OLLAMA_SEARCH_URL = "https://ollama.com/search?c=cloud&c=vision"
OLLAMA_TAGS_URL = "https://ollama.com/library/{name}/tags"
_HEADERS = {"User-Agent": "Mozilla/5.0 (vision-model-evaluator)"}


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


def is_model_usable(model: str, ollama_url: str, timeout: int = 60) -> bool:
    """Probe a model with a one-token request through the Ollama server.

    ollama.com does not publish which plan a model belongs to, but the server
    answers 402 for models outside the Free plan (and 400/403 for models that
    cannot be used), so only a 200 response counts as free and usable.
    """
    try:
        response = requests.post(
            f"{ollama_url.rstrip('/')}/api/chat",
            json={
                "model": model,
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
                "options": {"num_predict": 1},
            },
            timeout=timeout,
        )
    except requests.RequestException:
        return False
    return response.status_code == 200


def fetch_free_cloud_vision_models(
    ollama_url: str = "http://localhost:11434", timeout: int = 20
) -> List[str]:
    """Return cloud vision models that the Ollama server accepts on the Free plan."""
    candidates = fetch_cloud_vision_models(timeout)
    if not candidates:
        return []
    with ThreadPoolExecutor(max_workers=4) as pool:
        usable = list(pool.map(lambda m: is_model_usable(m, ollama_url), candidates))
    return [model for model, ok in zip(candidates, usable) if ok]


if __name__ == "__main__":
    print("\n".join(fetch_free_cloud_vision_models()))
