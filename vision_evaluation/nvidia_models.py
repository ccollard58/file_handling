"""Discover NVIDIA-hosted vision models and build LangChain chat clients."""

import os
from dataclasses import dataclass
from typing import List, Optional

API_KEY_ENV = "NVIDIA_API_KEY"


@dataclass(frozen=True)
class NVIDIAVisionModel:
    model_id: str

    @property
    def is_free(self) -> bool:
        # NVIDIA API Catalog hosted models are available to Developer accounts
        # for prototyping within the account's usage limits.
        return True

    @property
    def display_name(self) -> str:
        return f"{self.model_id} [FREE]"


def get_api_key(explicit: Optional[str] = None) -> Optional[str]:
    return explicit or os.environ.get(API_KEY_ENV)


def fetch_vision_models(api_key: Optional[str] = None) -> List[NVIDIAVisionModel]:
    """Return non-deprecated vision models known to ChatNVIDIA.

    The LangChain catalog does not verify per-account model activation. NVIDIA
    checks that access when an inference request is made.
    """
    key = get_api_key(api_key)
    if not key:
        raise RuntimeError(
            f"NVIDIA API key missing: set {API_KEY_ENV} or enter it in the GUI"
        )

    from langchain_nvidia_ai_endpoints import ChatNVIDIA

    client = ChatNVIDIA(api_key=key)
    models = {}
    for model in client.available_models:
        model_type = (getattr(model, "model_type", None) or "").lower()
        model_id = getattr(model, "id", None)
        if (
            model_id
            and "vlm" in model_type
            and not getattr(model, "deprecated", False)
        ):
            models[model_id] = NVIDIAVisionModel(model_id)

    return [models[model_id] for model_id in sorted(models)]


def create_chat_model(
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    timeout: int = 120,
    api_key: Optional[str] = None,
):
    """Create a LangChain ChatNVIDIA client."""
    key = get_api_key(api_key)
    if not key:
        raise RuntimeError(
            f"NVIDIA API key missing: set {API_KEY_ENV} or pass --nvidia-api-key"
        )

    from langchain_nvidia_ai_endpoints import ChatNVIDIA

    return ChatNVIDIA(
        model=model,
        api_key=key,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    )
