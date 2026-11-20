"""Discover NVIDIA-hosted vision models and build LangChain chat clients."""

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List, Optional

import requests

API_KEY_ENV = "NVIDIA_API_KEY"
CATALOG_URL = "https://build.nvidia.com"
KNOWN_MULTIMODAL_MODEL_IDS = {
    "deepseek-ai/deepseek-v4.1-flash",
    "google/paligemma",
    "microsoft/phi-3-vision-128k-instruct",
    "mistral-large-4",
    "stepfun-ai/step-5-preview",
    "z-ai/glm-5.3-flash",
    "meta/muse-glimmer-30b",
}
KNOWN_REASONING_TEXT_MODEL_IDS = {
    "z-ai/glm-5.3-flash",
    "meta/muse-glimmer-30b",
}
CATALOG_MODEL_PATHS = {
    "google/paligemma": "google/google-paligemma",
    "mistral-large-4": "mistralai/mistral-large-4-1t-a52b",
    "stepfun-ai/step-5-preview": "stepfun-ai/step-5-preview",
    "z-ai/glm-5.3-flash": "z-ai/glm-5-3-flash",
}
GLM_REASONING_EFFORT = "max"
NVIDIA_LONG_RUNNING_VISION_MODEL_IDS = {
    "deepseek-ai/deepseek-v4.1-flash",
    "z-ai/glm-5.3-flash",
}
NVIDIA_NON_STREAMING_MODEL_IDS = {
    "microsoft/phi-3-vision-128k-instruct",
}
NON_TEXT_MODEL_TYPES = {"embedding", "ranking", "reranker", "qa", "image_in", "image_out"}
NON_TEXT_NAME_HINTS = (
    "embed", "rerank", "retriever", "parse", "guard", "safety", "reward",
    "clip", "paligemma", "neva", "kosmos", "vila", "riva", "asr", "tts",
    "translate", "diffusion", "flux", "stable-", "cosmos", "nemoretriever",
)
NVIDIA_VISION_MAX_TOKENS = 4096
NVIDIA_CHAT_COMPLETIONS_ENDPOINT = "https://integrate.api.nvidia.com/v1/chat/completions"


def _register_glm_53_flash() -> None:
    """Register GLM-5.3-Flash as a multimodal ChatNVIDIA model."""
    from langchain_nvidia_ai_endpoints import Model, register_model
    from langchain_nvidia_ai_endpoints._statics import determine_model

    current = determine_model("z-ai/glm-5.3-flash")
    if (
        current
        and current.model_type == "vlm"
        and current.endpoint == NVIDIA_CHAT_COMPLETIONS_ENDPOINT
    ):
        return
    register_model(
        Model(
            id="z-ai/glm-5.3-flash",
            model_type="vlm",
            client="ChatNVIDIA",
            endpoint=NVIDIA_CHAT_COMPLETIONS_ENDPOINT,
            supports_thinking=True,
        )
    )


def _register_mistral_large_4() -> None:
    """Register Mistral Large 4 as a multimodal ChatNVIDIA model."""
    from langchain_nvidia_ai_endpoints import Model, register_model
    from langchain_nvidia_ai_endpoints._statics import determine_model

    current = determine_model("mistral-large-4")
    if (
        current
        and current.model_type == "vlm"
        and current.endpoint == NVIDIA_CHAT_COMPLETIONS_ENDPOINT
    ):
        return
    register_model(
        Model(
            id="mistral-large-4",
            model_type="vlm",
            client="ChatNVIDIA",
            endpoint=NVIDIA_CHAT_COMPLETIONS_ENDPOINT,
        )
    )


def _register_step_5_preview() -> None:
    """Register Step 5 Preview as a multimodal ChatNVIDIA model."""
    from langchain_nvidia_ai_endpoints import Model, register_model
    from langchain_nvidia_ai_endpoints._statics import determine_model

    current = determine_model("stepfun-ai/step-5-preview")
    if (
        current
        and current.model_type == "vlm"
        and current.endpoint == NVIDIA_CHAT_COMPLETIONS_ENDPOINT
    ):
        return
    register_model(
        Model(
            id="stepfun-ai/step-5-preview",
            model_type="vlm",
            client="ChatNVIDIA",
            endpoint=NVIDIA_CHAT_COMPLETIONS_ENDPOINT,
            supports_thinking=True,
        )
    )


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
        if self.model_id == "mistral-large-4":
            return "Mistral Large 4 [FREE]"
        if self.model_id == "stepfun-ai/step-5-preview":
            return "Step 5 Preview [FREE]"
        return f"{self.model_id} [FREE]"


def get_api_key(explicit: Optional[str] = None) -> Optional[str]:
    return explicit or os.environ.get(API_KEY_ENV)


def should_stream_model(model_id: str) -> bool:
    """Return whether the model should use NVIDIA's streaming response path."""
    return model_id not in NVIDIA_NON_STREAMING_MODEL_IDS


def has_hosted_endpoint(model_id: str, timeout: int = 15) -> bool:
    """Return False only when build.nvidia.com says no hosted endpoint exists.

    /v1/models can still list models whose hosted function was removed; calling
    them fails with "[404] Function ... Not found for account". The model's
    ?nim=hosted page returns 404 for those. Any other outcome (network error,
    5xx) keeps the model so a transient failure does not hide valid models.
    """
    try:
        response = requests.get(
            f"{CATALOG_URL}/{CATALOG_MODEL_PATHS.get(model_id, model_id)}",
            params={"nim": "hosted"},
            headers={"Accept": "text/markdown"},
            timeout=timeout,
        )
    except requests.RequestException:
        return True
    return response.status_code != 404


def fetch_vision_models(
    api_key: Optional[str] = None,
    include_unlisted: bool = False,
    verify_hosted: bool = True,
) -> List[NVIDIAVisionModel]:
    """Return vision models that NVIDIA's live /v1/models listing reports.

    Set include_unlisted to also return models only present in the LangChain
    static table (these may be end-of-life). verify_hosted drops models that
    build.nvidia.com no longer offers as a hosted endpoint. Per-account
    activation is still checked by NVIDIA when an inference request is made.
    """
    key = get_api_key(api_key)
    if not key:
        raise RuntimeError(
            f"NVIDIA API key missing: set {API_KEY_ENV} or enter it in the GUI"
        )

    from langchain_nvidia_ai_endpoints import ChatNVIDIA

    _register_glm_53_flash()
    _register_mistral_large_4()
    _register_step_5_preview()
    client = ChatNVIDIA(api_key=key)
    # ChatNVIDIA.available_models mixes the library's static model table into
    # the live /v1/models listing, so retired models (e.g. past end of life)
    # linger. The live listing is the authoritative source for what is served.
    # client.available_models also drops live models the library does not know
    # (their client is unset), so start from the raw live listing instead.
    candidates = list(client._client.available_models)
    if include_unlisted:
        candidates += client.available_models
    else:
        # These hosted multimodal endpoints may be absent from /v1/models but
        # remain available through the LangChain integration's static catalog.
        candidates += [
            model
            for model in client.available_models
            if getattr(model, "id", None) in KNOWN_MULTIMODAL_MODEL_IDS
        ]
    models = {}
    for model in candidates:
        model_type = (getattr(model, "model_type", None) or "").lower()
        model_id = getattr(model, "id", None)
        # The NVIDIA listing contains IDs but no capability metadata. Unknown
        # models default to "chat" in LangChain, so document-confirmed
        # multimodal models need an explicit override until the library catches up.
        is_vision = (
            "vlm" in model_type or model_id in KNOWN_MULTIMODAL_MODEL_IDS
        )
        if (
            model_id
            and is_vision
            and not getattr(model, "deprecated", False)
        ):
            models[model_id] = NVIDIAVisionModel(model_id)

    # Explicitly include known multimodal catalog models even when the live
    # /v1/models response omits them or the integration has no type metadata.
    for model_id in KNOWN_MULTIMODAL_MODEL_IDS:
        models.setdefault(model_id, NVIDIAVisionModel(model_id))

    model_ids = sorted(models)
    if verify_hosted:
        with ThreadPoolExecutor(max_workers=8) as pool:
            hosted = list(pool.map(has_hosted_endpoint, model_ids))
        model_ids = [m for m, ok in zip(model_ids, hosted) if ok]

    return [models[model_id] for model_id in model_ids]


def fetch_text_models(api_key: Optional[str] = None) -> List[str]:
    """Return live NVIDIA chat models that can serve as text judges."""
    key = get_api_key(api_key)
    if not key:
        raise RuntimeError(
            f"NVIDIA API key missing: set {API_KEY_ENV} or enter it in the GUI"
        )

    from langchain_nvidia_ai_endpoints import ChatNVIDIA

    _register_glm_53_flash()
    client = ChatNVIDIA(api_key=key)
    candidates = list(client._client.available_models) + list(client.available_models)
    models = {}
    for model in candidates:
        model_id = getattr(model, "id", None)
        model_type = (getattr(model, "model_type", None) or "").lower()
        # The live listing leaves model_type unset for many chat models, so
        # exclude known non-generative types/names instead of requiring "chat".
        # Multimodal models also handle text prompts.
        if (
            model_id
            and model_type not in NON_TEXT_MODEL_TYPES
            and not any(s in model_id.lower() for s in NON_TEXT_NAME_HINTS)
            and not getattr(model, "deprecated", False)
        ):
            models[model_id] = model_id

    for model_id in KNOWN_REASONING_TEXT_MODEL_IDS:
        if has_hosted_endpoint(model_id):
            models[model_id] = model_id
    return sorted(models)


def create_chat_model(
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    timeout: int = 300,
    api_key: Optional[str] = None,
):
    """Create a LangChain ChatNVIDIA client."""
    key = get_api_key(api_key)
    if not key:
        raise RuntimeError(
            f"NVIDIA API key missing: set {API_KEY_ENV} or pass --nvidia-api-key"
        )

    from langchain_nvidia_ai_endpoints import ChatNVIDIA

    _register_glm_53_flash()
    _register_mistral_large_4()
    _register_step_5_preview()
    if model == "google/paligemma":
        # PaLI-Gemma's NVIDIA endpoint accepts at most 1,024 output tokens.
        max_tokens = min(max_tokens, 1024)
        # The endpoint rejects temperature=0 (requires > 0).
        temperature = max(temperature, 0.01)
    elif model in NVIDIA_LONG_RUNNING_VISION_MODEL_IDS:
        # NVIDIA's hosted gateway times out at about five minutes while these
        # reasoning models are allowed to generate a 16k-token response.
        max_tokens = min(max_tokens, NVIDIA_VISION_MAX_TOKENS)
    elif model == "microsoft/phi-3-vision-128k-instruct":
        # Keep the legacy Phi endpoint's response bounded. Its streamed
        # connection is also unreliable on Windows clients.
        max_tokens = min(max_tokens, NVIDIA_VISION_MAX_TOKENS)

    extra = {}
    if model == "z-ai/glm-5.3-flash":
        # reasoning_effort defaults to "max" server-side, which runs past the
        # gateway's ~300s limit and returns 504. Use the lowest effort.
        extra["model_kwargs"] = {
            "reasoning_effort": GLM_REASONING_EFFORT,
            "chat_template_kwargs": {"clear_thinking": True},
        }

    return ChatNVIDIA(
        model=model,
        api_key=key,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        **extra,
    )
