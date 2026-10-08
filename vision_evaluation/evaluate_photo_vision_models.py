#!/usr/bin/env python3
"""
Evaluate vision-capable Ollama models on photo description quality.

This script runs one input photo through multiple vision models and stores each
model's qualitative analysis in a text report and JSON output.

Primary target input is Sony ARW raw images. Other common image types are also
accepted for convenience.
"""

import argparse
import base64
import json
import logging
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Optional, Tuple

import requests
from PIL import Image
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

try:
    import rawpy

    RAWPY_AVAILABLE = True
except ImportError:
    RAWPY_AVAILABLE = False


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("photo_evaluation.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


VISION_MODEL_KEYWORDS = [
    "llava",
    "vision",
    "minicpm-v",
    "qwen2.5vl",
    "qwen3-vl",
    "moondream",
    "bakllava",
    "mistral-small3",
    "granite3.2-vision",
    "cogvlm",
    "pixtral",
    "gemma3",
    "gemma4",
    "llama4",
]

# Context window sizes for models that may not be locally installed in Ollama
# (e.g. cloud-hosted models where /api/show returns 404).
# Keys are lowercase substrings matched against the full model name.
KNOWN_MODEL_CTX_LIMITS: Dict[str, int] = {
    "qwen2.5vl:32b-fp16": 32768,
    "qwen3-vl": 131072,
    "qwen3.5": 131072,
    "gemma4": 131072,
}

SUPPORTED_IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".gif",
    ".webp",
    ".tif",
    ".tiff",
    ".arw",
}


@dataclass
class PhotoEvaluationResult:
    model_name: str
    description_text: str
    execution_time: float
    error_message: Optional[str] = None


@dataclass
class PhotoEvaluationSummary:
    input_photo: str
    total_models_tested: int
    evaluation_date: str
    results: List[PhotoEvaluationResult]


def is_qwen_model(model_name: str) -> bool:
    model_lower = model_name.lower()
    return "qwen" in model_lower or "qwq" in model_lower


def discover_vision_models(ollama_base_url: str) -> List[str]:
    try:
        response = requests.get(f"{ollama_base_url}/api/tags", timeout=10)
        response.raise_for_status()
        models = response.json().get("models", [])
        all_models = [model.get("name", "") for model in models]
        vision_models = [
            model
            for model in all_models
            if model
            and any(keyword in model.lower() for keyword in VISION_MODEL_KEYWORDS)
        ]
        logger.info("Discovered %s vision-capable models", len(vision_models))
        return vision_models
    except requests.RequestException as exc:
        logger.error("Failed to discover models: %s", exc)
        return []


def extract_ctx_limit(payload: Any) -> Optional[int]:
    if isinstance(payload, dict):
        for key, value in payload.items():
            key_lower = str(key).lower()
            if key_lower in {"n_ctx_train", "num_ctx", "context_length", "n_ctx"}:
                if isinstance(value, (int, float)) and int(value) > 0:
                    return int(value)
            found = extract_ctx_limit(value)
            if found:
                return found
    elif isinstance(payload, list):
        for item in payload:
            found = extract_ctx_limit(item)
            if found:
                return found
    return None


def _known_ctx_for_model(model_name: str) -> Optional[int]:
    """Return a hardcoded context size for models not available via /api/show."""
    name_lower = model_name.lower()
    for pattern, ctx in KNOWN_MODEL_CTX_LIMITS.items():
        if pattern in name_lower:
            return ctx
    return None


def get_model_ctx_limit(
    model_name: str,
    ollama_base_url: str,
    default_num_ctx: int,
    model_ctx_cache: Dict[str, int],
) -> int:
    if model_name in model_ctx_cache:
        return model_ctx_cache[model_name]

    fallback_ctx = max(1024, default_num_ctx)
    try:
        response = requests.post(
            f"{ollama_base_url}/api/show",
            json={"model": model_name},
            timeout=10,
        )
        response.raise_for_status()
        detected_ctx = extract_ctx_limit(response.json())
        if detected_ctx:
            model_ctx_cache[model_name] = detected_ctx
            return detected_ctx
    except requests.RequestException as exc:
        logger.warning("Could not read context limit for %s: %s", model_name, exc)
        known = _known_ctx_for_model(model_name)
        if known:
            logger.info("Using known context limit %s for %s", known, model_name)
            model_ctx_cache[model_name] = known
            return known

    model_ctx_cache[model_name] = fallback_ctx
    return fallback_ctx


def resolve_num_ctx(
    model_name: str,
    ollama_base_url: str,
    default_num_ctx: int,
    model_ctx_cache: Dict[str, int],
) -> int:
    model_limit = get_model_ctx_limit(
        model_name,
        ollama_base_url,
        default_num_ctx,
        model_ctx_cache,
    )
    resolved = max(1024, model_limit)
    logger.info(
        "Resolved num_ctx=%s for model=%s (model_limit=%s, default=%s)",
        resolved,
        model_name,
        model_limit,
        default_num_ctx,
    )
    return resolved


def convert_arw_to_png(input_path: Path) -> Tuple[Path, List[Path]]:
    if not RAWPY_AVAILABLE:
        raise RuntimeError(
            "rawpy is required for ARW input. Install with: pip install rawpy"
        )

    with rawpy.imread(str(input_path)) as raw:
        rgb_array = raw.postprocess(output_bps=8)

    image = Image.fromarray(rgb_array)
    if image.mode not in ("RGB", "RGBA"):
        image = image.convert("RGB")

    temp_file = tempfile.NamedTemporaryFile(suffix="_converted.png", delete=False)
    temp_path = Path(temp_file.name)
    temp_file.close()
    image.save(temp_path, format="PNG")
    return temp_path, [temp_path]


def convert_tiff_to_png(input_path: Path) -> Tuple[Path, List[Path]]:
    with Image.open(input_path) as image:
        if image.mode not in ("RGB", "RGBA"):
            image = image.convert("RGB")
        temp_file = tempfile.NamedTemporaryFile(suffix="_converted.png", delete=False)
        temp_path = Path(temp_file.name)
        temp_file.close()
        image.save(temp_path, format="PNG")
    return temp_path, [temp_path]


def prepare_image_path(photo_path: Path) -> Tuple[Path, List[Path]]:
    ext = photo_path.suffix.lower()
    if ext not in SUPPORTED_IMAGE_EXTENSIONS:
        raise RuntimeError(f"Unsupported image type: {ext}")

    if ext == ".arw":
        return convert_arw_to_png(photo_path)

    if ext in {".tif", ".tiff"}:
        return convert_tiff_to_png(photo_path)

    return photo_path, []


def image_to_base64(image_path: Path) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def get_mime_type(image_path: Path) -> str:
    ext = image_path.suffix.lower()
    if ext in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if ext == ".gif":
        return "image/gif"
    if ext == ".webp":
        return "image/webp"
    if ext == ".bmp":
        return "image/bmp"
    return "image/png"


def build_photo_analysis_prompt() -> str:
    return (
        "Analyze this photograph and provide a structured response with the "
        "following sections:\n\n"
        "1. Keywords\n"
        "Keywords that apply tothe photograph.\n\n"
        "2. What Is In The Photo\n"
        "Describe the main subjects, objects, and scene details.\n\n"
        "3. Photographic Composition\n"
        "Identify composition techniques (rule of thirds, leading lines, "
        "framing, symmetry, depth, perspective, etc.) and explain the effect.\n\n"
        "4. Probable Location And Context\n"
        "Infer where this was likely taken (indoor/outdoor, environment type, "
        "region clues) and why.\n\n"
        "5. Colors And Palette\n"
        "Describe dominant colors, contrast, saturation, color temperature, and "
        "overall palette mood.\n\n"
        "6. Visual Style\n"
        "Characterize style (documentary, portrait, street, landscape, fine art, "
        "commercial, cinematic, minimal, etc.) with supporting cues.\n\n"
        "7. Artistic Merit\n"
        "Give a reasoned judgment of artistic merit. Include strengths, "
        "weaknesses, and a score from 1 to 10 with rationale.\n\n"
        "If uncertain, state likely interpretations and confidence.\n\n"
        "8. focus and noise\n"
        "Evaluate focus quality and noise levels, noting any areas of softness.\n\n" 
        "9. Recommended Improvements\n"
        "Suggest specific ways to improve the photograph through post-processing."
    )


def create_vision_llm(
    model_name: str,
    temperature: float,
    ollama_base_url: str,
    request_timeout: int,
    num_ctx: int,
) -> ChatOllama:
    common_kwargs: Dict[str, Any] = {
        "model": model_name,
        "temperature": temperature,
        "base_url": ollama_base_url,
        "num_ctx": num_ctx,
        "num_predict": 1024,
        "client_kwargs": {"timeout": request_timeout},
        "sync_client_kwargs": {"timeout": request_timeout},
        "async_client_kwargs": {"timeout": request_timeout},
    }

    if is_qwen_model(model_name):
        common_kwargs.update({"top_p": 0.8, "top_k": 20, "min_p": 0.0})

    return ChatOllama(**common_kwargs)


def evaluate_single_model(
    model_name: str,
    prepared_image_path: Path,
    ollama_base_url: str,
    request_timeout: int,
    default_num_ctx: int,
    temperature: float,
    model_ctx_cache: Dict[str, int],
) -> PhotoEvaluationResult:
    start = perf_counter()
    prompt_text = build_photo_analysis_prompt()

    try:
        num_ctx = resolve_num_ctx(
            model_name,
            ollama_base_url,
            default_num_ctx,
            model_ctx_cache,
        )
        llm = create_vision_llm(
            model_name=model_name,
            temperature=temperature,
            ollama_base_url=ollama_base_url,
            request_timeout=request_timeout,
            num_ctx=num_ctx,
        )

        image_b64 = image_to_base64(prepared_image_path)
        mime_type = get_mime_type(prepared_image_path)

        message = HumanMessage(
            content=[
                {
                    "type": "image_url",
                    "image_url": f"data:{mime_type};base64,{image_b64}",
                },
                {"type": "text", "text": prompt_text},
            ]
        )

        response = llm.invoke([message])
        response_text = response.content if hasattr(response, "content") else str(response)
        elapsed = perf_counter() - start

        logger.info(
            "Model %s completed in %.2fs (response chars: %s)",
            model_name,
            elapsed,
            len(response_text),
        )
        return PhotoEvaluationResult(
            model_name=model_name,
            description_text=response_text or "",
            execution_time=elapsed,
        )
    except Exception as exc:
        elapsed = perf_counter() - start
        error_text = str(exc)
        logger.error("Model %s failed: %s", model_name, error_text)
        return PhotoEvaluationResult(
            model_name=model_name,
            description_text=f"[Error: {error_text}]",
            execution_time=elapsed,
            error_message=error_text,
        )


def evaluate_photo_models(
    photo_path: Path,
    output_dir: Path,
    ollama_base_url: str,
    request_timeout: int,
    default_num_ctx: int,
    temperature: float,
    explicit_models: Optional[List[str]],
) -> PhotoEvaluationSummary:
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_models = explicit_models or discover_vision_models(ollama_base_url)
    if not selected_models:
        raise RuntimeError("No vision-capable models found")

    prepared_image_path, temporary_files = prepare_image_path(photo_path)
    model_ctx_cache: Dict[str, int] = {}

    try:
        results = [
            evaluate_single_model(
                model_name=model_name,
                prepared_image_path=prepared_image_path,
                ollama_base_url=ollama_base_url,
                request_timeout=request_timeout,
                default_num_ctx=default_num_ctx,
                temperature=temperature,
                model_ctx_cache=model_ctx_cache,
            )
            for model_name in selected_models
        ]
    finally:
        for temp_file in temporary_files:
            try:
                os.unlink(temp_file)
            except OSError:
                pass

    return PhotoEvaluationSummary(
        input_photo=str(photo_path),
        total_models_tested=len(results),
        evaluation_date=datetime.now().isoformat(),
        results=results,
    )


def write_json_output(summary: PhotoEvaluationSummary, output_dir: Path, timestamp: str) -> Path:
    json_path = output_dir / f"evaluation_results_{timestamp}.json"
    payload = {
        "input_photo": summary.input_photo,
        "total_models_tested": summary.total_models_tested,
        "evaluation_date": summary.evaluation_date,
        "results": [
            {
                "model_name": result.model_name,
                "description_text": result.description_text,
                "execution_time": result.execution_time,
                "error_message": result.error_message,
            }
            for result in summary.results
        ],
    }

    with open(json_path, "w", encoding="utf-8") as json_file:
        json.dump(payload, json_file, indent=2, ensure_ascii=False)

    return json_path


def write_report_output(summary: PhotoEvaluationSummary, output_dir: Path, timestamp: str) -> Path:
    report_path = output_dir / f"evaluation_report_{timestamp}.txt"

    with open(report_path, "w", encoding="utf-8") as report_file:
        report_file.write("=" * 80 + "\n")
        report_file.write("PHOTO DESCRIPTION EVALUATION REPORT\n")
        report_file.write("=" * 80 + "\n\n")
        report_file.write(f"Input Photo: {summary.input_photo}\n")
        report_file.write(f"Models Tested: {summary.total_models_tested}\n")
        report_file.write(f"Evaluation Date: {summary.evaluation_date}\n\n")

        report_file.write("MODEL OUTPUTS\n")
        report_file.write("-" * 80 + "\n")

        for result in summary.results:
            report_file.write(f"\nModel: {result.model_name}\n")
            report_file.write(f"Execution Time: {result.execution_time:.2f}s\n")
            if result.error_message:
                report_file.write(f"Status: ERROR\n")
                report_file.write(f"Error: {result.error_message}\n")
            else:
                report_file.write("Status: OK\n")

            report_file.write("\nComplete Description:\n")
            report_file.write("-" * 60 + "\n")
            if result.description_text:
                report_file.write(result.description_text)
                if not result.description_text.endswith("\n"):
                    report_file.write("\n")
            else:
                report_file.write("[No output]\n")
            report_file.write("-" * 60 + "\n")

    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate vision models on qualitative photo description"
    )
    parser.add_argument("photo_path", help="Path to input photo (ARW preferred)")
    parser.add_argument(
        "--output-dir",
        default="evaluation_results_photo",
        help="Output directory for report and JSON",
    )
    parser.add_argument(
        "--ollama-url",
        default="http://localhost:11434",
        help="Ollama server URL",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="Specific models to test (defaults to all discovered vision models)",
    )
    parser.add_argument(
        "--request-timeout",
        type=int,
        default=120,
        help="Per-request timeout in seconds",
    )
    parser.add_argument(
        "--num-ctx",
        type=int,
        default=8192,
        help="Default context window cap for model requests",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Model temperature for photo description generation",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    photo_path = Path(args.photo_path)
    if not photo_path.exists():
        print(f"Error: photo file not found: {photo_path}")
        return 1

    try:
        summary = evaluate_photo_models(
            photo_path=photo_path,
            output_dir=Path(args.output_dir),
            ollama_base_url=args.ollama_url,
            request_timeout=max(10, args.request_timeout),
            default_num_ctx=max(1024, args.num_ctx),
            temperature=args.temperature,
            explicit_models=args.models,
        )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = write_json_output(summary, Path(args.output_dir), timestamp)
        report_path = write_report_output(summary, Path(args.output_dir), timestamp)

        print("\n" + "=" * 60)
        print("PHOTO EVALUATION COMPLETE")
        print("=" * 60)
        print(f"Models tested: {summary.total_models_tested}")
        print(f"Report: {report_path}")
        print(f"JSON: {json_path}")

        return 0
    except Exception as exc:
        logger.error("Evaluation failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
