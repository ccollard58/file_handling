#!/usr/bin/env python3
"""
LLM-as-a-Judge Evaluation of Vision Models for Handwritten Text Extraction

This script performs automated evaluation of multiple vision-capable LLMs running via Ollama,
specifically assessing their ability to extract handwritten text from multi-page scanned 
documents (PDFs with embedded images). The evaluation process leverages an LLM-as-a-judge 
framework to benchmark model performance against ground truth.

Author: GitHub Copilot
Date: July 17, 2025
"""

import os
import json
import requests
import logging
import argparse
import tempfile
import re
import html
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import statistics

# LangChain imports
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
import base64
import csv

# PDF and image processing
try:
    import fitz  # PyMuPDF
    FITZ_AVAILABLE = True
except ImportError:
    FITZ_AVAILABLE = False

try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False

from PIL import Image

# Text comparison metrics
import difflib
try:
    import Levenshtein
    LEVENSHTEIN_AVAILABLE = True
except ImportError:
    LEVENSHTEIN_AVAILABLE = False

try:
    from evaluate import load
    EVALUATE_AVAILABLE = True
except ImportError:
    EVALUATE_AVAILABLE = False

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('handwriting_evaluation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Context window sizes for models that may not be locally installed in Ollama
# (e.g. cloud-hosted models where /api/show returns 404).
# Keys are lowercase substrings matched against the full model name.
KNOWN_MODEL_CTX_LIMITS: Dict[str, int] = {
    "qwen2.5vl:32b-fp16": 32768,
    "qwen3-vl": 262144,
    "qwen3.5": 262144,
    "gemma4": 262144,
}


@dataclass
class EvaluationResult:
    """Stores the evaluation results for a single model."""
    model_name: str
    extracted_text: str
    levenshtein_distance: float
    word_error_rate: float
    character_accuracy: float
    semantic_similarity: float
    execution_time: float
    error_message: Optional[str] = None
    temperature: Optional[float] = None


@dataclass
class EvaluationSummary:
    """Stores the complete evaluation summary."""
    input_pdf: str
    ground_truth_file: str
    total_models_tested: int
    evaluation_date: str
    results: List[EvaluationResult]
    best_model: str
    best_score: float
    ranking: List[Tuple[str, float]]


class VisionModelEvaluator:
    """
    Evaluates multiple vision-capable LLMs for handwritten text extraction.
    """
    
    def __init__(
        self,
        ollama_base_url: str = "http://localhost:11434",
        judge_model: str = "gpt-oss:120b",
        request_timeout: int = 120,
        default_num_ctx: int = 131072,
        provider: str = "ollama",
        openrouter_api_key: Optional[str] = None,
        openrouter_base_url: str = "https://openrouter.ai/api/v1",
    ):
        """
        Initialize the evaluator.
        
        Args:
            ollama_base_url: URL of the Ollama server
            judge_model: Model to use for LLM-as-a-judge evaluations
            provider: "ollama" or "openrouter"
        """
        self.provider = provider
        self.openrouter_api_key = openrouter_api_key
        self.openrouter_base_url = openrouter_base_url
        self.ollama_base_url = ollama_base_url
        self.judge_model = judge_model
        self.judge_llm = None
        self.request_timeout = request_timeout
        self.default_num_ctx = default_num_ctx
        self._model_ctx_cache: Dict[str, int] = {}
        # Single log files per evaluator instance to keep outputs together
        _ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._vision_log_path = f"vision_llm_outputs_{_ts}.log"
        self._judge_log_path = f"judge_llm_outputs_{_ts}.log"
        self.poppler_path = self._find_poppler_path()
        
        # Initialize judge LLM
        self._initialize_judge_llm()
        
        # Ensure required libraries are available
        self._check_dependencies()
    
    def _find_poppler_path(self) -> Optional[str]:
        """Find Poppler path on Windows."""
        possible_paths = [
            r"C:\Program Files\poppler\bin",
            r"C:\Program Files (x86)\poppler\bin",
            r"C:\poppler\bin",
            r"C:\tools\poppler\bin"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        return None
    
    def _check_dependencies(self):
        """Check if required dependencies are available."""
        missing_deps = []
        
        if not (FITZ_AVAILABLE or PDF2IMAGE_AVAILABLE):
            missing_deps.append("PDF processing (fitz or pdf2image)")
        
        if not LEVENSHTEIN_AVAILABLE:
            logger.warning("Levenshtein package not available. Installing python-Levenshtein is recommended.")
        
        if missing_deps:
            raise RuntimeError(f"Missing required dependencies: {', '.join(missing_deps)}")
    
    def _initialize_judge_llm(self):
        """Initialize the LLM-as-a-judge model."""
        try:
            if self.provider == "openrouter":
                from openrouter_models import create_chat_model
                self.judge_llm = create_chat_model(
                    self.judge_model, temperature=0.1, max_tokens=256,
                    timeout=self.request_timeout, base_url=self.openrouter_base_url,
                    api_key=self.openrouter_api_key,
                )
                logger.info(f"OpenRouter judge LLM initialized with model: {self.judge_model}")
                return
            judge_kwargs = {
                "model": self.judge_model,
                "temperature": 0.1,
                "base_url": self.ollama_base_url,
                "num_ctx": self._resolve_num_ctx(self.judge_model),
                "num_predict": 256,
                "client_kwargs": {"timeout": self.request_timeout},
                "sync_client_kwargs": {"timeout": self.request_timeout},
                "async_client_kwargs": {"timeout": self.request_timeout},
            }
            if self._is_qwen_model(self.judge_model):
                judge_kwargs.update({"top_p": 0.8, "top_k": 20, "min_p": 0.0})

            self.judge_llm = ChatOllama(**judge_kwargs)
            logger.info(f"Judge LLM initialized with model: {self.judge_model}")
        except Exception as e:
            logger.error(f"Failed to initialize judge LLM: {e}")
            self.judge_llm = None

    def _extract_ctx_limit(self, payload: Any) -> Optional[int]:
        """Recursively search Ollama model metadata for context-length fields."""
        if isinstance(payload, dict):
            for key, value in payload.items():
                key_lower = str(key).lower()
                if key_lower in {"n_ctx_train", "num_ctx", "context_length", "n_ctx"}:
                    if isinstance(value, (int, float)) and int(value) > 0:
                        return int(value)
                found = self._extract_ctx_limit(value)
                if found:
                    return found
        elif isinstance(payload, list):
            for item in payload:
                found = self._extract_ctx_limit(item)
                if found:
                    return found
        return None

    def _get_model_ctx_limit(self, model_name: str) -> int:
        """Get model context limit from Ollama metadata, with safe fallback."""
        if model_name in self._model_ctx_cache:
            return self._model_ctx_cache[model_name]

        fallback_ctx = max(262144, self.default_num_ctx)
        try:
            response = requests.post(
                f"{self.ollama_base_url}/api/show",
                json={"model": model_name},
                timeout=10,
            )
            response.raise_for_status()
            model_data = response.json()
            detected_ctx = self._extract_ctx_limit(model_data)
            if detected_ctx:
                self._model_ctx_cache[model_name] = detected_ctx
                return detected_ctx
        except Exception as exc:
            logger.warning(f"Could not read context limit for {model_name}: {exc}")
            known = self._known_ctx_for_model(model_name)
            if known:
                logger.info(f"Using known context limit {known} for {model_name}")
                self._model_ctx_cache[model_name] = known
                return known

        self._model_ctx_cache[model_name] = fallback_ctx
        return fallback_ctx

    def _known_ctx_for_model(self, model_name: str) -> Optional[int]:
        """Return a hardcoded context size for models not available via /api/show."""
        name_lower = model_name.lower()
        for pattern, ctx in KNOWN_MODEL_CTX_LIMITS.items():
            if pattern in name_lower:
                return ctx
        return None

    def _resolve_num_ctx(self, model_name: str) -> int:
        """Resolve num_ctx to match the model's actual context limit."""
        model_limit = self._get_model_ctx_limit(model_name)
        resolved = max(262144, model_limit)
        logger.info(
            "Resolved num_ctx=%s for model=%s (model_limit=%s, default=%s)",
            resolved,
            model_name,
            model_limit,
            self.default_num_ctx,
        )
        return resolved
    
    def _log_vision_llm_output(
        self,
        model_name: str,
        page_number: int,
        prompt: str,
        response: str,
        temperature: float = 0.0,
        response_time_seconds: Optional[float] = None,
    ):
        """Log the complete vision LLM output to a detailed log file."""
        try:
            with open(self._vision_log_path, 'a', encoding='utf-8') as f:
                f.write("=" * 120 + "\n")
                f.write(f"TIMESTAMP: {datetime.now().isoformat()}\n")
                f.write(f"MODEL: {model_name}\n")
                f.write(f"TEMPERATURE: {temperature}\n")
                f.write(f"PAGE: {page_number}\n")
                if response_time_seconds is not None:
                    f.write(f"RESPONSE_TIME_SECONDS: {response_time_seconds:.3f}\n")
                f.write(f"RESPONSE_LENGTH: {len(response)} characters\n")
                f.write("-" * 60 + "\n")
                f.write("PROMPT SENT TO MODEL:\n")
                f.write(prompt)
                f.write("\n" + "-" * 60 + "\n")
                f.write("COMPLETE MODEL RESPONSE:\n")
                f.write(response)
                f.write("\n" + "=" * 120 + "\n\n")
        except Exception as e:
            logger.error(f"Error writing vision LLM output log: {e}")
    
    def _log_judge_llm_output(
        self,
        reference: str,
        hypothesis: str,
        prompt: str,
        response: str,
        final_score: float,
        response_time_seconds: Optional[float] = None,
    ):
        """Log the complete judge LLM output to a detailed log file."""
        try:
            with open(self._judge_log_path, 'a', encoding='utf-8') as f:
                f.write("=" * 120 + "\n")
                f.write(f"TIMESTAMP: {datetime.now().isoformat()}\n")
                f.write(f"JUDGE MODEL: {self.judge_model}\n")
                if response_time_seconds is not None:
                    f.write(f"RESPONSE_TIME_SECONDS: {response_time_seconds:.3f}\n")
                f.write(f"FINAL SCORE: {final_score}\n")
                f.write(f"REFERENCE_LENGTH: {len(reference)} characters\n")
                f.write(f"HYPOTHESIS_LENGTH: {len(hypothesis)} characters\n")
                f.write("-" * 60 + "\n")
                f.write("PROMPT SENT TO JUDGE:\n")
                f.write(prompt)
                f.write("\n" + "-" * 60 + "\n")
                f.write("JUDGE RESPONSE:\n")
                f.write(response)
                f.write("\n" + "-" * 60 + "\n")
                f.write("REFERENCE TEXT (first 500 chars):\n")
                f.write(reference[:500])
                f.write("\n" + "-" * 60 + "\n")
                f.write("HYPOTHESIS TEXT (first 500 chars):\n")
                f.write(hypothesis[:500])
                f.write("\n" + "=" * 120 + "\n\n")
        except Exception as e:
            logger.error(f"Error writing judge LLM output log: {e}")
    
    def discover_vision_models(self) -> List[str]:
        """
        Discover all available vision-capable models in Ollama.
        
        Returns:
            List of vision model names
        """
        try:
            if self.provider == "openrouter":
                from openrouter_models import fetch_free_vision_models
                return fetch_free_vision_models(self.openrouter_base_url, api_key=self.openrouter_api_key)
            response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=10)
            response.raise_for_status()
            models = response.json().get("models", [])
            all_models = [model["name"] for model in models]
            
            # Filter for vision-capable models based on Ollama's vision models
            # Updated list from https://ollama.com/search?c=vision
            vision_keywords = [
                'llava',           # LLaVA models (llava, llava-llama3, llava-phi3)
                'vision',          # Models with 'vision' in name (llama3.2-vision, granite3.2-vision)
                'minicpm-v',       # MiniCPM vision models
                'qwen2.5vl',       # Qwen vision-language models
                'qwen3-vl',        # Qwen3 vision-language models
                'moondream',       # Moondream vision models
                'bakllava',        # BakLLaVA models
                'mistral-small3',  # Mistral Small 3.1/3.2 with vision
                'granite3.2-vision', # Granite vision models
                'cogvlm',          # CogVLM models
                'pixtral',         # Pixtral models
                'gemma3',          # Gemma3 has vision capabilities
                'llama4',          # Llama4 multimodal models
                'glm-ocr',
                'translategemma'
            ]
            
            vision_models = []
            for model in all_models:
                model_lower = model.lower()
                if any(keyword in model_lower for keyword in vision_keywords):
                    vision_models.append(model)
            
            logger.info(f"Discovered {len(vision_models)} vision-capable models: {vision_models}")
            return vision_models
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error discovering Ollama models: {e}")
            return []
    
    def pdf_to_images(self, pdf_path: str) -> List[str]:
        """
        Convert PDF pages to images.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of temporary image file paths
        """
        image_paths = []
        
        try:
            if PDF2IMAGE_AVAILABLE:
                # Use pdf2image (preferred method)
                if self.poppler_path:
                    images = convert_from_path(pdf_path, poppler_path=self.poppler_path, dpi=300)
                else:
                    images = convert_from_path(pdf_path, dpi=300)
                
                for i, image in enumerate(images):
                    temp_file = tempfile.NamedTemporaryFile(suffix=f'_page_{i+1}.png', delete=False)
                    image.save(temp_file.name, 'PNG')
                    image_paths.append(temp_file.name)
                    temp_file.close()
                    
            elif FITZ_AVAILABLE:
                # Fallback to PyMuPDF
                doc = fitz.open(pdf_path)
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better quality
                    temp_file = tempfile.NamedTemporaryFile(suffix=f'_page_{page_num+1}.png', delete=False)
                    pix.save(temp_file.name)
                    image_paths.append(temp_file.name)
                    temp_file.close()
                doc.close()
            
            logger.info(f"Converted PDF to {len(image_paths)} images")
            return image_paths
            
        except Exception as e:
            logger.error(f"Error converting PDF to images: {e}")
            return []
    
    def _image_to_base64(self, image_path: str) -> str:
        """Convert an image file to base64 encoded string."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    
    def _is_qwen_model(self, model_name: str) -> bool:
        """Check if model is a Qwen model requiring special hyperparameters."""
        model_lower = model_name.lower()
        return 'qwen' in model_lower or 'qwq' in model_lower
    
    def _create_vision_llm(self, model_name: str, temperature: float = 0.0):
        """Create a chat client (Ollama or OpenRouter) with model-specific hyperparameters."""
        if self.provider == "openrouter":
            from openrouter_models import create_chat_model
            return create_chat_model(
                model_name, temperature=temperature, max_tokens=1024,
                timeout=self.request_timeout, base_url=self.openrouter_base_url,
                api_key=self.openrouter_api_key,
            )
        common_kwargs = {
            "model": model_name,
            "temperature": temperature,
            "base_url": self.ollama_base_url,
            "num_ctx": self._resolve_num_ctx(model_name),
            "num_predict": 1024,
            "client_kwargs": {"timeout": self.request_timeout},
            "sync_client_kwargs": {"timeout": self.request_timeout},
            "async_client_kwargs": {"timeout": self.request_timeout},
        }

        if self._is_qwen_model(model_name):
            # Qwen models require specific hyperparameters
            logger.info(f"Using Qwen-specific hyperparameters for {model_name}")
            return ChatOllama(
                **common_kwargs,
                top_p=0.8,
                top_k=20,
                min_p=0.0
            )
        return ChatOllama(**common_kwargs)
    
    def extract_text_with_vision_model(self, model_name: str, image_paths: List[str], temperature: float = 0.0) -> Tuple[str, float]:
        """
        Extract text from images using a specific vision model.
        
        Uses ChatOllama with base64-encoded images and HumanMessage for multimodal input.
        
        Args:
            model_name: Name of the vision model
            image_paths: List of image file paths
            temperature: Temperature setting for the model (default: 0.0)
            
        Returns:
            Tuple of (extracted text, execution time in seconds)
        """
        start_time = datetime.now()
        
        try:
            # Initialize the vision model with model-specific hyperparameters
            vision_llm = self._create_vision_llm(model_name, temperature)
            
            extracted_text = ""
            
            # Process each image/page
            for i, image_path in enumerate(image_paths):
                prompt_text = f"""Please carefully examine this image from page {i+1} and extract ALL visible text.

Pay special attention to:
1. Handwritten text (cursive, print, or mixed)
2. Typed or printed text
3. Numbers, dates, and special characters
4. Headers, footers, and margins
5. Any text that might be faint, stylized, or difficult to read

Please transcribe the text exactly as it appears, maintaining the original formatting and line breaks where possible. If you cannot read certain words clearly, indicate this with [unclear] but still provide your best interpretation.

Be thorough and systematic in your transcription."""

                try:
                    # Convert image to base64 and create multimodal message
                    image_b64 = self._image_to_base64(image_path)
                    
                    # Determine image format from file extension
                    ext = Path(image_path).suffix.lower()
                    mime_type = {
                        '.png': 'image/png',
                        '.jpg': 'image/jpeg',
                        '.jpeg': 'image/jpeg',
                        '.gif': 'image/gif',
                        '.webp': 'image/webp'
                    }.get(ext, 'image/png')
                    
                    # Create HumanMessage with multimodal content (image + text)
                    message = HumanMessage(
                        content=[
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:{mime_type};base64,{image_b64}"}
                            },
                            {
                                "type": "text",
                                "text": prompt_text
                            }
                        ]
                    )
                    
                    # Invoke the vision model with the multimodal message
                    response_start = datetime.now()
                    response = vision_llm.invoke([message])
                    response_time_seconds = (datetime.now() - response_start).total_seconds()
                    response_text = response.content if hasattr(response, 'content') else str(response)

                    # --- Debugging: console preview of raw response ---
                    stripped = response_text.strip() if response_text else ""
                    word_count = len(stripped.split()) if stripped else 0
                    logger.info(
                        "[%s] page %d — response time: %.2fs, %d chars, %d words",
                        model_name, i + 1, response_time_seconds, len(stripped), word_count
                    )
                    if len(stripped) < 80:
                        logger.warning(
                            "[%s] page %d — SUSPICIOUSLY SHORT response: %r",
                            model_name, i + 1, stripped
                        )
                    else:
                        logger.info(
                            "[%s] page %d — response preview: %s",
                            model_name, i + 1, stripped[:300]
                        )

                    # Log the complete vision LLM output
                    self._log_vision_llm_output(
                        model_name,
                        i + 1,
                        prompt_text,
                        response_text,
                        temperature,
                        response_time_seconds,
                    )
                    
                    if response_text and response_text.strip():
                        extracted_text += f"=== Page {i+1} ===\n{response_text.strip()}\n\n"
                        logger.info(f"Model {model_name} extracted {len(response_text.strip())} characters from page {i+1}")
                
                except Exception as e:
                    logger.error(f"Error processing page {i+1} with model {model_name}: {e}")
                    extracted_text += f"=== Page {i+1} ===\n[Error processing page: {str(e)}]\n\n"
            
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Model {model_name} completed extraction in {execution_time:.2f} seconds")
            
            return extracted_text.strip(), execution_time
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Error with model {model_name}: {e}")
            return f"[Error: {str(e)}]", execution_time
    
    def calculate_levenshtein_distance(self, text1: str, text2: str) -> float:
        """Calculate normalized Levenshtein distance."""
        if LEVENSHTEIN_AVAILABLE:
            distance = Levenshtein.distance(text1, text2)
            max_len = max(len(text1), len(text2))
            return distance / max_len if max_len > 0 else 0.0
        else:
            # Fallback using difflib
            matcher = difflib.SequenceMatcher(None, text1, text2)
            return 1.0 - matcher.ratio()
    
    def calculate_word_error_rate(self, reference: str, hypothesis: str) -> float:
        """Calculate Word Error Rate (WER)."""
        ref_words = reference.split()
        hyp_words = hypothesis.split()
        
        # Create distance matrix
        d = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]
        
        # Initialize first row and column
        for i in range(len(ref_words) + 1):
            d[i][0] = i
        for j in range(len(hyp_words) + 1):
            d[0][j] = j
        
        # Fill the matrix
        for i in range(1, len(ref_words) + 1):
            for j in range(1, len(hyp_words) + 1):
                if ref_words[i-1] == hyp_words[j-1]:
                    d[i][j] = d[i-1][j-1]
                else:
                    d[i][j] = min(
                        d[i-1][j] + 1,      # deletion
                        d[i][j-1] + 1,      # insertion
                        d[i-1][j-1] + 1     # substitution
                    )
        
        return d[len(ref_words)][len(hyp_words)] / len(ref_words) if len(ref_words) > 0 else 0.0
    
    def calculate_character_accuracy(self, reference: str, hypothesis: str) -> float:
        """Calculate character-level accuracy."""
        if not reference:
            return 1.0 if not hypothesis else 0.0
        
        matcher = difflib.SequenceMatcher(None, reference, hypothesis)
        return matcher.ratio()
    
    def calculate_semantic_similarity(self, reference: str, hypothesis: str) -> float:
        """Calculate semantic similarity using LLM-as-a-judge."""
        if not self.judge_llm:
            logger.warning("Judge LLM not available, using character accuracy as semantic similarity")
            return self.calculate_character_accuracy(reference, hypothesis)
        
        try:
            prompt = f"""You are an expert evaluator assessing the semantic similarity between two text passages. The first passage is the ground truth reference, and the second is a machine-generated transcription.

Please rate the semantic similarity on a scale from 0.0 to 1.0, where:
- 1.0 = Perfect semantic match (meaning is identical)
- 0.8-0.9 = Very high similarity (minor differences that don't change meaning)
- 0.6-0.7 = Good similarity (some differences but core meaning preserved)
- 0.4-0.5 = Moderate similarity (partial meaning preserved)
- 0.2-0.3 = Low similarity (little meaning preserved)
- 0.0-0.1 = No meaningful similarity

Consider:
- Overall meaning and intent
- Key information preservation
- Contextual understanding
- Important details retention

Reference Text:
{reference[:10000]}...

Generated Text:
{hypothesis[:10000]}...

Respond with only a single number between 0.0 and 1.0 representing the semantic similarity score."""

            judge_start = datetime.now()
            response = self.judge_llm.invoke(prompt)
            judge_response_time_seconds = (datetime.now() - judge_start).total_seconds()
            response_text = response.content if hasattr(response, 'content') else str(response)

            logger.info(
                "[judge:%s] response time: %.2fs",
                self.judge_model,
                judge_response_time_seconds,
            )
            
            # Log the complete judge LLM interaction
            self._log_judge_llm_output(
                reference,
                hypothesis,
                prompt,
                response_text,
                -1.0,
                judge_response_time_seconds,
            )  # -1.0 indicates score not yet extracted
            
            # Extract numeric score from response
            score_match = re.search(r'(\d+(?:\.\d+)?)', response_text.strip())
            if score_match:
                score = float(score_match.group(1))
                # Ensure score is in valid range
                final_score = max(0.0, min(1.0, score))
                
                # Log the judge LLM output
                self._log_judge_llm_output(
                    reference,
                    hypothesis,
                    prompt,
                    response_text,
                    final_score,
                    judge_response_time_seconds,
                )
                
                return final_score
            else:
                logger.warning("Could not extract numeric score from judge response, falling back to character accuracy")
                fallback_score = self.calculate_character_accuracy(reference, hypothesis)
                
                # Log the failed judge LLM output
                self._log_judge_llm_output(
                    reference,
                    hypothesis,
                    prompt,
                    response_text,
                    fallback_score,
                    judge_response_time_seconds,
                )
                
                return fallback_score
                
        except Exception as e:
            logger.error(f"Error in semantic similarity calculation: {e}")
            return self.calculate_character_accuracy(reference, hypothesis)
    
    def evaluate_model(self, model_name: str, image_paths: List[str], ground_truth: str, temperature: float = 0.0, judge_llm=None) -> EvaluationResult:
        """
        Evaluate a single vision model.
        
        Args:
            model_name: Name of the vision model
            image_paths: List of image file paths
            ground_truth: Reference text
            temperature: Temperature setting for the model (default: 0.0)
            
        Returns:
            EvaluationResult object
        """
        logger.info(f"Evaluating model: {model_name} with temperature: {temperature}")
        
        try:
            # Extract text using the vision model with specified temperature
            extracted_text, execution_time = self.extract_text_with_vision_model(model_name, image_paths, temperature)
            if extracted_text.startswith("[Error:"):
                return EvaluationResult(
                    model_name=model_name,
                    extracted_text=extracted_text,
                    levenshtein_distance=1.0,
                    word_error_rate=1.0,
                    character_accuracy=0.0,
                    semantic_similarity=0.0,
                    execution_time=execution_time,
                    error_message=extracted_text,
                    temperature=temperature
                )
            # Clean extracted text for comparison
            clean_extracted = self._clean_text_for_comparison(extracted_text)
            clean_ground_truth = self._clean_text_for_comparison(ground_truth)
            # Calculate metrics except semantic similarity
            levenshtein_dist = self.calculate_levenshtein_distance(clean_ground_truth, clean_extracted)
            wer = self.calculate_word_error_rate(clean_ground_truth, clean_extracted)
            char_accuracy = self.calculate_character_accuracy(clean_ground_truth, clean_extracted)
            # Semantic similarity will be filled in a second pass if judge_llm is provided
            semantic_sim = None if judge_llm is not None else self.calculate_semantic_similarity(clean_ground_truth, clean_extracted)
            result = EvaluationResult(
                model_name=model_name,
                extracted_text=extracted_text,
                levenshtein_distance=levenshtein_dist,
                word_error_rate=wer,
                character_accuracy=char_accuracy,
                semantic_similarity=semantic_sim,
                execution_time=execution_time,
                temperature=temperature
            )
            logger.info(f"Model {model_name} - Char Accuracy: {char_accuracy:.3f}, WER: {wer:.3f}")

            # --- Debugging: side-by-side snippet comparison on console ---
            gt_preview = clean_ground_truth[:5000].replace('\n', ' ')
            ex_preview = clean_extracted[:5000].replace('\n', ' ')
            logger.info(
                "[%s] Ground truth (first 5000 chars): %s",
                model_name, gt_preview
            )
            logger.info(
                "[%s] Extracted   (first 5000 chars): %s",
                model_name, ex_preview
            )
            gt_words = len(clean_ground_truth.split())
            ex_words = len(clean_extracted.split())
            logger.info(
                "[%s] Word count — ground truth: %d, extracted: %d (%.0f%%)",
                model_name, gt_words, ex_words,
                (ex_words / gt_words * 100) if gt_words else 0
            )

            return result
        except Exception as e:
            logger.error(f"Error evaluating model {model_name}: {e}")
            return EvaluationResult(
                model_name=model_name,
                extracted_text="",
                levenshtein_distance=1.0,
                word_error_rate=1.0,
                character_accuracy=0.0,
                semantic_similarity=0.0,
                execution_time=0.0,
                error_message=str(e),
                temperature=temperature
            )
    
    def _clean_text_for_comparison(self, text: str) -> str:
        """Clean text for fair comparison."""
        # Remove page markers
        text = re.sub(r'=== Page \d+ ===\n?', '', text)

        # Remove presentation markup that some models add (HTML/Markdown wrappers).
        # Keep human-readable content so OCR quality, not formatting style, drives scores.
        text = self._strip_markup_wrappers(text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Convert to lowercase for case-insensitive comparison
        text = text.lower()
        
        return text

    def _strip_markup_wrappers(self, text: str) -> str:
        """Strip common HTML/Markdown wrappers while preserving textual content."""
        if not text:
            return ""

        cleaned = text

        # Decode HTML entities before tag stripping so text content is preserved.
        cleaned = html.unescape(cleaned)

        # Remove script/style blocks and common HTML tags.
        cleaned = re.sub(r'(?is)<(script|style)\b[^>]*>.*?</\1>', ' ', cleaned)
        cleaned = re.sub(r'(?is)</?(html|head|body|main|section|article|div|span|p|br|hr|table|thead|tbody|tr|td|th|ul|ol|li|h[1-6])\b[^>]*>', ' ', cleaned)
        cleaned = re.sub(r'(?is)<[^>]+>', ' ', cleaned)

        # Drop markdown fences while keeping their inner text.
        cleaned = re.sub(r'```[\w+-]*\s*\n', '\n', cleaned)
        cleaned = cleaned.replace('```', ' ')

        # Keep visible link text and image alt text, discard URLs.
        cleaned = re.sub(r'!\[([^\]]*)\]\([^\)]*\)', r' \1 ', cleaned)
        cleaned = re.sub(r'\[([^\]]+)\]\([^\)]*\)', r' \1 ', cleaned)

        # Remove remaining markdown control characters/wrappers.
        cleaned = re.sub(r'`([^`]*)`', r' \1 ', cleaned)
        cleaned = re.sub(r'(?m)^\s{0,3}>+\s?', '', cleaned)
        cleaned = re.sub(r'(?m)^\s{0,3}(?:[-*+]\s+|\d+[.)]\s+)', '', cleaned)
        cleaned = cleaned.replace('**', ' ').replace('__', ' ').replace('~~', ' ')

        # Replace single emphasis markers with spaces to avoid token sticking.
        cleaned = re.sub(r'(?<!\w)\*(?!\w)|(?<!\w)_(?!\w)', ' ', cleaned)

        return cleaned
    
    def calculate_composite_score(self, result: EvaluationResult) -> float:
        """
        Calculate a composite score for ranking models.
        
        Args:
            result: EvaluationResult object
            
        Returns:
            Composite score (higher is better)
        """
        if result.error_message:
            return 0.0
        
        # Weighted combination of metrics (higher is better)
        char_weight = 0.3
        semantic_weight = 0.4
        wer_weight = 0.2
        levenshtein_weight = 0.1
        
        # Convert WER and Levenshtein (lower is better) to higher-is-better scores
        wer_score = max(0.0, 1.0 - result.word_error_rate)
        levenshtein_score = max(0.0, 1.0 - result.levenshtein_distance)
        
        composite = (
            result.character_accuracy * char_weight +
            result.semantic_similarity * semantic_weight +
            wer_score * wer_weight +
            levenshtein_score * levenshtein_weight
        )
        
        return composite
    
    def evaluate_all_models(
        self,
        pdf_path: str,
        ground_truth_path: str,
        output_dir: str = "evaluation_results",
        models: Optional[List[str]] = None,
        progress_callback=None,
        should_cancel=None,
    ) -> EvaluationSummary:
        """
        Evaluate all available vision models.
        
        Args:
            pdf_path: Path to the PDF file
            ground_truth_path: Path to the ground truth text file
            output_dir: Directory to save results
            
        Returns:
            EvaluationSummary object
        """
        logger.info(f"Starting evaluation pipeline for {pdf_path}")
        
        # Create output directory
        Path(output_dir).mkdir(exist_ok=True)
        
        # Load ground truth
        try:
            with open(ground_truth_path, 'r', encoding='utf-8') as f:
                ground_truth = f.read()
            logger.info(f"Loaded ground truth from {ground_truth_path} ({len(ground_truth)} characters)")
        except Exception as e:
            raise RuntimeError(f"Error loading ground truth file: {e}")
        
        # Convert PDF to images
        image_paths = self.pdf_to_images(pdf_path)
        if not image_paths:
            raise RuntimeError("Failed to convert PDF to images")
        
        try:
            # Discover vision models
            vision_models = models if models else self.discover_vision_models()
            if not vision_models:
                raise RuntimeError("No vision-capable models found")

            total_steps = len(vision_models) * 2
            step = 0

            def report(message: str):
                if progress_callback:
                    progress_callback(step, total_steps, message)

            def cancelled() -> bool:
                return bool(should_cancel and should_cancel())

            # Phase 1: Run all vision model extractions first (no judge model calls)
            partial_results = []
            for model_name in vision_models:
                if cancelled():
                    break
                report(f"Extracting text with {model_name}")
                result = self.evaluate_model(model_name, image_paths, ground_truth, judge_llm=True)
                partial_results.append(result)
                step += 1
                report(f"Finished extraction: {model_name}")

            if not partial_results:
                raise RuntimeError("Evaluation cancelled")

            # Phase 2: Run all judge model evaluations in a batch (minimize model swap)
            step = len(vision_models)
            for result in partial_results:
                report(f"Judging {result.model_name}")
                step += 1
                if result.error_message:
                    result.semantic_similarity = 0.0
                else:
                    clean_extracted = self._clean_text_for_comparison(result.extracted_text)
                    clean_ground_truth = self._clean_text_for_comparison(ground_truth)
                    result.semantic_similarity = self.calculate_semantic_similarity(clean_ground_truth, clean_extracted)

            step = total_steps
            report("Evaluation complete")

            # Calculate composite scores and rank models
            model_scores = [(result.model_name, self.calculate_composite_score(result)) for result in partial_results]
            model_scores.sort(key=lambda x: x[1], reverse=True)

            best_model = model_scores[0][0] if model_scores else "None"
            best_score = model_scores[0][1] if model_scores else 0.0

            # Create evaluation summary
            summary = EvaluationSummary(
                input_pdf=pdf_path,
                ground_truth_file=ground_truth_path,
                total_models_tested=len(partial_results),
                evaluation_date=datetime.now().isoformat(),
                results=partial_results,
                best_model=best_model,
                best_score=best_score,
                ranking=model_scores
            )

            # Save results
            self._save_results(summary, output_dir)

            return summary

        finally:
            # Clean up temporary image files
            for image_path in image_paths:
                try:
                    os.unlink(image_path)
                except:
                    pass
    
    def _save_results(self, summary: EvaluationSummary, output_dir: str):
        """Save evaluation results to files (JSON, text report, and CSV)."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed JSON results
        json_path = os.path.join(output_dir, f"evaluation_results_{timestamp}.json")
        
        # Convert results to serializable format
        results_data = {
            "input_pdf": summary.input_pdf,
            "ground_truth_file": summary.ground_truth_file,
            "total_models_tested": summary.total_models_tested,
            "evaluation_date": summary.evaluation_date,
            "best_model": summary.best_model,
            "best_score": summary.best_score,
            "ranking": summary.ranking,
            "detailed_results": []
        }
        
        for result in summary.results:
            results_data["detailed_results"].append({
                "model_name": result.model_name,
                "extracted_text": result.extracted_text[:1000] + "..." if len(result.extracted_text) > 1000 else result.extracted_text,
                "levenshtein_distance": result.levenshtein_distance,
                "word_error_rate": result.word_error_rate,
                "character_accuracy": result.character_accuracy,
                "semantic_similarity": result.semantic_similarity,
                "execution_time": result.execution_time,
                "composite_score": self.calculate_composite_score(result),
                "error_message": result.error_message
            })
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        # Save human-readable report
        report_path = os.path.join(output_dir, f"evaluation_report_{timestamp}.txt")
        self._generate_report(summary, report_path)
        
        # Save CSV summary
        csv_path = os.path.join(output_dir, f"evaluation_summary_{timestamp}.csv")
        self._generate_csv(summary, csv_path)
        
        logger.info(f"Results saved to {json_path}, {report_path}, and {csv_path}")
    
    def _generate_csv(self, summary: EvaluationSummary, csv_path: str):
        """Generate a CSV summary of evaluation results."""
        fieldnames = [
            'model_name',
            'composite_score',
            'character_accuracy',
            'semantic_similarity',
            'word_error_rate',
            'levenshtein_distance',
            'execution_time',
            'error'
        ]
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in summary.results:
                writer.writerow({
                    'model_name': result.model_name,
                    'composite_score': f"{self.calculate_composite_score(result):.4f}",
                    'character_accuracy': f"{result.character_accuracy:.4f}",
                    'semantic_similarity': f"{result.semantic_similarity:.4f}" if result.semantic_similarity is not None else "N/A",
                    'word_error_rate': f"{result.word_error_rate:.4f}",
                    'levenshtein_distance': f"{result.levenshtein_distance:.4f}",
                    'execution_time': f"{result.execution_time:.2f}",
                    'error': result.error_message or ""
                })
    
    def _generate_report(self, summary: EvaluationSummary, report_path: str):
        """Generate a human-readable evaluation report."""
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("HANDWRITTEN TEXT EXTRACTION EVALUATION REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Input PDF: {summary.input_pdf}\n")
            f.write(f"Ground Truth: {summary.ground_truth_file}\n")
            f.write(f"Models Tested: {summary.total_models_tested}\n")
            f.write(f"Evaluation Date: {summary.evaluation_date}\n")
            f.write(f"Best Model: {summary.best_model} (Score: {summary.best_score:.3f})\n\n")
            
            f.write("MODEL RANKING\n")
            f.write("-" * 40 + "\n")
            for i, (model, score) in enumerate(summary.ranking, 1):
                f.write(f"{i:2d}. {model:<25} Score: {score:.3f}\n")
            f.write("\n")
            
            f.write("DETAILED RESULTS\n")
            f.write("-" * 40 + "\n")
            
            for result in summary.results:
                f.write(f"\nModel: {result.model_name}\n")
                f.write(f"  Character Accuracy: {result.character_accuracy:.3f}\n")
                f.write(f"  Word Error Rate: {result.word_error_rate:.3f}\n")
                f.write(f"  Levenshtein Distance: {result.levenshtein_distance:.3f}\n")
                f.write(f"  Semantic Similarity: {result.semantic_similarity:.3f}\n")
                f.write(f"  Execution Time: {result.execution_time:.2f}s\n")
                f.write(f"  Composite Score: {self.calculate_composite_score(result):.3f}\n")
                
                if result.error_message:
                    f.write(f"  ERROR: {result.error_message}\n")

                f.write("\n  Complete Transcription:\n")
                f.write("  " + "-" * 60 + "\n")
                if result.extracted_text:
                    f.write(result.extracted_text)
                    if not result.extracted_text.endswith("\n"):
                        f.write("\n")
                else:
                    f.write("[No transcription output]\n")
                f.write("  " + "-" * 60 + "\n")


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(description="Evaluate vision models for handwritten text extraction")
    parser.add_argument("pdf_path", nargs="?", help="Path to the PDF file to evaluate")
    parser.add_argument("ground_truth_path", nargs="?", help="Path to the ground truth text file")
    parser.add_argument("--cloud", action="store_true", help="Automatically evaluate all free Ollama cloud vision models found online")
    parser.add_argument("--gui", action="store_true", help="Launch the model selection GUI")
    parser.add_argument("--output-dir", default="evaluation_results", help="Output directory for results")
    parser.add_argument("--ollama-url", default="http://localhost:11434", help="Ollama server URL")
    parser.add_argument("--judge-model", default=None, help="Model to use for LLM-as-a-judge (default: gpt-oss:120b, or openai/gpt-oss-120b:free with --provider openrouter)")
    parser.add_argument("--provider", choices=["ollama", "openrouter"], default="ollama", help="Model provider")
    parser.add_argument("--openrouter-api-key", default=None, help="OpenRouter API key (default: OPENROUTER_API_KEY env var)")
    parser.add_argument("--list-openrouter-free", action="store_true", help="List free OpenRouter vision models and exit")
    parser.add_argument("--models", nargs="+", help="Specific models to test (if not provided, all vision models will be tested)")
    parser.add_argument("--request-timeout", type=int, default=120, help="Per-request timeout in seconds")
    parser.add_argument("--num-ctx", type=int, default=131072, help="Default context window cap for model requests")
    
    args = parser.parse_args()

    if args.list_openrouter_free:
        from openrouter_models import fetch_free_vision_models
        print("\n".join(fetch_free_vision_models(api_key=args.openrouter_api_key)))
        return 0

    if args.judge_model is None:
        args.judge_model = "openai/gpt-oss-120b:free" if args.provider == "openrouter" else "gpt-oss:120b"

    if args.gui:
        from evaluation_gui import run_gui
        return run_gui(args)

    if not args.pdf_path or not args.ground_truth_path:
        parser.error("pdf_path and ground_truth_path are required unless --gui is used")

    # Validate input files
    if not os.path.exists(args.pdf_path):
        print(f"Error: PDF file '{args.pdf_path}' not found")
        return 1
    
    if not os.path.exists(args.ground_truth_path):
        print(f"Error: Ground truth file '{args.ground_truth_path}' not found")
        return 1
    
    try:
        # Initialize evaluator
        evaluator = VisionModelEvaluator(
            ollama_base_url=args.ollama_url,
            judge_model=args.judge_model,
            request_timeout=max(10, args.request_timeout),
            default_num_ctx=max(262144, args.num_ctx),
            provider=args.provider,
            openrouter_api_key=args.openrouter_api_key,
        )
        
        # Override model discovery if specific models requested
        if args.cloud and not args.models:
            if args.provider == "openrouter":
                from openrouter_models import fetch_free_vision_models
                args.models = fetch_free_vision_models(api_key=args.openrouter_api_key)
            else:
                from ollama_cloud_models import fetch_free_cloud_vision_models
                args.models = fetch_free_cloud_vision_models(args.ollama_url)
            print(f"Free cloud vision models found: {args.models}")

        if args.models:
            # Strip stray commas that occur when the user passes comma-separated
            # values with spaces (e.g. --models a, b, c) in the shell.
            cleaned_models = [m.strip().rstrip(",") for m in args.models if m.strip().rstrip(",")]
            evaluator.discover_vision_models = lambda: cleaned_models
        
        # Run evaluation
        summary = evaluator.evaluate_all_models(
            args.pdf_path,
            args.ground_truth_path,
            args.output_dir
        )
        
        # Print summary
        print("\n" + "=" * 60)
        print("EVALUATION COMPLETE")
        print("=" * 60)
        print(f"Models tested: {summary.total_models_tested}")
        print(f"Best model: {summary.best_model} (Score: {summary.best_score:.3f})")
        print("\nTop 3 models:")
        for i, (model, score) in enumerate(summary.ranking[:3], 1):
            print(f"  {i}. {model} - {score:.3f}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
