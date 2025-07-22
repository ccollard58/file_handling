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
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import statistics

# LangChain imports
from langchain_ollama import OllamaLLM

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
    
    def __init__(self, ollama_base_url: str = "http://localhost:11434", judge_model: str = "llama3.1:latest"):
        """
        Initialize the evaluator.
        
        Args:
            ollama_base_url: URL of the Ollama server
            judge_model: Model to use for LLM-as-a-judge evaluations
        """
        self.ollama_base_url = ollama_base_url
        self.judge_model = judge_model
        self.judge_llm = None
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
            self.judge_llm = OllamaLLM(model=self.judge_model, temperature=0.1)
            logger.info(f"Judge LLM initialized with model: {self.judge_model}")
        except Exception as e:
            logger.error(f"Failed to initialize judge LLM: {e}")
            self.judge_llm = None
    
    def _log_vision_llm_output(self, model_name: str, page_number: int, prompt: str, response: str, temperature: float = 0.0):
        """Log the complete vision LLM output to a detailed log file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"vision_llm_outputs_{timestamp}.log"
        
        try:
            with open(log_filename, 'a', encoding='utf-8') as f:
                f.write("=" * 120 + "\n")
                f.write(f"TIMESTAMP: {datetime.now().isoformat()}\n")
                f.write(f"MODEL: {model_name}\n")
                f.write(f"TEMPERATURE: {temperature}\n")
                f.write(f"PAGE: {page_number}\n")
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
    
    def _log_judge_llm_output(self, reference: str, hypothesis: str, prompt: str, response: str, final_score: float):
        """Log the complete judge LLM output to a detailed log file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"judge_llm_outputs_{timestamp}.log"
        
        try:
            with open(log_filename, 'a', encoding='utf-8') as f:
                f.write("=" * 120 + "\n")
                f.write(f"TIMESTAMP: {datetime.now().isoformat()}\n")
                f.write(f"JUDGE MODEL: {self.judge_model}\n")
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
            response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=10)
            response.raise_for_status()
            models = response.json().get("models", [])
            all_models = [model["name"] for model in models]
            
            # Filter for vision-capable models
            vision_keywords = [
                'llava', 'vision', 'multimodal', 'minicpm', 'moondream', 
                'bakllava', 'cogvlm', 'vl', 'qwen2.5-vl', 'pixtral',
                'qwq', 'phi3.5-vision', 'llama3.2-vision', 'mistral-small',
                'gemma3', 'llama4'
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
    
    def extract_text_with_vision_model(self, model_name: str, image_paths: List[str], temperature: float = 0.0) -> Tuple[str, float]:
        """
        Extract text from images using a specific vision model.
        
        Args:
            model_name: Name of the vision model
            image_paths: List of image file paths
            temperature: Temperature setting for the model (default: 0.0)
            
        Returns:
            Tuple of (extracted text, execution time in seconds)
        """
        start_time = datetime.now()
        
        try:
            # Initialize the vision model with specified temperature
            vision_llm = OllamaLLM(model=model_name, temperature=temperature)
            
            extracted_text = ""
            
            # Process each image/page
            for i, image_path in enumerate(image_paths):
                prompt = f"""Please carefully examine this image from page {i+1} and extract ALL visible text.

Pay special attention to:
1. Handwritten text (cursive, print, or mixed)
2. Typed or printed text
3. Numbers, dates, and special characters
4. Headers, footers, and margins
5. Any text that might be faint, stylized, or difficult to read

Please transcribe the text exactly as it appears, maintaining the original formatting and line breaks where possible. If you cannot read certain words clearly, indicate this with [unclear] but still provide your best interpretation.

Be thorough and systematic in your transcription."""

                try:
                    response = vision_llm.invoke(prompt, images=[image_path])
                    
                    # Log the complete vision LLM output
                    self._log_vision_llm_output(model_name, i+1, prompt, response, temperature)
                    
                    if response and response.strip():
                        extracted_text += f"=== Page {i+1} ===\n{response.strip()}\n\n"
                        logger.debug(f"Model {model_name} extracted {len(response.strip())} characters from page {i+1}")
                
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

            response = self.judge_llm.invoke(prompt)
            
            # Log the complete judge LLM interaction
            
            # Extract numeric score from response
            score_match = re.search(r'(\d+(?:\.\d+)?)', response.strip())
            if score_match:
                score = float(score_match.group(1))
                # Ensure score is in valid range
                final_score = max(0.0, min(1.0, score))
                
                # Log the judge LLM output
                self._log_judge_llm_output(reference, hypothesis, prompt, response, final_score)
                
                return final_score
            else:
                logger.warning("Could not extract numeric score from judge response, falling back to character accuracy")
                fallback_score = self.calculate_character_accuracy(reference, hypothesis)
                
                # Log the failed judge LLM output
                self._log_judge_llm_output(reference, hypothesis, prompt, response, fallback_score)
                
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
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Convert to lowercase for case-insensitive comparison
        text = text.lower()
        
        return text
    
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
    
    def evaluate_all_models(self, pdf_path: str, ground_truth_path: str, output_dir: str = "evaluation_results") -> EvaluationSummary:
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
            vision_models = self.discover_vision_models()
            if not vision_models:
                raise RuntimeError("No vision-capable models found")

            # Phase 1: Run all vision model extractions first (no judge model calls)
            partial_results = []
            for model_name in vision_models:
                result = self.evaluate_model(model_name, image_paths, ground_truth, judge_llm=True)
                partial_results.append(result)

            # Phase 2: Run all judge model evaluations in a batch (minimize model swap)
            for result in partial_results:
                if result.error_message:
                    result.semantic_similarity = 0.0
                else:
                    clean_extracted = self._clean_text_for_comparison(result.extracted_text)
                    clean_ground_truth = self._clean_text_for_comparison(ground_truth)
                    result.semantic_similarity = self.calculate_semantic_similarity(clean_ground_truth, clean_extracted)

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
        """Save evaluation results to files."""
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
        
        logger.info(f"Results saved to {json_path} and {report_path}")
    
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
                
                # Show a sample of extracted text
                if result.extracted_text and not result.error_message:
                    sample = result.extracted_text[:200].replace('\n', ' ')
                    f.write(f"  Sample Text: {sample}...\n")


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(description="Evaluate vision models for handwritten text extraction")
    parser.add_argument("pdf_path", help="Path to the PDF file to evaluate")
    parser.add_argument("ground_truth_path", help="Path to the ground truth text file")
    parser.add_argument("--output-dir", default="evaluation_results", help="Output directory for results")
    parser.add_argument("--ollama-url", default="http://localhost:11434", help="Ollama server URL")
    parser.add_argument("--judge-model", default="phi4-mini:3.8b", help="Model to use for LLM-as-a-judge")
    parser.add_argument("--models", nargs="+", help="Specific models to test (if not provided, all vision models will be tested)")
    
    args = parser.parse_args()
    
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
            judge_model=args.judge_model
        )
        
        # Override model discovery if specific models requested
        if args.models:
            evaluator.discover_vision_models = lambda: args.models
        
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
