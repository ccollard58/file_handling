#!/usr/bin/env python3
"""
Temperature Optimization for Vision Models

This script runs the vision model evaluation system across different temperature settings
to determine the optimal temperature configuration for each model for handwritten text extraction.

Usage:
    python optimize_temperature.py [options]
"""

import os
import sys
import json
import logging
import argparse
import itertools
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple
import statistics

# Optional plotting dependencies
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

# Import our evaluation system
from evaluate_handwriting_vision_models import VisionModelEvaluator, EvaluationResult, EvaluationSummary

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('temperature_optimization.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class TemperatureOptimizer:
    """
    Optimizes temperature settings for vision models by running evaluations
    across multiple temperature values and analyzing the results.
    """
    
    def __init__(self, 
                 pdf_path: str, 
                 ground_truth_path: str,
                 ollama_base_url: str = "http://localhost:11434",
                 judge_model: str = "llama3.1:latest"):
        """
        Initialize the temperature optimizer.
        
        Args:
            pdf_path: Path to the PDF file for evaluation
            ground_truth_path: Path to the ground truth text file
            ollama_base_url: URL of the Ollama server
            judge_model: Model to use for LLM-as-a-judge evaluations
        """
        self.pdf_path = pdf_path
        self.ground_truth_path = ground_truth_path
        self.ollama_base_url = ollama_base_url
        self.judge_model = judge_model
        
        # Temperature ranges to test
        self.temperature_values = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]
        
        # Results storage
        self.optimization_results = {}
        self.best_configurations = {}
        
        # Load ground truth once
        with open(ground_truth_path, 'r', encoding='utf-8') as f:
            self.ground_truth = f.read()
    
    def get_available_models(self) -> List[str]:
        """Get list of available vision models."""
        evaluator = VisionModelEvaluator(self.ollama_base_url, self.judge_model)
        return evaluator.discover_vision_models()
    
    def evaluate_model_at_temperature(self, model_name: str, temperature: float) -> EvaluationResult:
        """
        Evaluate a single model at a specific temperature.
        
        Args:
            model_name: Name of the vision model
            temperature: Temperature setting to test
            
        Returns:
            EvaluationResult object
        """
        step_timings = {}
        total_start_time = time.time()
        
        logger.info(f"Evaluating {model_name} at temperature {temperature}")
        
        # Step 1: Initialize evaluator
        step_start = time.time()
        evaluator = VisionModelEvaluator(self.ollama_base_url, self.judge_model)
        step_timings['evaluator_init'] = time.time() - step_start
        
        # Step 2: Convert PDF to images
        step_start = time.time()
        image_paths = evaluator.pdf_to_images(self.pdf_path)
        step_timings['pdf_to_images'] = time.time() - step_start
        logger.info(f"PDF conversion took {step_timings['pdf_to_images']:.2f} seconds")
        
        try:
            # Step 3: Main evaluation
            step_start = time.time()
            result = evaluator.evaluate_model(model_name, image_paths, self.ground_truth, temperature)
            step_timings['model_evaluation'] = time.time() - step_start
            logger.info(f"Model evaluation took {step_timings['model_evaluation']:.2f} seconds")
            
            # Add temperature and timing info to the result
            result.temperature = temperature
            if hasattr(result, 'step_timings'):
                result.step_timings.update(step_timings)
            else:
                result.step_timings = step_timings
            
            step_timings['total_evaluation'] = time.time() - total_start_time
            result.step_timings['total_evaluation'] = step_timings['total_evaluation']
            
            logger.info(f"Total evaluation time: {step_timings['total_evaluation']:.2f} seconds")
            logger.info(f"Timing breakdown - Init: {step_timings['evaluator_init']:.2f}s, "
                       f"PDF: {step_timings['pdf_to_images']:.2f}s, "
                       f"Eval: {step_timings['model_evaluation']:.2f}s")
            
            return result
            
        finally:
            # Step 4: Cleanup
            step_start = time.time()
            for image_path in image_paths:
                try:
                    os.unlink(image_path)
                except:
                    pass
            step_timings['cleanup'] = time.time() - step_start
    
    def optimize_model(self, model_name: str, max_runs_per_temp: int = 3) -> Dict[str, Any]:
        """
        Optimize temperature for a single model by testing multiple values.
        
        Args:
            model_name: Name of the model to optimize
            max_runs_per_temp: Number of runs per temperature for averaging
            
        Returns:
            Dictionary with optimization results for this model
        """
        model_optimization_start = time.time()
        logger.info(f"Starting temperature optimization for {model_name}")
        
        model_results = {
            'model_name': model_name,
            'temperature_results': {},
            'best_temperature': None,
            'best_score': 0.0,
            'best_metrics': None,
            'optimization_timings': {}
        }
        
        temp_timings = {}
        
        for temperature in self.temperature_values:
            temp_start_time = time.time()
            logger.info(f"Testing {model_name} at temperature {temperature}")
            
            temp_results = []
            
            # Run multiple evaluations for this temperature to get average performance
            for run in range(max_runs_per_temp):
                run_start_time = time.time()
                logger.info(f"  Run {run + 1}/{max_runs_per_temp}")
                
                try:
                    result = self.evaluate_model_at_temperature(model_name, temperature)
                    temp_results.append(result)
                    
                    run_time = time.time() - run_start_time
                    
                    # Log the extracted text for analysis
                    logger.info(f"  Run {run + 1} - Char Acc: {result.character_accuracy:.3f}, "
                              f"WER: {result.word_error_rate:.3f}, "
                              f"Semantic: {result.semantic_similarity:.3f}, "
                              f"Run Time: {run_time:.2f}s")
                    
                    # Log the actual extracted text (truncated for readability)
                    extracted_preview = result.extracted_text[:200].replace('\n', ' ').strip()
                    if len(result.extracted_text) > 200:
                        extracted_preview += "..."
                    logger.info(f"  Run {run + 1} - Extracted text preview: '{extracted_preview}'")
                    
                    # Log full extracted text to a detailed log file
                    self._log_detailed_extraction(model_name, temperature, run + 1, result)
                    
                except Exception as e:
                    logger.error(f"Error in run {run + 1} for {model_name} at T={temperature}: {e}")
                    continue
            
            temp_total_time = time.time() - temp_start_time
            temp_timings[temperature] = temp_total_time
            
            if temp_results:
                # Calculate average metrics for this temperature
                avg_metrics = self._calculate_average_metrics(temp_results)
                composite_score = self._calculate_composite_score(avg_metrics)
                
                # Calculate timing statistics for this temperature
                timing_stats = self._calculate_timing_statistics(temp_results)
                
                model_results['temperature_results'][temperature] = {
                    'avg_metrics': avg_metrics,
                    'composite_score': composite_score,
                    'individual_results': temp_results,
                    'std_dev': self._calculate_std_dev(temp_results),
                    'timing_stats': timing_stats,
                    'temperature_total_time': temp_total_time
                }
                
                logger.info(f"Temperature {temperature} - Average Composite Score: {composite_score:.3f}, "
                           f"Total Time: {temp_total_time:.2f}s, "
                           f"Avg per Run: {temp_total_time/len(temp_results):.2f}s")
                
                # Track best temperature
                if composite_score > model_results['best_score']:
                    model_results['best_score'] = composite_score
                    model_results['best_temperature'] = temperature
                    model_results['best_metrics'] = avg_metrics
        
        # Record total optimization time
        total_optimization_time = time.time() - model_optimization_start
        model_results['optimization_timings'] = {
            'total_optimization_time': total_optimization_time,
            'temperature_timings': temp_timings,
            'average_time_per_temperature': total_optimization_time / len(self.temperature_values) if self.temperature_values else 0
        }
        
        logger.info(f"Best temperature for {model_name}: {model_results['best_temperature']} "
                   f"(score: {model_results['best_score']:.3f})")
        logger.info(f"Total optimization time for {model_name}: {total_optimization_time:.2f} seconds")
        
        return model_results
    
    def _calculate_average_metrics(self, results: List[EvaluationResult]) -> Dict[str, float]:
        """Calculate average metrics from multiple evaluation results."""
        if not results:
            return {}
        
        return {
            'character_accuracy': statistics.mean([r.character_accuracy for r in results]),
            'word_error_rate': statistics.mean([r.word_error_rate for r in results]),
            'levenshtein_distance': statistics.mean([r.levenshtein_distance for r in results]),
            'semantic_similarity': statistics.mean([r.semantic_similarity for r in results]),
            'execution_time': statistics.mean([r.execution_time for r in results])
        }
    
    def _calculate_std_dev(self, results: List[EvaluationResult]) -> Dict[str, float]:
        """Calculate standard deviation for metrics."""
        if len(results) < 2:
            return {}
        
        return {
            'character_accuracy': statistics.stdev([r.character_accuracy for r in results]),
            'word_error_rate': statistics.stdev([r.word_error_rate for r in results]),
            'levenshtein_distance': statistics.stdev([r.levenshtein_distance for r in results]),
            'semantic_similarity': statistics.stdev([r.semantic_similarity for r in results]),
            'execution_time': statistics.stdev([r.execution_time for r in results])
        }
    
    def _calculate_timing_statistics(self, results: List[EvaluationResult]) -> Dict[str, float]:
        """Calculate timing statistics from multiple evaluation results."""
        if not results:
            return {}
        
        timing_stats = {
            'total_runs': len(results),
            'avg_execution_time': statistics.mean([r.execution_time for r in results]),
            'min_execution_time': min([r.execution_time for r in results]),
            'max_execution_time': max([r.execution_time for r in results])
        }
        
        if len(results) > 1:
            timing_stats['std_execution_time'] = statistics.stdev([r.execution_time for r in results])
        
        # Add step timing statistics if available
        step_timings_available = all(hasattr(r, 'step_timings') and r.step_timings for r in results)
        if step_timings_available:
            # Get all step names
            all_steps = set()
            for result in results:
                all_steps.update(result.step_timings.keys())
            
            # Calculate statistics for each step
            step_stats = {}
            for step in all_steps:
                step_times = []
                for result in results:
                    if step in result.step_timings:
                        step_times.append(result.step_timings[step])
                
                if step_times:
                    step_stats[f'{step}_avg'] = statistics.mean(step_times)
                    step_stats[f'{step}_min'] = min(step_times)
                    step_stats[f'{step}_max'] = max(step_times)
                    if len(step_times) > 1:
                        step_stats[f'{step}_std'] = statistics.stdev(step_times)
            
            timing_stats['step_statistics'] = step_stats
        
        return timing_stats
    
    def _calculate_composite_score(self, metrics: Dict[str, float]) -> float:
        """Calculate composite score from average metrics."""
        if not metrics:
            return 0.0
        
        # Use same weights as main evaluation system
        char_weight = 0.3
        semantic_weight = 0.4
        wer_weight = 0.2
        levenshtein_weight = 0.1
        
        # Convert WER and Levenshtein (lower is better) to higher-is-better scores
        wer_score = max(0.0, 1.0 - min(1.0, metrics.get('word_error_rate', 1.0)))
        levenshtein_score = max(0.0, 1.0 - metrics.get('levenshtein_distance', 1.0))
        
        composite = (
            metrics.get('character_accuracy', 0.0) * char_weight +
            metrics.get('semantic_similarity', 0.0) * semantic_weight +
            wer_score * wer_weight +
            levenshtein_score * levenshtein_weight
        )
        
        return composite
    
    def optimize_all_models(self, 
                           models: List[str] = None, 
                           max_runs_per_temp: int = 3,
                           output_dir: str = "temperature_optimization_results") -> Dict[str, Any]:
        """
        Optimize temperature for all available models.
        
        Args:
            models: List of specific models to test (if None, tests all available)
            max_runs_per_temp: Number of runs per temperature for averaging
            output_dir: Directory to save results
            
        Returns:
            Complete optimization results
        """
        logger.info("Starting temperature optimization for all models")
        
        # Create output directory
        Path(output_dir).mkdir(exist_ok=True)
        
        # Get models to test
        if models is None:
            models = self.get_available_models()
        
        if not models:
            raise RuntimeError("No vision models available for testing")
        
        logger.info(f"Testing {len(models)} models: {models}")
        
        # Optimize each model
        all_results = {
            'optimization_date': datetime.now().isoformat(),
            'pdf_file': self.pdf_path,
            'ground_truth_file': self.ground_truth_path,
            'temperature_values_tested': self.temperature_values,
            'runs_per_temperature': max_runs_per_temp,
            'models_tested': len(models),
            'model_results': {},
            'best_overall_model': None,
            'best_overall_score': 0.0,
            'summary_statistics': {}
        }
        
        for model_name in models:
            try:
                model_results = self.optimize_model(model_name, max_runs_per_temp)
                all_results['model_results'][model_name] = model_results
                
                # Track best overall model
                if model_results['best_score'] > all_results['best_overall_score']:
                    all_results['best_overall_score'] = model_results['best_score']
                    all_results['best_overall_model'] = {
                        'model_name': model_name,
                        'best_temperature': model_results['best_temperature'],
                        'best_score': model_results['best_score']
                    }
                
            except Exception as e:
                logger.error(f"Error optimizing {model_name}: {e}")
                all_results['model_results'][model_name] = {
                    'error': str(e),
                    'model_name': model_name
                }
        
        # Calculate summary statistics
        all_results['summary_statistics'] = self._calculate_summary_statistics(all_results)
        
        # Save results
        self._save_optimization_results(all_results, output_dir)
        
        # Generate visualizations
        self._create_visualizations(all_results, output_dir)
        
        return all_results
    
    def _calculate_summary_statistics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate summary statistics across all models."""
        successful_models = []
        best_temps = []
        best_scores = []
        
        for model_name, model_results in results['model_results'].items():
            if 'error' not in model_results and model_results.get('best_temperature') is not None:
                successful_models.append(model_name)
                best_temps.append(model_results['best_temperature'])
                best_scores.append(model_results['best_score'])
        
        if not successful_models:
            return {'error': 'No successful model optimizations'}
        
        # Temperature distribution
        temp_distribution = {}
        for temp in best_temps:
            temp_distribution[temp] = temp_distribution.get(temp, 0) + 1
        
        return {
            'successful_models': len(successful_models),
            'failed_models': len(results['model_results']) - len(successful_models),
            'average_best_score': statistics.mean(best_scores),
            'score_std_dev': statistics.stdev(best_scores) if len(best_scores) > 1 else 0.0,
            'best_temperature_distribution': temp_distribution,
            'most_common_best_temperature': max(temp_distribution.items(), key=lambda x: x[1])[0] if temp_distribution else None,
            'temperature_range': [min(best_temps), max(best_temps)] if best_temps else None
        }
    
    def _save_optimization_results(self, results: Dict[str, Any], output_dir: str):
        """Save optimization results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed JSON results
        json_path = os.path.join(output_dir, f"temperature_optimization_{timestamp}.json")
        
        # Create a serializable version of results
        serializable_results = self._make_serializable(results)
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        # Save human-readable report
        report_path = os.path.join(output_dir, f"temperature_optimization_report_{timestamp}.txt")
        self._generate_optimization_report(results, report_path)
        
        # Save CSV summary for easy analysis
        csv_path = os.path.join(output_dir, f"temperature_optimization_summary_{timestamp}.csv")
        self._save_csv_summary(results, csv_path)
        
        logger.info(f"Optimization results saved to {output_dir}/")
    
    def _make_serializable(self, obj):
        """Make object JSON serializable by converting complex types."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif hasattr(obj, '__dict__'):
            # Convert objects with attributes to dictionaries
            return self._make_serializable(obj.__dict__)
        else:
            return obj
    
    def _generate_optimization_report(self, results: Dict[str, Any], report_path: str):
        """Generate human-readable optimization report."""
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("TEMPERATURE OPTIMIZATION REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Optimization Date: {results['optimization_date']}\n")
            f.write(f"PDF File: {results['pdf_file']}\n")
            f.write(f"Ground Truth: {results['ground_truth_file']}\n")
            f.write(f"Temperature Values Tested: {results['temperature_values_tested']}\n")
            f.write(f"Runs per Temperature: {results['runs_per_temperature']}\n")
            f.write(f"Models Tested: {results['models_tested']}\n\n")
            
            # Best overall result
            if results['best_overall_model']:
                best = results['best_overall_model']
                f.write("BEST OVERALL CONFIGURATION\n")
                f.write("-" * 40 + "\n")
                f.write(f"Model: {best['model_name']}\n")
                f.write(f"Temperature: {best['best_temperature']}\n")
                f.write(f"Score: {best['best_score']:.3f}\n\n")
            
            # Summary statistics
            if 'summary_statistics' in results:
                stats = results['summary_statistics']
                f.write("SUMMARY STATISTICS\n")
                f.write("-" * 40 + "\n")
                f.write(f"Successful Models: {stats.get('successful_models', 0)}\n")
                f.write(f"Failed Models: {stats.get('failed_models', 0)}\n")
                f.write(f"Average Best Score: {stats.get('average_best_score', 0):.3f}\n")
                f.write(f"Score Std Dev: {stats.get('score_std_dev', 0):.3f}\n")
                f.write(f"Most Common Best Temperature: {stats.get('most_common_best_temperature', 'N/A')}\n")
                
                if stats.get('best_temperature_distribution'):
                    f.write("\nTemperature Distribution:\n")
                    for temp, count in sorted(stats['best_temperature_distribution'].items()):
                        f.write(f"  {temp}: {count} models\n")
                f.write("\n")
            
            # Individual model results
            f.write("INDIVIDUAL MODEL RESULTS\n")
            f.write("-" * 40 + "\n")
            
            for model_name, model_results in results['model_results'].items():
                f.write(f"\nModel: {model_name}\n")
                
                if 'error' in model_results:
                    f.write(f"  ERROR: {model_results['error']}\n")
                    continue
                
                f.write(f"  Best Temperature: {model_results.get('best_temperature', 'N/A')}\n")
                f.write(f"  Best Score: {model_results.get('best_score', 0):.3f}\n")
                
                # Add timing information
                if model_results.get('optimization_timings'):
                    timings = model_results['optimization_timings']
                    f.write(f"  Total Optimization Time: {timings.get('total_optimization_time', 0):.2f}s\n")
                    f.write(f"  Average Time per Temperature: {timings.get('average_time_per_temperature', 0):.2f}s\n")
                
                if model_results.get('best_metrics'):
                    metrics = model_results['best_metrics']
                    f.write(f"  Character Accuracy: {metrics.get('character_accuracy', 0):.3f}\n")
                    f.write(f"  Word Error Rate: {metrics.get('word_error_rate', 0):.3f}\n")
                    f.write(f"  Semantic Similarity: {metrics.get('semantic_similarity', 0):.3f}\n")
                    f.write(f"  Execution Time: {metrics.get('execution_time', 0):.2f}s\n")
                
                # Show performance across temperatures
                if model_results.get('temperature_results'):
                    f.write("  Temperature Performance:\n")
                    for temp, temp_data in sorted(model_results['temperature_results'].items()):
                        score = temp_data.get('composite_score', 0)
                        temp_time = temp_data.get('temperature_total_time', 0)
                        f.write(f"    T={temp}: Score={score:.3f}, Time={temp_time:.2f}s\n")
                        
                        # Add detailed timing stats if available
                        if temp_data.get('timing_stats'):
                            timing_stats = temp_data['timing_stats']
                            avg_time = timing_stats.get('avg_execution_time', 0)
                            f.write(f"      Avg per run: {avg_time:.2f}s")
                            if 'step_statistics' in timing_stats:
                                step_stats = timing_stats['step_statistics']
                                if 'model_evaluation_avg' in step_stats:
                                    f.write(f", Model eval: {step_stats['model_evaluation_avg']:.2f}s")
                                if 'pdf_to_images_avg' in step_stats:
                                    f.write(f", PDF conv: {step_stats['pdf_to_images_avg']:.2f}s")
                            f.write("\n")
    
    def _save_csv_summary(self, results: Dict[str, Any], csv_path: str):
        """Save results summary as CSV for easy analysis."""
        rows = []
        
        for model_name, model_results in results['model_results'].items():
            if 'error' in model_results:
                continue
                
            base_row = {
                'model_name': model_name,
                'best_temperature': model_results.get('best_temperature'),
                'best_score': model_results.get('best_score', 0)
            }
            
            # Add timing data
            if model_results.get('optimization_timings'):
                timings = model_results['optimization_timings']
                base_row.update({
                    'total_optimization_time': timings.get('total_optimization_time', 0),
                    'avg_time_per_temperature': timings.get('average_time_per_temperature', 0)
                })
            
            # Add best metrics
            if model_results.get('best_metrics'):
                metrics = model_results['best_metrics']
                base_row.update({
                    'best_char_accuracy': metrics.get('character_accuracy', 0),
                    'best_word_error_rate': metrics.get('word_error_rate', 0),
                    'best_semantic_similarity': metrics.get('semantic_similarity', 0),
                    'best_execution_time': metrics.get('execution_time', 0)
                })
            
            # Add performance at each temperature
            if model_results.get('temperature_results'):
                for temp, temp_data in model_results['temperature_results'].items():
                    row = base_row.copy()
                    row.update({
                        'temperature': temp,
                        'composite_score': temp_data.get('composite_score', 0),
                        'temperature_total_time': temp_data.get('temperature_total_time', 0)
                    })
                    
                    if temp_data.get('avg_metrics'):
                        avg_metrics = temp_data['avg_metrics']
                        row.update({
                            'char_accuracy': avg_metrics.get('character_accuracy', 0),
                            'word_error_rate': avg_metrics.get('word_error_rate', 0),
                            'semantic_similarity': avg_metrics.get('semantic_similarity', 0),
                            'execution_time': avg_metrics.get('execution_time', 0)
                        })
                    
                    # Add timing statistics
                    if temp_data.get('timing_stats'):
                        timing_stats = temp_data['timing_stats']
                        row.update({
                            'avg_run_time': timing_stats.get('avg_execution_time', 0),
                            'min_run_time': timing_stats.get('min_execution_time', 0),
                            'max_run_time': timing_stats.get('max_execution_time', 0),
                            'total_runs': timing_stats.get('total_runs', 0)
                        })
                        
                        # Add step timing averages if available
                        if timing_stats.get('step_statistics'):
                            step_stats = timing_stats['step_statistics']
                            row.update({
                                'avg_pdf_conversion_time': step_stats.get('pdf_to_images_avg', 0),
                                'avg_model_eval_time': step_stats.get('model_evaluation_avg', 0),
                                'avg_init_time': step_stats.get('evaluator_init_avg', 0),
                                'avg_cleanup_time': step_stats.get('cleanup_avg', 0)
                            })
                    
                    rows.append(row)
        
        if rows and PANDAS_AVAILABLE:
            df = pd.DataFrame(rows)
            df.to_csv(csv_path, index=False)
            logger.info(f"CSV summary saved to {csv_path}")
        elif rows:
            # Fallback CSV writing without pandas
            import csv
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                if rows:
                    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                    writer.writeheader()
                    writer.writerows(rows)
            logger.info(f"CSV summary saved to {csv_path}")
        else:
            logger.warning("No data to save to CSV")
    
    def _log_detailed_extraction(self, model_name: str, temperature: float, run_number: int, result):
        """Log detailed extraction results to a separate file for analysis."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"detailed_extractions_{timestamp}.log"
        
        try:
            with open(log_filename, 'a', encoding='utf-8') as f:
                f.write("=" * 100 + "\n")
                f.write(f"MODEL: {model_name}\n")
                f.write(f"TEMPERATURE: {temperature}\n")
                f.write(f"RUN: {run_number}\n")
                f.write(f"TIMESTAMP: {datetime.now().isoformat()}\n")
                f.write(f"EXECUTION TIME: {result.execution_time:.2f} seconds\n")
                
                # Add step timing details if available
                if hasattr(result, 'step_timings') and result.step_timings:
                    f.write("STEP TIMINGS:\n")
                    for step, timing in result.step_timings.items():
                        f.write(f"  {step}: {timing:.3f} seconds\n")
                
                f.write("-" * 50 + "\n")
                f.write("METRICS:\n")
                f.write(f"  Character Accuracy: {result.character_accuracy:.4f}\n")
                f.write(f"  Word Error Rate: {result.word_error_rate:.4f}\n")
                f.write(f"  Levenshtein Distance: {result.levenshtein_distance:.4f}\n")
                f.write(f"  Semantic Similarity: {result.semantic_similarity:.4f}\n")
                f.write("-" * 50 + "\n")
                f.write("EXTRACTED TEXT:\n")
                f.write(result.extracted_text)
                f.write("\n" + "-" * 50 + "\n")
                f.write("GROUND TRUTH COMPARISON:\n")
                f.write(f"Ground Truth Length: {len(self.ground_truth)} characters\n")
                f.write(f"Extracted Length: {len(result.extracted_text)} characters\n")
                f.write("First 300 chars of Ground Truth:\n")
                f.write(repr(self.ground_truth[:300]))
                f.write("\nFirst 300 chars of Extracted:\n")
                f.write(repr(result.extracted_text[:300]))
                f.write("\n" + "=" * 100 + "\n\n")
        except Exception as e:
            logger.error(f"Error writing detailed extraction log: {e}")
    
    def _create_visualizations(self, results: Dict[str, Any], output_dir: str):
        """Create visualization plots for the optimization results."""
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available. Skipping visualizations.")
            return
            
        try:
            # Set style if seaborn is available
            if SEABORN_AVAILABLE:
                plt.style.use('seaborn-v0_8')
                sns.set_palette("husl")
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 1. Temperature vs Score for each model
            self._plot_temperature_vs_score(results, output_dir, timestamp)
            
            # 2. Best temperature distribution
            self._plot_temperature_distribution(results, output_dir, timestamp)
            
            # 3. Model comparison at best temperatures
            self._plot_model_comparison(results, output_dir, timestamp)
            
            # 4. Performance metrics heatmap
            self._plot_metrics_heatmap(results, output_dir, timestamp)
            
            logger.info(f"Visualizations saved to {output_dir}/")
            
        except ImportError:
            logger.warning("Matplotlib/Seaborn not available, skipping visualizations")
        except Exception as e:
            logger.error(f"Error creating visualizations: {e}")
    
    def _plot_temperature_vs_score(self, results: Dict[str, Any], output_dir: str, timestamp: str):
        """Plot temperature vs composite score for each model."""
        plt.figure(figsize=(12, 8))
        
        for model_name, model_results in results['model_results'].items():
            if 'error' in model_results or not model_results.get('temperature_results'):
                continue
            
            temps = []
            scores = []
            
            for temp, temp_data in sorted(model_results['temperature_results'].items()):
                temps.append(temp)
                scores.append(temp_data.get('composite_score', 0))
            
            plt.plot(temps, scores, marker='o', label=model_name, linewidth=2, markersize=6)
        
        plt.xlabel('Temperature', fontsize=12)
        plt.ylabel('Composite Score', fontsize=12)
        plt.title('Temperature vs Performance for Vision Models', fontsize=14, fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, f'temperature_vs_score_{timestamp}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_temperature_distribution(self, results: Dict[str, Any], output_dir: str, timestamp: str):
        """Plot distribution of best temperatures."""
        if 'summary_statistics' not in results or 'best_temperature_distribution' not in results['summary_statistics']:
            return
        
        temp_dist = results['summary_statistics']['best_temperature_distribution']
        
        plt.figure(figsize=(10, 6))
        temps = list(temp_dist.keys())
        counts = list(temp_dist.values())
        
        bars = plt.bar(temps, counts, alpha=0.7, color='skyblue', edgecolor='navy')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom', fontweight='bold')
        
        plt.xlabel('Best Temperature', fontsize=12)
        plt.ylabel('Number of Models', fontsize=12)
        plt.title('Distribution of Best Temperatures Across Models', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, f'temperature_distribution_{timestamp}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_model_comparison(self, results: Dict[str, Any], output_dir: str, timestamp: str):
        """Plot comparison of models at their best temperatures."""
        models = []
        scores = []
        temperatures = []
        
        for model_name, model_results in results['model_results'].items():
            if 'error' not in model_results and model_results.get('best_score') is not None:
                models.append(model_name.replace(':', '\n'))  # Break long names
                scores.append(model_results['best_score'])
                temperatures.append(model_results.get('best_temperature', 0))
        
        if not models:
            return
        
        plt.figure(figsize=(12, 8))
        bars = plt.bar(range(len(models)), scores, alpha=0.7, color='lightcoral', edgecolor='darkred')
        
        # Add temperature labels on bars
        for i, (bar, temp) in enumerate(zip(bars, temperatures)):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'T={temp}', ha='center', va='bottom', fontweight='bold', fontsize=9)
            plt.text(bar.get_x() + bar.get_width()/2., height/2,
                    f'{height:.3f}', ha='center', va='center', fontweight='bold', color='white')
        
        plt.xlabel('Model', fontsize=12)
        plt.ylabel('Best Composite Score', fontsize=12)
        plt.title('Model Performance Comparison (at Best Temperature)', fontsize=14, fontweight='bold')
        plt.xticks(range(len(models)), models, rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, f'model_comparison_{timestamp}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_metrics_heatmap(self, results: Dict[str, Any], output_dir: str, timestamp: str):
        """Create heatmap of performance metrics across models and temperatures."""
        # Prepare data for heatmap
        models = []
        temps = set()
        
        for model_name, model_results in results['model_results'].items():
            if 'error' not in model_results and model_results.get('temperature_results'):
                models.append(model_name)
                temps.update(model_results['temperature_results'].keys())
        
        if not models:
            return
        
        temps = sorted(list(temps))
        
        # Create matrix for composite scores
        score_matrix = []
        for model_name in models:
            model_results = results['model_results'][model_name]
            row = []
            for temp in temps:
                if temp in model_results.get('temperature_results', {}):
                    score = model_results['temperature_results'][temp].get('composite_score', 0)
                    row.append(score)
                else:
                    row.append(0)
            score_matrix.append(row)
        
        plt.figure(figsize=(10, len(models) * 0.5 + 2))
        
        # Create heatmap
        ax = sns.heatmap(score_matrix, 
                        xticklabels=[f'T={t}' for t in temps],
                        yticklabels=[m.replace(':', '\n') for m in models],
                        annot=True, 
                        fmt='.3f', 
                        cmap='YlOrRd',
                        cbar_kws={'label': 'Composite Score'})
        
        plt.title('Performance Heatmap: Models vs Temperatures', fontsize=14, fontweight='bold')
        plt.xlabel('Temperature', fontsize=12)
        plt.ylabel('Model', fontsize=12)
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, f'metrics_heatmap_{timestamp}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """Main CLI interface for temperature optimization."""
    parser = argparse.ArgumentParser(description="Optimize temperature settings for vision models")
    parser.add_argument("pdf_path", help="Path to the PDF file to evaluate")
    parser.add_argument("ground_truth_path", help="Path to the ground truth text file")
    parser.add_argument("--output-dir", default="temperature_optimization_results", 
                       help="Output directory for results")
    parser.add_argument("--ollama-url", default="http://localhost:11434", 
                       help="Ollama server URL")
    parser.add_argument("--judge-model", default="llama3.1:latest", 
                       help="Model to use for LLM-as-a-judge")
    parser.add_argument("--models", nargs="+", 
                       help="Specific models to test (if not provided, all vision models will be tested)")
    parser.add_argument("--runs-per-temp", type=int, default=2,
                       help="Number of runs per temperature for averaging (default: 2)")
    parser.add_argument("--temperature-values", nargs="+", type=float,
                       help="Custom temperature values to test (default: 0.0 0.1 0.3 0.5 0.7 0.9)")
    
    args = parser.parse_args()
    
    # Validate input files
    if not os.path.exists(args.pdf_path):
        print(f"Error: PDF file '{args.pdf_path}' not found")
        return 1
    
    if not os.path.exists(args.ground_truth_path):
        print(f"Error: Ground truth file '{args.ground_truth_path}' not found")
        return 1
    
    try:
        # Initialize optimizer
        optimizer = TemperatureOptimizer(
            args.pdf_path,
            args.ground_truth_path,
            args.ollama_url,
            args.judge_model
        )
        
        # Override temperature values if provided
        if args.temperature_values:
            optimizer.temperature_values = sorted(args.temperature_values)
            logger.info(f"Using custom temperature values: {optimizer.temperature_values}")
        
        # Run optimization
        results = optimizer.optimize_all_models(
            models=args.models,
            max_runs_per_temp=args.runs_per_temp,
            output_dir=args.output_dir
        )
        
        # Print summary
        print("\n" + "=" * 80)
        print("TEMPERATURE OPTIMIZATION COMPLETE")
        print("=" * 80)
        
        if results['best_overall_model']:
            best = results['best_overall_model']
            print(f"Best Configuration:")
            print(f"  Model: {best['model_name']}")
            print(f"  Temperature: {best['best_temperature']}")
            print(f"  Score: {best['best_score']:.3f}")
        
        if 'summary_statistics' in results:
            stats = results['summary_statistics']
            print(f"\nSummary:")
            print(f"  Successful models: {stats.get('successful_models', 0)}")
            print(f"  Average best score: {stats.get('average_best_score', 0):.3f}")
            print(f"  Most common best temperature: {stats.get('most_common_best_temperature', 'N/A')}")
        
        print(f"\nDetailed results saved to: {args.output_dir}/")
        
        return 0
        
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
