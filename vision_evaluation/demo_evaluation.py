#!/usr/bin/env python3
"""
Demonstration script for the Vision Model Evaluation System

This script shows how to use the evaluate_handwriting_vision_models.py script
to benchmark vision-capable LLMs for handwritten text extraction.

Usage examples:
1. Basic evaluation: python demo_evaluation.py
2. Test specific models: python demo_evaluation.py --models llava:latest qwen2-vl:latest
3. Custom output directory: python demo_evaluation.py --output-dir custom_results
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

def check_ollama_connection():
    """Check if Ollama server is running."""
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        return response.status_code == 200
    except:
        return False

def list_available_pdfs():
    """List available PDF files in test_docs directory."""
    test_docs_dir = Path("test_docs")
    if not test_docs_dir.exists():
        return []
    
    return [f for f in test_docs_dir.glob("*.pdf")]

def list_available_ground_truth():
    """List available ground truth files in test_docs directory."""
    test_docs_dir = Path("test_docs")
    if not test_docs_dir.exists():
        return []
    
    return [f for f in test_docs_dir.glob("*.txt")]

def run_evaluation(pdf_path, ground_truth_path, args):
    """Run the evaluation script with specified parameters."""
    cmd = [
        sys.executable,
        "evaluate_handwriting_vision_models.py",
        str(pdf_path),
        str(ground_truth_path)
    ]
    
    if args.output_dir:
        cmd.extend(["--output-dir", args.output_dir])
    
    if args.ollama_url:
        cmd.extend(["--ollama-url", args.ollama_url])
    
    if args.judge_model:
        cmd.extend(["--judge-model", args.judge_model])
    
    if args.models:
        cmd.extend(["--models"] + args.models)
    
    print(f"Running command: {' '.join(cmd)}")
    print("-" * 60)
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Evaluation failed with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        print("Error: evaluate_handwriting_vision_models.py not found")
        return False

def main():
    parser = argparse.ArgumentParser(description="Demonstration of Vision Model Evaluation")
    parser.add_argument("--pdf", help="Specific PDF file to evaluate")
    parser.add_argument("--ground-truth", help="Specific ground truth file")
    parser.add_argument("--output-dir", help="Output directory for results")
    parser.add_argument("--ollama-url", default="http://localhost:11434", help="Ollama server URL")
    parser.add_argument("--judge-model", default="llama3.1:latest", help="Judge model")
    parser.add_argument("--models", nargs="+", help="Specific models to test")
    parser.add_argument("--list-files", action="store_true", help="List available files and exit")
    
    args = parser.parse_args()
    
    # List available files if requested
    if args.list_files:
        print("Available PDF files:")
        pdfs = list_available_pdfs()
        for pdf in pdfs:
            print(f"  {pdf}")
        
        print("\nAvailable ground truth files:")
        ground_truths = list_available_ground_truth()
        for gt in ground_truths:
            print(f"  {gt}")
        return 0
    
    # Check Ollama connection
    print("Checking Ollama server connection...")
    if not check_ollama_connection():
        print("⚠️  Warning: Cannot connect to Ollama server at http://localhost:11434")
        print("   Make sure Ollama is running: 'ollama serve'")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            return 1
    else:
        print("✅ Ollama server is running")
    
    # Determine files to use
    if args.pdf and args.ground_truth:
        pdf_path = Path(args.pdf)
        ground_truth_path = Path(args.ground_truth)
    else:
        # Use default files
        pdf_path = Path("test_docs/document_0001.pdf")
        ground_truth_path = Path("test_docs/sample_ground_truth.txt")
        
        print(f"Using default files:")
        print(f"  PDF: {pdf_path}")
        print(f"  Ground Truth: {ground_truth_path}")
    
    # Validate files exist
    if not pdf_path.exists():
        print(f"Error: PDF file '{pdf_path}' not found")
        
        # List available PDFs
        available_pdfs = list_available_pdfs()
        if available_pdfs:
            print("\nAvailable PDF files:")
            for pdf in available_pdfs:
                print(f"  {pdf}")
        return 1
    
    if not ground_truth_path.exists():
        print(f"Error: Ground truth file '{ground_truth_path}' not found")
        
        # List available ground truth files
        available_gts = list_available_ground_truth()
        if available_gts:
            print("\nAvailable ground truth files:")
            for gt in available_gts:
                print(f"  {gt}")
        return 1
    
    print(f"\n📄 Evaluating: {pdf_path}")
    print(f"📝 Ground Truth: {ground_truth_path}")
    
    if args.models:
        print(f"🔍 Testing specific models: {', '.join(args.models)}")
    else:
        print("🔍 Testing all available vision models")
    
    print()
    
    # Run the evaluation
    success = run_evaluation(pdf_path, ground_truth_path, args)
    
    if success:
        print("\n✅ Evaluation completed successfully!")
        output_dir = args.output_dir or "evaluation_results"
        print(f"📊 Results saved to: {output_dir}/")
        return 0
    else:
        print("\n❌ Evaluation failed!")
        return 1

if __name__ == "__main__":
    exit(main())
