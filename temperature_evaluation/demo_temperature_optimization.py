#!/usr/bin/env python3
"""
Demo script for Temperature Optimization

This script demonstrates how to run temperature optimization for vision models
with different configurations and provides examples for various use cases.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

def check_requirements():
    """Check if required files and dependencies are available."""
    required_files = [
        "optimize_temperature.py",
        "evaluate_handwriting_vision_models.py"
    ]
    
    missing_files = []
    for file_name in required_files:
        if not Path(file_name).exists():
            missing_files.append(file_name)
    
    if missing_files:
        print(f"❌ Missing required files: {', '.join(missing_files)}")
        return False
    
    # Check for test files
    test_pdf = Path("test_docs/document_0001.pdf")
    test_ground_truth = Path("test_docs/sample_ground_truth.txt")
    
    if not test_pdf.exists():
        print(f"❌ Test PDF not found: {test_pdf}")
        return False
    
    if not test_ground_truth.exists():
        print(f"❌ Test ground truth not found: {test_ground_truth}")
        return False
    
    print("✅ All required files found")
    return True

def run_optimization(command_args):
    """Run the optimization with specified arguments."""
    cmd = [sys.executable, "optimize_temperature.py"] + command_args
    
    print(f"Running: {' '.join(cmd)}")
    print("-" * 60)
    
    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"❌ Optimization failed with exit code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print("\n⚠️ Optimization interrupted by user")
        return False

def main():
    """Main demo interface."""
    parser = argparse.ArgumentParser(description="Temperature Optimization Demo")
    parser.add_argument("--mode", choices=["quick", "comprehensive", "single-model", "custom"], 
                       default="quick", help="Optimization mode")
    parser.add_argument("--model", help="Specific model to test (for single-model mode)")
    parser.add_argument("--output-dir", help="Custom output directory")
    parser.add_argument("--runs", type=int, default=2, help="Runs per temperature")
    
    args = parser.parse_args()
    
    # Check requirements
    if not check_requirements():
        return 1
    
    print("🔬 TEMPERATURE OPTIMIZATION DEMO")
    print("=" * 50)
    
    # Base command arguments
    base_args = [
        "test_docs/document_0001.pdf",
        "test_docs/sample_ground_truth.txt",
        "--runs-per-temp", str(args.runs)
    ]
    
    if args.output_dir:
        base_args.extend(["--output-dir", args.output_dir])
    
    # Configure based on mode
    if args.mode == "quick":
        print("🚀 QUICK OPTIMIZATION")
        print("Testing 2 temperatures (0.0, 0.5) with 1 run each for fast results")
        print()
        
        command_args = base_args + [
            "--temperature-values", "0.0", "0.5",
            "--runs-per-temp", "1"
        ]
        
    elif args.mode == "comprehensive":
        print("🔬 COMPREHENSIVE OPTIMIZATION")
        print("Testing all temperature values (0.0-0.9) with multiple runs")
        print("⚠️  This will take a long time!")
        print()
        
        response = input("Continue with comprehensive optimization? (y/N): ")
        if response.lower() != 'y':
            return 0
        
        command_args = base_args + [
            "--runs-per-temp", str(max(2, args.runs))
        ]
        
    elif args.mode == "single-model":
        if not args.model:
            print("❌ --model argument required for single-model mode")
            return 1
        
        print(f"🎯 SINGLE MODEL OPTIMIZATION")
        print(f"Testing only: {args.model}")
        print()
        
        command_args = base_args + [
            "--models", args.model
        ]
        
    elif args.mode == "custom":
        print("⚙️  CUSTOM OPTIMIZATION")
        print("You can modify the command arguments below:")
        print()
        
        print("Available options:")
        print("  --models MODEL1 MODEL2 ...    # Test specific models")
        print("  --temperature-values 0.0 0.3 0.7  # Test specific temperatures")
        print("  --runs-per-temp N             # Number of runs per temperature")
        print("  --output-dir DIR              # Custom output directory")
        print()
        
        custom_args = input("Enter additional arguments (or press Enter for defaults): ").strip()
        if custom_args:
            command_args = base_args + custom_args.split()
        else:
            command_args = base_args
    
    else:
        print(f"❌ Unknown mode: {args.mode}")
        return 1
    
    # Run the optimization
    success = run_optimization(command_args)
    
    if success:
        print("\n✅ Temperature optimization completed successfully!")
        output_dir = args.output_dir or "temperature_optimization_results"
        print(f"📊 Results saved to: {output_dir}/")
        
        # List generated files
        result_dir = Path(output_dir)
        if result_dir.exists():
            files = list(result_dir.glob("temperature_optimization_*"))
            if files:
                print("\n📁 Generated files:")
                for file in sorted(files):
                    print(f"  - {file.name}")
    else:
        print("\n❌ Temperature optimization failed!")
        print("Check the logs for error details.")
    
    return 0 if success else 1

if __name__ == "__main__":
    try:
        exit(main())
    except KeyboardInterrupt:
        print("\n⚠️ Demo interrupted by user")
        exit(1)
