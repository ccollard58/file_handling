#!/usr/bin/env python3
"""
Setup and dependency checker for the Vision Model Evaluation System

This script helps ensure all required dependencies are installed and 
the system is properly configured for running vision model evaluations.
"""

import os
import sys
import subprocess
import importlib
import requests
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} detected. Python 3.8+ required.")
        return False
    else:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible.")
        return True

def check_package(package_name, import_name=None):
    """Check if a package is installed."""
    if import_name is None:
        import_name = package_name
    
    try:
        importlib.import_module(import_name)
        print(f"✅ {package_name} is installed")
        return True
    except ImportError:
        print(f"❌ {package_name} is not installed")
        return False

def install_package(package_name):
    """Install a package using pip."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        print(f"✅ Successfully installed {package_name}")
        return True
    except subprocess.CalledProcessError:
        print(f"❌ Failed to install {package_name}")
        return False

def check_ollama_connection():
    """Check if Ollama server is accessible."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            print(f"✅ Ollama server is running with {len(models)} models")
            return True, models
        else:
            print(f"❌ Ollama server responded with status {response.status_code}")
            return False, []
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to Ollama server: {e}")
        return False, []

def check_vision_models(models):
    """Check for vision-capable models."""
    vision_keywords = [
        'llava', 'vision', 'multimodal', 'minicpm', 'moondream', 
        'bakllava', 'cogvlm', 'qwen2-vl', 'qwen2.5-vl', 'pixtral',
        'gemma2-vl', 'phi3.5-vision', 'llama3.2-vision'
    ]
    
    vision_models = []
    for model in models:
        model_name = model.get("name", "").lower()
        if any(keyword in model_name for keyword in vision_keywords):
            vision_models.append(model["name"])
    
    if vision_models:
        print(f"✅ Found {len(vision_models)} vision-capable models:")
        for model in vision_models:
            print(f"   - {model}")
        return True
    else:
        print("❌ No vision-capable models found")
        print("   Consider installing models like: ollama pull llava:latest")
        return False

def check_poppler():
    """Check if Poppler is available (Windows-specific)."""
    if os.name != 'nt':
        return True  # Non-Windows systems typically have poppler in PATH
    
    possible_paths = [
        r"C:\Program Files\poppler\bin",
        r"C:\Program Files (x86)\poppler\bin",
        r"C:\poppler\bin",
        r"C:\tools\poppler\bin"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"✅ Poppler found at: {path}")
            return True
    
    print("❌ Poppler not found. PDF processing may fail.")
    print("   Download from: https://poppler.freedesktop.org/")
    print("   Or install via: conda install poppler")
    return False

def main():
    print("=" * 60)
    print("VISION MODEL EVALUATION SYSTEM - SETUP CHECKER")
    print("=" * 60)
    
    all_good = True
    
    # Check Python version
    print("\n1. Checking Python version...")
    if not check_python_version():
        all_good = False
    
    # Check core dependencies
    print("\n2. Checking core dependencies...")
    required_packages = [
        ("langchain-ollama", "langchain_ollama"),
        ("requests", "requests"),
        ("Pillow", "PIL"),
        ("pathlib", "pathlib"),
    ]
    
    for package, import_name in required_packages:
        if not check_package(package, import_name):
            all_good = False
    
    # Check optional but recommended packages
    print("\n3. Checking optional dependencies...")
    optional_packages = [
        ("PyMuPDF", "fitz"),
        ("pdf2image", "pdf2image"),
        ("python-Levenshtein", "Levenshtein"),
        ("evaluate", "evaluate"),
    ]
    
    missing_optional = []
    for package, import_name in optional_packages:
        if not check_package(package, import_name):
            missing_optional.append(package)
    
    # Check Poppler (for PDF processing)
    print("\n4. Checking Poppler...")
    check_poppler()
    
    # Check Ollama server
    print("\n5. Checking Ollama server...")
    ollama_ok, models = check_ollama_connection()
    if ollama_ok:
        check_vision_models(models)
    else:
        all_good = False
    
    # Check if evaluation script exists
    print("\n6. Checking evaluation script...")
    if Path("evaluate_handwriting_vision_models.py").exists():
        print("✅ Evaluation script found")
    else:
        print("❌ Evaluation script not found")
        all_good = False
    
    # Summary and recommendations
    print("\n" + "=" * 60)
    print("SETUP SUMMARY")
    print("=" * 60)
    
    if all_good:
        print("🎉 System is ready for vision model evaluation!")
    else:
        print("⚠️  Some issues detected. Please address them before running evaluations.")
    
    if missing_optional:
        print(f"\n📦 Optional packages to install for better performance:")
        for package in missing_optional:
            print(f"   pip install {package}")
    
    # Quick install option
    if missing_optional:
        print("\n" + "-" * 40)
        response = input("Install missing optional packages now? (y/N): ")
        if response.lower() == 'y':
            print("Installing packages...")
            for package in missing_optional:
                install_package(package)
    
    print("\n📋 Next steps:")
    print("1. Make sure Ollama is running: ollama serve")
    print("2. Install vision models: ollama pull llava:latest")
    print("3. Run demo: python demo_evaluation.py --list-files")
    print("4. Start evaluation: python demo_evaluation.py")
    
    return 0 if all_good else 1

if __name__ == "__main__":
    exit(main())
