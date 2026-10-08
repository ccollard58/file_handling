# Vision Model Evaluation System for Handwritten Text Extraction

This system provides automated evaluation of multiple vision-capable LLMs running via Ollama, specifically assessing their ability to extract handwritten text from multi-page scanned documents (PDFs with embedded images). The evaluation process leverages an LLM-as-a-judge framework to benchmark model performance against ground truth.

## Features

- **Automated Model Discovery**: Discovers all vision-capable models available in your local Ollama installation
- **Comprehensive Evaluation**: Tests multiple metrics including Levenshtein distance, Word Error Rate (WER), character accuracy, and semantic similarity
- **LLM-as-a-Judge**: Uses an LLM to evaluate semantic similarity between extracted text and ground truth
- **PDF Processing**: Converts multi-page PDFs to images for vision model processing
- **Detailed Reporting**: Generates both machine-readable JSON results and human-readable reports
- **Model Ranking**: Automatically ranks models by composite performance scores

## System Requirements

- Python 3.8+
- Ollama server running locally
- At least one vision-capable model installed in Ollama
- Windows, macOS, or Linux

## Installation

1. **Clone/Download the files**:
   ```bash
   # Ensure you have these files in your working directory:
   # - evaluate_handwriting_vision_models.py
   # - demo_evaluation.py  
   # - setup_checker.py
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Check system setup**:
   ```bash
   python setup_checker.py
   ```

4. **Install and start Ollama** (if not already done):
   ```bash
   # Install Ollama from https://ollama.ai
   
   # Start Ollama server
   ollama serve
   
   # Install vision models (in another terminal)
   ollama pull llava:latest
   ollama pull qwen2-vl:latest
   ollama pull moondream:latest
   ```

## Quick Start

### 1. Check Available Files
```bash
python demo_evaluation.py --list-files
```

### 2. Run Basic Evaluation
```bash
# Uses default PDF and ground truth files
python demo_evaluation.py
```

### 3. Evaluate Specific Files
```bash
python demo_evaluation.py --pdf path/to/document.pdf --ground-truth path/to/ground_truth.txt
```

### 4. Test Specific Models
```bash
python demo_evaluation.py --models llava:latest qwen2-vl:latest
```

## Advanced Usage

### Direct Script Usage

```bash
python evaluate_handwriting_vision_models.py <pdf_path> <ground_truth_path> [options]
```

**Options:**
- `--output-dir DIR`: Output directory for results (default: evaluation_results)
- `--ollama-url URL`: Ollama server URL (default: http://localhost:11434)
- `--judge-model MODEL`: Model for LLM-as-a-judge (default: llama3.1:latest)
- `--models MODEL1 MODEL2 ...`: Specific models to test

**Example:**
```bash
python evaluate_handwriting_vision_models.py \
    test_docs/document_0001.pdf \
    test_docs/sample_ground_truth.txt \
    --output-dir custom_results \
    --models llava:latest qwen2-vl:latest
```

### Batch Evaluation

To evaluate multiple documents, create a simple script:

```python
import os
from pathlib import Path
import subprocess

# Define your document pairs
document_pairs = [
    ("doc1.pdf", "doc1_truth.txt"),
    ("doc2.pdf", "doc2_truth.txt"),
    ("doc3.pdf", "doc3_truth.txt")
]

for pdf, truth in document_pairs:
    print(f"Evaluating {pdf}...")
    subprocess.run([
        "python", "evaluate_handwriting_vision_models.py",
        pdf, truth,
        "--output-dir", f"results_{Path(pdf).stem}"
    ])
```

## Input File Requirements

### PDF Files
- Multi-page scanned documents with embedded images
- Handwritten or mixed handwritten/printed text
- Common formats: .pdf

### Ground Truth Files
- Plain text files (.txt) with UTF-8 encoding
- Should contain the human-validated transcription of the handwritten content
- Should match the content structure of the PDF as closely as possible

**Example ground truth format:**
```
Medical Report

Patient: John Doe
Date: 2024-03-15
Doctor: Dr. Smith

Chief Complaint: Annual physical examination
...
```

## Output Files

The system generates several output files:

### 1. JSON Results (`evaluation_results_YYYYMMDD_HHMMSS.json`)
Machine-readable results containing:
- Detailed metrics for each model
- Model rankings
- Execution times
- Error messages (if any)

### 2. Human-Readable Report (`evaluation_report_YYYYMMDD_HHMMSS.txt`)
Formatted report with:
- Model ranking table
- Detailed performance metrics
- Sample extracted text
- Summary statistics

## Evaluation Metrics

The system calculates multiple metrics to assess model performance:

### 1. **Character Accuracy** (0.0 - 1.0, higher is better)
- Measures character-level similarity using sequence matching
- Good for assessing overall transcription quality

### 2. **Word Error Rate (WER)** (0.0+, lower is better)
- Standard metric for transcription evaluation
- Calculates insertions, deletions, and substitutions at word level

### 3. **Levenshtein Distance** (0.0 - 1.0, lower is better)
- Normalized edit distance between texts
- Measures minimum number of character edits needed

### 4. **Semantic Similarity** (0.0 - 1.0, higher is better)
- Uses LLM-as-a-judge to evaluate meaning preservation
- Assesses whether key information is preserved even if exact wording differs

### 5. **Composite Score** (0.0 - 1.0, higher is better)
- Weighted combination of all metrics
- Used for final model ranking
- Weights: Semantic (40%), Character (30%), WER (20%), Levenshtein (10%)

## Supported Vision Models

The system automatically detects vision-capable models based on common naming patterns:

**Recommended Models:**
- `llava:latest` - General-purpose vision model
- `qwen2-vl:latest` - High-performance vision-language model
- `qwen2.5-vl:latest` - Latest Qwen vision model
- `moondream:latest` - Compact vision model
- `phi3.5-vision:latest` - Microsoft's vision model
- `llama3.2-vision:latest` - Meta's vision model

**Installation:**
```bash
ollama pull llava:latest
ollama pull qwen2-vl:latest
ollama pull moondream:latest
```

## Performance Considerations

### System Resources
- Vision models require significant GPU memory (8GB+ recommended)
- PDF processing can be memory-intensive for large documents
- Evaluation time scales with: number of models × number of pages × model inference time

### Optimization Tips
1. **Limit page count**: Large PDFs are automatically limited to first 5 pages
2. **Use specific models**: Test specific models instead of all available ones
3. **Batch processing**: Process multiple documents sequentially rather than in parallel
4. **Monitor GPU memory**: Ensure sufficient GPU memory for large vision models

## Troubleshooting

### Common Issues

1. **"No vision-capable models found"**
   ```bash
   # Install a vision model
   ollama pull llava:latest
   ```

2. **"Cannot connect to Ollama server"**
   ```bash
   # Start Ollama server
   ollama serve
   ```

3. **"PDF processing failed"**
   ```bash
   # Install PDF processing dependencies
   pip install PyMuPDF pdf2image
   # For Windows, install Poppler
   ```

4. **"Judge LLM not available"**
   ```bash
   # Install a text model for judging
   ollama pull llama3.1:latest
   ```

### Debug Mode

For detailed debugging information:

```bash
# Enable debug logging
export PYTHONPATH=.
python -c "
import logging
logging.basicConfig(level=logging.DEBUG)
from evaluate_handwriting_vision_models import VisionModelEvaluator
# ... rest of your code
"
```

### Checking System Status

Run the setup checker to diagnose issues:

```bash
python setup_checker.py
```

## API Reference

### VisionModelEvaluator Class

```python
from evaluate_handwriting_vision_models import VisionModelEvaluator

# Initialize evaluator
evaluator = VisionModelEvaluator(
    ollama_base_url="http://localhost:11434",
    judge_model="llama3.1:latest"
)

# Discover available models
models = evaluator.discover_vision_models()

# Evaluate all models
summary = evaluator.evaluate_all_models(
    pdf_path="document.pdf",
    ground_truth_path="ground_truth.txt",
    output_dir="results"
)

# Access results
print(f"Best model: {summary.best_model}")
print(f"Best score: {summary.best_score}")
```

### EvaluationResult Dataclass

```python
@dataclass
class EvaluationResult:
    model_name: str
    extracted_text: str
    levenshtein_distance: float
    word_error_rate: float
    character_accuracy: float
    semantic_similarity: float
    execution_time: float
    error_message: Optional[str] = None
```

## Contributing

To extend the evaluation system:

1. **Add new metrics**: Implement in `VisionModelEvaluator` class
2. **Support new model types**: Update `discover_vision_models()` method
3. **Add new output formats**: Extend `_save_results()` method

## License

This evaluation system is provided as-is for research and evaluation purposes.

## Changelog

### Version 1.0 (July 17, 2025)
- Initial release
- Support for multiple vision models
- LLM-as-a-judge evaluation
- Comprehensive metric calculation
- PDF processing with multiple backends
- Detailed reporting system
