# Document Category Analyzer

This script analyzes a corpus of documents to suggest meaningful categories for document organization.

## Features

- Analyzes a representative sample of documents from your corpus
- Uses LLM to analyze document content and suggest categories
- Provides detailed descriptions and examples for each suggested category
- Saves analysis results to JSON for further review

## Requirements

- Python 3.7+
- Ollama with gemma3:27b-it-fp16 model
- Dependencies from the main document processing application

## Usage

```
python category_analyzer.py [document_directory_path]
```

If no path is provided, the script will default to `E:\Dropbox\Admin\Scanned Documents`.

## Output

The script will:

1. Log its progress to `category_analysis.log`
2. Print suggested categories to the console
3. Save detailed analysis results to `category_analysis_results.json`

## Customization

You can adjust the following parameters in the script:

- `sample_size`: Number of documents to analyze (default: 100)
- `supported_extensions`: File types to include in analysis
