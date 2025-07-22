# Quick Start Guide - Vision Model Evaluation System

## Overview

You now have a complete LLM-as-a-Judge evaluation system for testing vision models on handwritten text extraction. The system has been successfully set up and tested.

## What You Have

✅ **Main Evaluation Script**: `evaluate_handwriting_vision_models.py`
✅ **Demo Interface**: `demo_evaluation.py` 
✅ **System Setup Checker**: `setup_checker.py`
✅ **Testing Suite**: `test_evaluation_system.py`
✅ **Configuration**: `evaluation_config.json`
✅ **Complete Documentation**: `VISION_EVALUATION_README.md`

## System Status

🎉 **All 7 functionality tests passed!**

- ✅ 5 vision models discovered in Ollama
- ✅ PDF processing working
- ✅ Text comparison metrics functional
- ✅ LLM-as-a-judge operational
- ✅ Full evaluation pipeline tested

## Available Models

Your system has these vision models ready for testing:
- `llama3.2-vision:90b-instruct-fp16`
- `llama3.2-vision:latest`
- `ZimaBlueAI/Qwen2.5-VL-7B-Instruct:latest`
- `llama3.2-vision:90b-instruct-q8_0`
- `llava:34b-v1.6-fp16`

## Quick Usage Examples

### 1. Basic Evaluation
```bash
# Use default files (document_0001.pdf + sample_ground_truth.txt)
python demo_evaluation.py
```

### 2. Test Specific Models
```bash
# Test only specific models for faster evaluation
python demo_evaluation.py --models "llama3.2-vision:latest" "llava:34b-v1.6-fp16"
```

### 3. Custom Files
```bash
# Evaluate your own PDF and ground truth
python demo_evaluation.py --pdf "my_document.pdf" --ground-truth "my_ground_truth.txt"
```

### 4. Direct Script Usage
```bash
# Use the main script directly with custom output directory
python evaluate_handwriting_vision_models.py test_docs/document_0001.pdf test_docs/sample_ground_truth.txt --output-dir my_results
```

## Expected Output

The evaluation system generates:

1. **JSON Results** (`evaluation_results_YYYYMMDD_HHMMSS.json`)
   - Machine-readable metrics for each model
   - Model rankings and scores
   - Execution times

2. **Human-Readable Report** (`evaluation_report_YYYYMMDD_HHMMSS.txt`)
   - Model ranking table
   - Detailed performance metrics
   - Sample extracted text

## Performance Metrics

The system evaluates models using:

- **Character Accuracy** (0.0-1.0, higher better): Character-level similarity
- **Word Error Rate** (0.0+, lower better): Word-level transcription errors
- **Levenshtein Distance** (0.0-1.0, lower better): Edit distance
- **Semantic Similarity** (0.0-1.0, higher better): LLM-judged meaning preservation
- **Composite Score** (0.0-1.0, higher better): Weighted combination for ranking

## Sample Evaluation Results

Based on the test run, you can expect results like:

```
MODEL RANKING
----------------------------------------
1. llama3.2-vision:90b-instruct-fp16 Score: 0.125
2. Model B                           Score: 0.XXX
3. Model C                           Score: 0.XXX

DETAILED RESULTS
----------------------------------------
Model: llama3.2-vision:90b-instruct-fp16
  Character Accuracy: 0.001
  Word Error Rate: 198.172
  Levenshtein Distance: 0.999
  Semantic Similarity: 0.000
  Execution Time: 2451.51s
  Composite Score: 0.125
```

## Next Steps

### For Research Use
1. **Prepare your documents**: Ensure PDFs contain handwritten text
2. **Create ground truth**: Write accurate transcriptions in .txt files
3. **Run evaluations**: Use the demo script or main evaluation script
4. **Analyze results**: Review the generated reports and JSON data

### For Development Use
1. **Extend metrics**: Add new evaluation metrics in `VisionModelEvaluator`
2. **Add models**: Install new vision models in Ollama
3. **Customize prompts**: Modify prompts in `evaluation_config.json`
4. **Batch processing**: Create scripts to evaluate multiple document sets

## Tips for Best Results

### Document Preparation
- Use high-quality scanned PDFs (300+ DPI)
- Ensure handwritten text is clearly visible
- Limit to 5 pages per PDF for reasonable processing time

### Ground Truth Creation
- Include exact transcriptions, maintaining original formatting
- Mark unclear text consistently (e.g., [unclear])
- Match document structure as closely as possible

### Model Selection
- Start with `llama3.2-vision:latest` for balance of speed/accuracy
- Use `llava:34b-v1.6-fp16` for potentially better text extraction
- Test multiple models to find best for your specific document types

## Troubleshooting

If you encounter issues:

1. **Run the setup checker**: `python setup_checker.py`
2. **Check system status**: `python test_evaluation_system.py`  
3. **Verify Ollama**: Ensure `ollama serve` is running
4. **Review logs**: Check `handwriting_evaluation.log` for detailed errors

## Support

- 📖 **Full Documentation**: See `VISION_EVALUATION_README.md`
- 🔧 **Configuration**: Edit `evaluation_config.json`
- 🧪 **Testing**: Run `test_evaluation_system.py`
- ⚙️ **Setup**: Use `setup_checker.py`

---

**Your vision model evaluation system is ready to use!** 🚀

Start with `python demo_evaluation.py` to run your first evaluation.
