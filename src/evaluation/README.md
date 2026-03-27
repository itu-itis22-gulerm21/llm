# Medical LLM Benchmark Evaluation

Comprehensive evaluation suite for fine-tuned medical Large Language Models (LLMs) on standard medical benchmarks.

## 📊 Overview

This evaluation framework tests medical LLMs on:
- **MedQA**: USMLE-style medical questions (4-option multiple choice)
- **PubMedQA**: Biomedical literature questions (yes/no/maybe)

**Metrics Calculated:**
- ✅ Accuracy
- ✅ F1 Score (Macro & Weighted)
- ✅ Precision & Recall
- ✅ Valid Response Rate
- ✅ Improvement over Base Models

## 🚀 Quick Start

### Option 1: Using Jupyter Notebook (Recommended for Interactive Use)

```bash
# Install dependencies
pip install -r requirements.txt

# Open notebook
jupyter notebook benchmark_evaluation.ipynb

# Run all cells
```

### Option 2: Using Python Script (Recommended for Automation)

```bash
# Install dependencies
pip install -r requirements.txt

# Run evaluation (500 samples per dataset)
python benchmark_evaluation.py --num_samples 500 --output_dir ./results

# Run on full datasets (will take hours)
python benchmark_evaluation.py --num_samples 0 --output_dir ./results_full
```

## 📁 File Structure

```
evaluation/
├── benchmark_evaluation.ipynb   # Interactive Jupyter notebook
├── benchmark_evaluation.py      # Automated Python script
├── requirements.txt             # Python dependencies
├── README.md                    # This file
└── evaluation_results/          # Output directory (created automatically)
    ├── model_comparison.csv     # CSV comparison table
    ├── evaluation_results.json  # Detailed results in JSON
    ├── benchmark_results.png    # Main visualization
    └── f1_score_comparison.png  # F1 score charts
```

## 🔧 Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended: 16GB+ VRAM)
- Google Colab (alternative to local GPU)

### Install Dependencies

```bash
pip install -r requirements.txt
```

**Note:** If you encounter issues with `unsloth`, you can skip it:
```bash
pip install torch transformers datasets accelerate bitsandbytes peft scikit-learn pandas numpy matplotlib seaborn tqdm
```

## 📖 Usage Examples

### Example 1: Quick Evaluation (500 samples)

```bash
python benchmark_evaluation.py --num_samples 500 --output_dir ./quick_results
```

### Example 2: Full Evaluation (All samples)

```bash
python benchmark_evaluation.py --num_samples 0 --output_dir ./full_results
```

### Example 3: Custom Configuration in Notebook

```python
# In notebook, modify these cells:
NUM_SAMPLES = 1000  # Number of samples to evaluate
OUTPUT_DIR = './custom_results/'
TEMPERATURE = 0.1  # Generation temperature
MAX_NEW_TOKENS = 256  # Maximum response length
```

## 📊 Output Files

### 1. `model_comparison.csv`
Comprehensive comparison table with columns:
- Model name
- Dataset (MedQA/PubMedQA)
- Base model accuracy
- Fine-tuned model accuracy
- Accuracy improvement (%)
- F1 scores and improvements

**Example:**
```csv
Model,Dataset,Base Accuracy,Fine-tuned Accuracy,Accuracy Improvement,Base F1,Fine-tuned F1,F1 Improvement
GEMMA,MedQA,0.4523,0.5124,+13.29%,0.4321,0.4987,+15.41%
```

### 2. `evaluation_results.json`
Detailed JSON with:
- All metrics for each model
- Sample predictions (first 10)
- Configuration parameters
- Model responses

### 3. Visualizations
- `benchmark_results.png`: 4-panel comparison chart
  - Base vs Fine-tuned accuracy (MedQA)
  - Base vs Fine-tuned accuracy (PubMedQA)
  - Improvement percentages (MedQA)
  - Improvement percentages (PubMedQA)
- `f1_score_comparison.png`: F1 score comparisons

## 🎯 Models Evaluated

### Fine-tuned Medical Models
1. **Gemma-1.1-7B-Medical** (`Shekswess/gemma-1.1-7b-it-bnb-4bit-medical`)
2. **LLaMA-2-7B-Medical** (`Shekswess/llama-2-7b-chat-bnb-4bit-medical`)
3. **LLaMA-3-8B-Medical** (`Shekswess/llama-3-8b-Instruct-bnb-4bit-medical`)
4. **Mistral-7B-Medical** (`Shekswess/mistral-7b-instruct-v0.2-bnb-4bit-medical`)

### Base Models (for comparison)
1. **Gemma-1.1-7B-Base** (`unsloth/gemma-1.1-7b-it-bnb-4bit`)
2. **LLaMA-2-7B-Base** (`unsloth/llama-2-7b-chat-bnb-4bit`)
3. **LLaMA-3-8B-Base** (`unsloth/llama-3-8b-Instruct-bnb-4bit`)
4. **Mistral-7B-Base** (`unsloth/mistral-7b-instruct-v0.2-bnb-4bit`)

## 📈 Metrics Explained

### Accuracy
Percentage of correct predictions:
```
Accuracy = (Correct Predictions) / (Total Predictions)
```

### F1 Score
Harmonic mean of precision and recall:
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

- **Macro F1**: Unweighted average across all classes
- **Weighted F1**: Weighted by class frequency

### Precision
Proportion of correct positive predictions:
```
Precision = True Positives / (True Positives + False Positives)
```

### Recall
Proportion of actual positives correctly identified:
```
Recall = True Positives / (True Positives + False Negatives)
```

### Valid Response Rate
Percentage of responses that could be parsed:
```
Valid Rate = (Valid Responses) / (Total Responses)
```

## 🔬 Technical Details

### Model Loading
- **Quantization**: 4-bit (NF4) with double quantization
- **Compute Dtype**: float16
- **Device Map**: Automatic GPU allocation

### Generation Parameters
- **Temperature**: 0.1 (low for deterministic outputs)
- **Max New Tokens**: 256
- **Sampling**: Greedy (do_sample=False)

### Answer Extraction
- **Multiple Choice**: Regex to find A/B/C/D
- **Yes/No/Maybe**: Case-insensitive keyword matching
- **Invalid Handling**: Marked as "INVALID" and excluded from metrics

## 🐛 Troubleshooting

### Issue: Out of Memory (OOM)

**Solution 1**: Reduce sample size
```bash
python benchmark_evaluation.py --num_samples 100
```

**Solution 2**: Use Google Colab
- Upload notebook to Colab
- Select GPU runtime (T4 or better)
- Run cells

### Issue: Model download fails

**Solution**: Check HuggingFace credentials
```bash
huggingface-cli login
```

### Issue: Dataset loading error

**Solution**: Try alternative datasets (code includes fallbacks)
```python
# MedQA alternatives are built into the code
# The script will automatically try backup datasets
```

### Issue: Slow evaluation

**Expected**: Full evaluation takes 2-4 hours per model
- MedQA: ~1000 samples × 4 models × 2 versions = 8000 inferences
- PubMedQA: ~1000 samples × 4 models × 2 versions = 8000 inferences
- Total: ~16,000 inferences

**Solution**: Use smaller sample size for testing
```bash
python benchmark_evaluation.py --num_samples 50  # ~15 minutes
```

## 📝 Interpretation Guide

### Good Results
- ✅ Accuracy > 50% on MedQA
- ✅ Accuracy > 60% on PubMedQA
- ✅ Positive improvement over base model
- ✅ Valid response rate > 90%

### Excellent Results
- 🌟 Accuracy > 60% on MedQA
- 🌟 Accuracy > 70% on PubMedQA
- 🌟 Improvement > 15% over base
- 🌟 Valid response rate > 95%

### Warning Signs
- ⚠️ Negative improvement (model degraded)
- ⚠️ Valid response rate < 80%
- ⚠️ Large variance across datasets

## 🔄 Integration with Main Project

### Recommended Placement

```
LLM-Medical-Finetuning/
├── src/
│   ├── data_processing/
│   ├── finetuning_notebooks/
│   └── evaluation/                    ← ADD HERE
│       ├── benchmark_evaluation.py
│       ├── benchmark_evaluation.ipynb
│       ├── requirements.txt
│       └── README.md
├── artifacts/
└── evaluation_results/                ← OUTPUT HERE
```

### Update Main README

Add to project README:
```markdown
## Evaluation

Benchmark evaluation on MedQA and PubMedQA:

\`\`\`bash
cd src/evaluation
python benchmark_evaluation.py --num_samples 500
\`\`\`

See [evaluation README](src/evaluation/README.md) for details.
```

## 📚 References

### Datasets
- **MedQA**: Jin et al. (2021) - "What Disease does this Patient Have?"
- **PubMedQA**: Jin et al. (2019) - "PubMedQA: A Dataset for Biomedical Research Question Answering"

### Benchmarks
- [MedQA on HuggingFace](https://huggingface.co/datasets/bigbio/med_qa)
- [PubMedQA on HuggingFace](https://huggingface.co/datasets/pubmed_qa)

## 🤝 Contributing

To add new benchmarks:

1. Add dataset loading function
2. Implement answer extraction logic
3. Update `evaluate_model_on_benchmark()` method
4. Add to visualization

Example:
```python
def load_medmcqa_dataset(num_samples=None):
    dataset = load_dataset("medmcqa", split="test")
    if num_samples:
        dataset = dataset.select(range(num_samples))
    return dataset
```

## 📄 License

This evaluation framework is part of the LLM-Medical-Finetuning project.
See main project LICENSE for details.

## 📧 Contact

For issues or questions:
- Open an issue on the main project repository
- Include evaluation logs and configuration

## ✨ Acknowledgments

- Built on HuggingFace Transformers
- Uses Unsloth for optimized inference
- Benchmark datasets from medical research community

---

**Last Updated**: 2025-11-13
**Version**: 1.0.0
