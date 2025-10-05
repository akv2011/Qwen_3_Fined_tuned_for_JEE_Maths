# JEE Mathematics Fine-tuning Project

Fine-tuning Qwen3-7B-Instruct on JEE (Joint Entrance Examination) Mathematics problems using Unsloth for optimized training.

## Project Structure

```
Qwen_3_Fined_tuned_for_JEE_Maths/
├── data/                          # Raw data and databases
│   ├── raw/                       # Original PDF files
│   │   └── 41 Years IIT JEE Mathematics by Amit M Agarwal.pdf
│   └── demo_data/                 # Demo database and samples
│       ├── jee_problems.db        # SQLite database (3 samples)
│       ├── processed/
│       ├── raw_pdfs/
│       ├── statistics/
│       └── structured/
│
├── datasets/                      # Processed datasets
│   └── jee_math/                  # HuggingFace format dataset
│       ├── dataset.json           # 625 JEE problems
│       ├── dataset_dict.json
│       ├── train/
│       └── test/
│
├── notebooks/                     # Jupyter notebooks
│   ├── fine_tune_qwen_with_unsloth.ipynb  # Main training notebook
│   ├── jee_math_env_and_smoke.ipynb
│   └── kaggle_jee_processing_pipeline.ipynb
│
├── scripts/                       # Python scripts
│   ├── analyze_dataset.py
│   ├── analyze_extraction.py
│   ├── create_huggingface_dataset.py
│   ├── create_jee_hf_dataset.py
│   ├── create_local_dataset.py
│   ├── convert_to_huggingface_format.py
│   ├── demo_local_dataset.py
│   ├── examine_data.py
│   ├── examine_llamaparse.py
│   ├── extraction_results_table.py
│   ├── jee_extraction_comparison.py
│   ├── jee_extraction_demo.py
│   ├── pdf_deep_analysis.py
│   ├── setup_extraction_comparison.py
│   ├── smoke_test.py
│   ├── test_extraction.py
│   └── test_langextract.py
│
├── results/                       # Analysis results
│   ├── extraction_results/        # PDF extraction outputs
│   ├── jee_sample_extraction.json
│   └── pdf_analysis_report.json
│
├── docs/                          # Documentation
│   ├── CLAUDE.md
│   ├── UPDATED_PLAN.md
│   ├── pdf_analysis_summary.md
│   ├── jee_sample_summary.txt
│   ├── requirements_comparison.txt
│   └── requirements_local.txt
│
├── terminal_outputs/              # Terminal logs
│
├── .env.example                   # Environment variables template
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd Qwen_3_Fined_tuned_for_JEE_Maths

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Setup Environment

```bash
# Copy environment template
cp .env.example .env

# Add your API keys to .env
# ANTHROPIC_API_KEY=your_key_here
# HUGGINGFACE_TOKEN=your_token_here
```

### 3. Run Fine-tuning

Open and run the main notebook:
```bash
jupyter notebook notebooks/fine_tune_qwen_with_unsloth.ipynb
```

## Dataset

- **Source**: 41 Years IIT JEE Mathematics by Amit M Agarwal
- **Size**: 625 problems extracted from PDF
- **Format**: Conversational format with problem/solution pairs
- **Topics**: Calculus, Algebra, Trigonometry, Geometry, etc.

## Model Details

- **Base Model**: Qwen 3-7B-Instruct (7.6B parameters)
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **Quantization**: 4-bit for memory efficiency
- **Framework**: Unsloth (2x faster fine-tuning)
- **Trainable Parameters**: ~40M (0.53% of total)

## Training Configuration

- **Batch Size**: 2 (effective: 8 with gradient accumulation)
- **Learning Rate**: 2e-4
- **Optimizer**: AdamW 8-bit
- **Max Steps**: 60
- **LoRA Rank**: 16
- **Target Modules**: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj

## Training Results

- **Final Loss**: 0.2250
- **Training Time**: ~9.34 minutes (560 seconds)
- **Hardware**: Tesla P100-PCIE-16GB GPU
- **Steps/Second**: 0.11

## Usage Example

```python
from unsloth import FastLanguageModel
import torch

# Load fine-tuned model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="outputs/final_model",
    max_seq_length=2048,
    load_in_4bit=True,
)

FastLanguageModel.for_inference(model)

# Test with a problem
problem = "Find the derivative of f(x) = x³ + 2x² - 5x + 3"
prompt = f"""Below is a mathematics problem. Solve it step by step.

### Problem:
{problem}

### Solution:"""

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=300)
solution = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(solution)
```

## 🛠️ Scripts Overview

### Data Processing
- `create_local_dataset.py` - Create local SQLite database
- `create_jee_hf_dataset.py` - Convert to HuggingFace format
- `convert_to_huggingface_format.py` - Format conversion utilities

### Analysis
- `analyze_dataset.py` - Dataset statistics and analysis
- `analyze_extraction.py` - PDF extraction quality analysis
- `pdf_deep_analysis.py` - Deep PDF structure analysis

### Extraction Comparison
- `jee_extraction_comparison.py` - Compare extraction methods
- `extraction_results_table.py` - Tabular comparison results

### Testing
- `smoke_test.py` - Quick functionality tests
- `test_extraction.py` - Test extraction pipeline
- `demo_local_dataset.py` - Demo dataset usage

## Key Notebooks

1. **fine_tune_qwen_with_unsloth.ipynb** - Main training pipeline
   - Data preparation (10 synthetic problems + 625 real problems)
   - Model loading and configuration
   - LoRA fine-tuning
   - Model testing and evaluation

2. **jee_math_env_and_smoke.ipynb** - Environment setup and smoke tests

3. **kaggle_jee_processing_pipeline.ipynb** - Full processing pipeline

## Next Steps

- [ ] Evaluate on full validation set
- [ ] Compare fine-tuned vs base model performance
- [ ] Upload to HuggingFace Hub
- [ ] Test on more complex JEE problems
- [ ] Create inference API
- [ ] Build web interface for problem solving

