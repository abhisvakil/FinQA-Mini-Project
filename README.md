# Explainable Quantitative Reasoning for Financial Reports using FinQA

## Project Overview

This project implements a system for automated analysis of complex financial reports, answering numerical questions by generating executable, step-by-step reasoning programs. We compare **Parameter-Efficient Fine-Tuning (PEFT)** methods (LoRA/QLoRA) against **In-Context Learning (ICL)** across 2-4 open-source language models.

### Approach
- **PEFT Methods**: LoRA (Low-Rank Adaptation) and QLoRA (Quantized LoRA) for efficient fine-tuning
  - Models: **Llama-3-8B** and **Mistral-7B** (optimal size for fine-tuning)
- **ICL**: Few-shot prompting without fine-tuning
  - Models: **Llama-3-70B** and **Qwen-2.5-72B** (larger models for better reasoning)
- **Evaluation**: Compare accuracy, efficiency, and reasoning quality across methods

## Project Structure

```
FinQA-Mini-Project/
├── data/                    # Dataset files
│   ├── train.json          # Training data
│   ├── dev.json            # Development/validation data
│   └── test.json           # Test data
├── src/                    # Core source code
│   ├── data_loader.py      # Data loading utilities
│   ├── executor.py         # Program executor
│   ├── evaluate.py         # Evaluation metrics
│   ├── peft_trainer.py     # LoRA/QLoRA training
│   ├── icl_inference.py    # ICL inference
│   └── model_utils.py      # Model loading utilities
├── configs/                # Configuration files
│   ├── lora_config.yaml    # LoRA hyperparameters
│   ├── qlora_config.yaml   # QLoRA hyperparameters
│   └── icl_config.yaml     # ICL prompt templates
├── notebooks/              # Jupyter notebooks
│   ├── data_exploration.ipynb
│   ├── model_comparison.ipynb
│   └── error_analysis.ipynb
├── results/                # Results and outputs
│   ├── lora/               # LoRA results
│   ├── qlora/              # QLoRA results
│   └── icl/                # ICL results
├── requirements.txt        # Python dependencies
├── environment.yml         # Conda environment
└── README.md              # This file
```

## Setup Instructions

### 1. Create Conda Environment (Recommended)

We provide a `environment.yml` file that includes all dependencies and Python version configuration:

```bash
conda env create -f environment.yml
conda activate finqa-mini
```

### Alternative: Create Virtual Environment

If you prefer using venv instead:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download FinQA Dataset

```bash
# Clone the FinQA repository
git clone https://github.com/czyssrs/FinQA.git
cd FinQA

# The dataset is in the dataset/ folder
# Copy train.json, dev.json, test.json to this project's data/ folder
```

### 4. Set Up Project Directories

```bash
mkdir -p data models/retriever models/generator models/specialist models/icl
mkdir -p src notebooks results configs
```

## Quick Start

### 1. Data Exploration

```bash
python src/data_loader.py
```

### 2. LoRA Fine-tuning

```bash
python src/peft_trainer.py --model llama-3-8b --method lora --config configs/lora_config.yaml
```

### 3. QLoRA Fine-tuning

```bash
python src/peft_trainer.py --model mistral-7b --method qlora --config configs/qlora_config.yaml
```

### 4. ICL Inference

```bash
python src/icl_inference.py --model llama-3-8b --config configs/icl_config.yaml
```

### 5. Evaluation

```bash
python src/evaluate.py --predictions results/predictions.json --gold data/test.json
```

## Implementation Phases

1. **Phase 1**: Environment setup and data preparation ✅
2. **Phase 2**: Model selection and setup (2-4 open-source models)
3. **Phase 3**: LoRA fine-tuning implementation
4. **Phase 4**: QLoRA fine-tuning implementation
5. **Phase 5**: ICL inference with few-shot prompting
6. **Phase 6**: Comparative analysis and evaluation
7. **Phase 7**: Documentation and reporting

See `IMPLEMENTATION_PLAN.md` for detailed steps.

## Key Metrics

- **Execution Accuracy**: Percentage of questions with correct final answer
- **Program Accuracy**: Percentage of questions with correct reasoning program
- **Inference Time**: Average time per question
- **Reasoning Quality**: Qualitative assessment of program clarity

## References

- FinQA Dataset: https://github.com/czyssrs/FinQA
- FinQA Paper: https://arxiv.org/abs/2109.00122
- FinQA Leaderboard: https://finllm-leaderboard.readthedocs.io/

## License

This project is for educational purposes as part of the mini-project assignment.

