# Explainable Quantitative Reasoning for Financial Reports using FinQA

## Project Overview

This project implements a system for automated analysis of complex financial reports, answering numerical questions by generating executable, step-by-step reasoning programs. We compare a fine-tuned specialist model against a large general-purpose in-context learning (ICL) model.

## Project Structure

```
LLM_Mini_Project/
├── data/                    # Dataset files (to be downloaded)
├── models/                  # Model implementations
│   ├── retriever/          # Retriever module
│   ├── generator/          # Program generator
│   ├── specialist/         # Fine-tuned specialist model
│   └── icl/                # ICL model implementation
├── src/                    # Core source code
│   ├── data_loader.py      # Data loading utilities
│   ├── preprocess.py       # Data preprocessing
│   ├── executor.py         # Program executor
│   └── evaluate.py         # Evaluation metrics
├── notebooks/              # Jupyter notebooks for exploration
├── results/                # Results and outputs
├── configs/                # Configuration files
├── requirements.txt        # Python dependencies
├── IMPLEMENTATION_PLAN.md  # Detailed implementation plan
└── TECHNICAL_GUIDE.md      # Technical specifications
```

## Setup Instructions

### 1. Create Virtual Environment

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

### Data Exploration

```bash
jupyter notebook notebooks/data_exploration.ipynb
```

### Training Baseline Model

```bash
# Train retriever
python models/retriever/train.py

# Train generator
python models/generator/train.py
```

### Evaluation

```bash
python src/evaluate.py --predictions results/predictions.json --gold data/test.json
```

## Implementation Phases

1. **Phase 1**: Environment setup and data preparation
2. **Phase 2**: Baseline implementation (FinQANet architecture)
3. **Phase 3**: Fine-tuned specialist model
4. **Phase 4**: Large general-purpose ICL model
5. **Phase 5**: Comparative analysis
6. **Phase 6**: Documentation and reporting

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

