# FinQA: LoRA vs In-Context Learning Comparison

Comparing **LoRA Fine-Tuning** vs **In-Context Learning (ICL)** for financial reasoning on the FinQA dataset.

## 🎯 Project Overview

This project evaluates two approaches for teaching language models to generate executable reasoning programs that answer numerical questions about financial reports:

1. **LoRA Fine-Tuning**: Parameter-efficient training with Low-Rank Adaptation
2. **In-Context Learning**: Few-shot prompting without training

**Models**: Llama-3-8B-Instruct, Mistral-7B-Instruct-v0.2  
**Dataset**: FinQA (6,251 train / 883 dev / 1,147 test examples)  
**Operations**: 6 mathematical functions (add, subtract, multiply, divide, greater, exp)

---

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone repository
git clone https://github.com/abhisvakil/FinQA-Mini-Project.git
cd FinQA-Mini-Project

# Create conda environment
conda env create -f environment.yml
conda activate finqa-mini

# Or use pip
pip install -r requirements.txt
```

### 2. Prepare Data

```bash
# Load and simplify FinQA dataset
cd src
python data_loader_simplified.py
cd ..
```

This creates simplified datasets in `data/simplified/`:
- `train_simplified.json` (6,251 examples)
- `dev_simplified.json` (883 examples)
- `test_simplified.json` (1,147 examples)

### 3. Run Experiments

```bash
# Make script executable
chmod +x run_all_experiments.sh

# Train LoRA models (2.5-3 hours per model)
./run_all_experiments.sh train-lora-mistral
./run_all_experiments.sh train-lora-llama

# Run LoRA inference (20-30 min per model)
./run_all_experiments.sh infer-lora-mistral
./run_all_experiments.sh infer-lora-llama

# Run ICL inference (20-30 min per model)
./run_all_experiments.sh infer-icl-mistral
./run_all_experiments.sh infer-icl-llama
```

### 4. Evaluate Results

```bash
cd src

# Evaluate LoRA predictions
python evaluate.py --predictions_file ../results/predictions/Mistral-7B-Instruct-v0.2_lora_predictions.json
python evaluate.py --predictions_file ../results/predictions/Meta-Llama-3-8B-Instruct_lora_predictions.json

# Evaluate ICL predictions
python evaluate.py --predictions_file ../results/icl/Mistral-7B-Instruct-v0.2/5shot_diverse/predictions.json
python evaluate.py --predictions_file ../results/icl/Meta-Llama-3-8B-Instruct/5shot_diverse/predictions.json
```

---

## 📁 Project Structure

```
FinQA-Mini-Project/
├── data/
│   └── simplified/                 # Processed datasets
│       ├── train_simplified.json
│       ├── dev_simplified.json
│       └── test_simplified.json
│
├── src/
│   ├── data_loader_simplified.py   # Data loading & preprocessing
│   ├── train_lora.py               # LoRA fine-tuning
│   ├── inference.py                # LoRA inference (unified)
│   ├── icl_inference.py            # ICL inference with YAML config
│   ├── evaluate.py                 # Evaluation metrics
│   └── executor.py                 # Program execution engine
│
├── configs/
│   ├── lora_config.yaml            # LoRA hyperparameters
│   ├── qlora_config.yaml           # QLoRA hyperparameters (optional)
│   ├── icl_config_1.yaml           # ICL config (5-shot diverse)
│   └── icl_config_2.yaml           # ICL config (alternative)
│
├── results/
│   ├── lora/                       # LoRA adapters (~50MB per model)
│   │   ├── Mistral-7B-Instruct-v0.2/
│   │   │   └── final_model/
│   │   └── Meta-Llama-3-8B-Instruct/
│   │       └── final_model/
│   │
│   ├── icl/                        # ICL results (organized by config)
│   │   ├── Mistral-7B-Instruct-v0.2/
│   │   │   └── 5shot_diverse/
│   │   │       ├── predictions.json
│   │   │       ├── config.yaml
│   │   │       └── metadata.json
│   │   └── Meta-Llama-3-8B-Instruct/
│   │       └── 5shot_diverse/
│   │           ├── predictions.json
│   │           ├── config.yaml
│   │           └── metadata.json
│   │
│   └── predictions/                # LoRA predictions
│       ├── Mistral-7B-Instruct-v0.2_lora_predictions.json
│       └── Meta-Llama-3-8B-Instruct_lora_predictions.json
│
├── run_all_experiments.sh          # Main experiment runner
├── environment.yml                 # Conda environment
├── requirements.txt                # Python dependencies
├── README.md                       # This file
└── PROJECT_GUIDE.md                # Detailed documentation
```

---

## 🧪 Methods Comparison

| Method | Training | Inference | Adapter Size | Memory | Key Feature |
|--------|----------|-----------|--------------|---------|-------------|
| **LoRA** | 2.5-3 hrs | 20-30 min | ~50 MB | ~20 GB | Parameter-efficient fine-tuning |
| **ICL** | None | 20-30 min | None | ~20 GB | Few-shot prompting (5 examples) |

*Times per model on AWS g5.2xlarge (NVIDIA A10G 24GB)*

---

## 💡 Key Features

### Unified ICL Inference
- ✅ **YAML-driven configuration**: All settings (model, generation params, ICL config) in one file
- ✅ **Organized results**: Saves predictions, config, and metadata together in structured folders
- ✅ **Consistent format**: Same prediction format as LoRA for fair comparison
- ✅ **Easy comparison**: Run multiple ICL configs (3-shot, 5-shot, 10-shot) and compare

### LoRA Fine-Tuning
- ✅ **Efficient training**: Adapters are 1000x smaller than full models
- ✅ **Fast inference**: Merge adapters with base model
- ✅ **Consistent output**: Unified inference script for all methods
- ✅ **Memory optimized**: Works on 24GB GPU (batch_size=1 for Llama)

### Evaluation & Comparison
- ✅ **Execution accuracy**: Correct numerical answer
- ✅ **Program accuracy**: Correct reasoning program
- ✅ **Fair comparison**: Same base models, temperature (0.1), and test set

---

## 📊 Results Structure

### LoRA Predictions
```json
{
  "id": "example_id",
  "question": "What was the percentage change?",
  "predicted_program": "divide(subtract(150, 100), 100)",
  "predicted_answer": "0.5",
  "gold_program": "divide(subtract(150, 100), 100)",
  "gold_answer": "0.5",
  "raw_output": "..."
}
```

### ICL Results Organization
```
results/icl/{model_name}/{num_shots}shot_{selection_method}/
├── predictions.json    # Predictions in same format as LoRA
├── config.yaml        # Exact config used for this run
└── metadata.json      # Run info (timestamp, num_shots, temperature, etc.)
```

This structure allows easy comparison of different ICL configurations:
- `3shot_diverse/` vs `5shot_diverse/` vs `10shot_diverse/`
- `5shot_random/` vs `5shot_diverse/`

---

## 🔧 Requirements

### Python Dependencies
- Python 3.10+
- PyTorch 2.0+
- Transformers 4.36+
- PEFT 0.7+
- BitsAndBytes 0.41+
- PyYAML, tqdm, datasets

### Hardware
- **GPU**: 24GB VRAM recommended (NVIDIA A10G, RTX 3090, or better)
- **Storage**: ~10 GB for models + adapters + results
- **RAM**: 32GB+ recommended

### Recommended Setup
- **AWS EC2**: g5.2xlarge instance (~$1.20/hr)
- **OS**: Ubuntu 20.04+ or Amazon Linux 2023
- **CUDA**: 11.8+

---

## 📚 Detailed Documentation

See **PROJECT_GUIDE.md** for:
- Complete workflow explanation
- Detailed command reference
- AWS setup guide with cost estimates
- Troubleshooting common issues
- Implementation details for each method
- Evaluation metrics explained

---

## 🎓 Academic Context

This project is part of the 11-667 course mini-project comparing parameter-efficient fine-tuning (LoRA) with in-context learning (ICL) for financial reasoning tasks.

**Research Questions**:
1. Does fine-tuning with LoRA improve reasoning over few-shot ICL?
2. How do Llama-3-8B and Mistral-7B compare on FinQA?
3. What is the trade-off between training time/cost and performance?

---

## 🏗️ Implementation Details

### LoRA Configuration
```yaml
r: 8                    # Rank
alpha: 16               # Scaling factor
target_modules:         # Layers to adapt
  - q_proj
  - v_proj
  - k_proj
  - o_proj
dropout: 0.05
bias: none
```

### ICL Configuration
```yaml
icl:
  num_shots: 5                    # Number of examples in prompt
  example_selection: diverse      # Selection strategy

generation:
  temperature: 0.1               # Low for deterministic output
  max_new_tokens: 256
  do_sample: true
  top_p: 0.95
```

### Training Parameters
- **Epochs**: 3
- **Batch size**: 4 (Mistral), 1 (Llama - to avoid OOM)
- **Gradient accumulation**: 4 (Mistral), 16 (Llama)
- **Learning rate**: 2e-4
- **Optimizer**: AdamW with paged_adamw_8bit
- **Scheduler**: Linear warmup

---

## 📝 Citation

```bibtex
@inproceedings{chen2021finqa,
  title={FinQA: A Dataset of Numerical Reasoning over Financial Data},
  author={Chen, Zhiyu and Chen, Wenhu and Smiley, Charese and Shah, Sameena and 
          Borova, Iana and Langdon, Dylan and Moussa, Reema and Beane, Matt and 
          Huang, Ting-Hao and Routledge, Bryan and Wang, William Yang},
  booktitle={EMNLP},
  year={2021}
}
```

---

## 🤝 Contributing

This is an educational project. For questions or issues:
1. Check **PROJECT_GUIDE.md** for troubleshooting
2. Review experiment logs in project root
3. Ensure all dependencies are correctly installed

---

## 📄 License

This project is for educational purposes as part of the 11-667 course mini-project.

---

## 🔗 Resources

- **FinQA Dataset**: https://github.com/czyssrs/FinQA
- **FinQA Paper**: https://arxiv.org/abs/2109.00122
- **LoRA Paper**: https://arxiv.org/abs/2106.09685
- **Llama 3**: https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
- **Mistral**: https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2

