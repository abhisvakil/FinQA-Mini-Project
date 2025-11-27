# FinQA PEFT vs ICL Comparison

Comparing **Parameter-Efficient Fine-Tuning** (LoRA/QLoRA) vs **In-Context Learning** for financial reasoning on the FinQA dataset.

## 🎯 Project Goal

Compare fine-tuning methods (LoRA, QLoRA) against few-shot prompting (ICL) for generating executable reasoning programs that answer numerical questions about financial reports.

**Models**: Llama-3-8B-Instruct, Mistral-7B-Instruct-v0.2 (same models for all methods)  
**Dataset**: FinQA (6,251 train / 883 dev / 1,147 test examples)  
**Operations**: 6 mathematical functions (add, subtract, multiply, divide, greater, exp)

---

## 🚀 Quick Start

### 1. Setup Environment

```bash
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

This creates simplified datasets in `data/simplified/` with only essential fields.

### 3. Run Experiments

```bash
# Make script executable
chmod +x run_all_experiments.sh

# Option A: Run everything (12-17 hours on g5.2xlarge)
./run_all_experiments.sh full

# Option B: Train first, infer later
./run_all_experiments.sh train-all    # 10-14 hours
./run_all_experiments.sh infer-all    # 2-3 hours

# Option C: Individual experiments
./run_all_experiments.sh train-lora-llama
./run_all_experiments.sh infer-lora-llama
./run_all_experiments.sh infer-icl-llama
```

### 4. Evaluate Results

```bash
cd src
python evaluate.py --predictions_file ../results/predictions/Meta-Llama-3-8B-Instruct_lora_predictions.json
# Repeat for all 6 prediction files
```

---

## 📁 Project Structure

```
FinQA-Mini-Project/
├── data/
│   └── simplified/              # Processed datasets
│       ├── train_simplified.json
│       ├── dev_simplified.json
│       └── test_simplified.json
│
├── src/
│   ├── data_loader_simplified.py   # Data processing
│   ├── train_lora.py               # LoRA training
│   ├── train_qlora.py              # QLoRA training
│   ├── inference.py                # Unified inference (LoRA/QLoRA)
│   ├── icl_inference.py            # ICL inference
│   ├── evaluate.py                 # Evaluation metrics
│   └── executor.py                 # Program execution
│
├── configs/
│   ├── lora_config.yaml            # LoRA hyperparameters
│   ├── qlora_config.yaml           # QLoRA hyperparameters
│   └── icl_config.yaml             # ICL prompt templates
│
├── results/
│   ├── lora/                       # LoRA adapters (~50MB each)
│   ├── qlora/                      # QLoRA adapters (~50MB each)
│   └── predictions/                # Inference outputs (6 JSON files)
│
├── run_all_experiments.sh          # Main experiment runner
├── environment.yml                 # Conda environment
├── requirements.txt                # Python dependencies
├── README.md                       # This file
└── PROJECT_GUIDE.md                # Detailed guide
```

---

## 🧪 Experiment Overview

| Method | Training Time | Adapter Size | Inference Time | Memory |
|--------|--------------|--------------|----------------|--------|
| **LoRA** | 2.5-3 hrs | ~50 MB | 20-30 min | ~20 GB |
| **QLoRA** | 2.5-3.5 hrs | ~50 MB | 20-30 min | ~16 GB |
| **ICL** | None | None | 20-30 min | ~20 GB |

*Times per model on AWS g5.2xlarge (NVIDIA A10G 24GB)*

**Total configurations**: 6 (2 models × 3 methods)

---

## 💡 Key Features

- ✅ **Unified data format** - Simplified FinQA loader extracting only essential fields
- ✅ **Consistent predictions** - All methods output same JSON format for easy comparison
- ✅ **Efficient training** - LoRA adapters are 1000x smaller than full models
- ✅ **Memory efficient** - QLoRA trains with 4-bit quantization
- ✅ **Fair comparison** - Same base models used across all methods
- ✅ **Easy execution** - Single script runs all experiments

---

## 📊 Expected Results

After running experiments, you'll have:

**Predictions**: 6 JSON files in `results/predictions/`
- `Meta-Llama-3-8B-Instruct_lora_predictions.json`
- `Meta-Llama-3-8B-Instruct_qlora_predictions.json`
- `Meta-Llama-3-8B-Instruct_predictions.json` (ICL)
- `Mistral-7B-Instruct-v0.2_lora_predictions.json`
- `Mistral-7B-Instruct-v0.2_qlora_predictions.json`
- `Mistral-7B-Instruct-v0.2_predictions.json` (ICL)

**Evaluation Metrics**:
- Execution Accuracy (correct numerical answer)
- Program Accuracy (correct reasoning program)

---

## 🔧 Requirements

- Python 3.10+
- PyTorch 2.0+
- Transformers 4.36+
- PEFT 0.7+
- BitsAndBytes 0.41+
- GPU: 24GB VRAM recommended (NVIDIA A10G or better)

---

## 📚 Additional Documentation

See **PROJECT_GUIDE.md** for:
- Detailed workflow explanation
- Command reference
- Troubleshooting tips
- AWS setup guide
- Cost estimates

---

## 🏗️ Implementation Details

**LoRA Configuration**:
- Rank: 8
- Alpha: 16
- Target modules: q_proj, v_proj, k_proj, o_proj
- Dropout: 0.05

**QLoRA Configuration**:
- 4-bit quantization (NF4)
- Double quantization enabled
- Compute dtype: bfloat16

**ICL Configuration**:
- 5-shot examples (diverse selection)
- Temperature: 0.1
- Max tokens: 256

---

## 📝 Citation

```bibtex
@inproceedings{chen2021finqa,
  title={FinQA: A Dataset of Numerical Reasoning over Financial Data},
  author={Chen, Zhiyu and Chen, Wenhu and Smiley, Charese and Shah, Sameena and Borova, Iana and Langdon, Dylan and Moussa, Reema and Beane, Matt and Huang, Ting-Hao and Routledge, Bryan and Wang, William Yang},
  booktitle={EMNLP},
  year={2021}
}
```

---

## 📞 Support

For detailed instructions and troubleshooting, see **PROJECT_GUIDE.md**

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

