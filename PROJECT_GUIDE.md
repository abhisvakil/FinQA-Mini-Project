# FinQA PEFT vs ICL - Complete Project Guide

## Table of Contents
1. [Overview](#overview)
2. [Complete Workflow](#complete-workflow)
3. [What Gets Saved](#what-gets-saved)
4. [Command Reference](#command-reference)
5. [AWS Setup Guide](#aws-setup-guide)
6. [Implementation Details](#implementation-details)
7. [Troubleshooting](#troubleshooting)

---

## Overview

This project compares three methods for financial reasoning:
- **LoRA**: Fine-tuning with Low-Rank Adaptation (efficient adapters)
- **QLoRA**: LoRA with 4-bit quantization (more memory efficient)
- **ICL**: In-Context Learning with few-shot prompting (no training)

**Goal**: Determine which method produces the best reasoning programs for numerical questions about financial reports.

**Models**: Llama-3-8B-Instruct, Mistral-7B-Instruct-v0.2  
**Dataset**: FinQA (6,251 train / 883 dev / 1,147 test)  
**Total Experiments**: 6 (2 models × 3 methods)

---

## Complete Workflow

```
┌──────────────┐
│   RAW DATA   │
│  FinQA JSON  │
└──────┬───────┘
       │
       │ data_loader_simplified.py
       ▼
┌──────────────────────┐
│  SIMPLIFIED DATA     │
│  Essential fields    │
│  only                │
└──────┬───────────────┘
       │
       ├────────────┬──────────────┬──────────────┐
       │            │              │              │
       ▼            ▼              ▼              │
  ┌────────┐  ┌─────────┐   ┌─────────┐         │
  │  LoRA  │  │ QLoRA   │   │   ICL   │         │
  │Training│  │Training │   │ (Direct)│         │
  │2.5-3hrs│  │2.5-3hrs │   │  No     │         │
  │        │  │         │   │Training │         │
  └───┬────┘  └────┬────┘   └────┬────┘         │
      │            │             │              │
      ▼            ▼             │              │
  ┌────────┐  ┌─────────┐       │              │
  │Adapters│  │Adapters │       │              │
  │ ~50MB  │  │ ~50MB   │       │              │
  └───┬────┘  └────┬────┘       │              │
      │            │             │              │
      ├────────────┴─────────────┴──────────────┘
      │
      ▼
┌──────────────────────┐
│    INFERENCE         │
│ inference.py (LoRA)  │
│ icl_inference.py     │
│                      │
│ Test set: 1,147      │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│   PREDICTIONS        │
│  6 JSON files        │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│    EVALUATION        │
│ Execution Accuracy   │
│ Program Accuracy     │
└──────────────────────┘
```

---

## What Gets Saved

### After Training

**LoRA** (`results/lora/`):
```
results/lora/
├── Meta-Llama-3-8B-Instruct/
│   ├── final_model/              ← Main output
│   │   ├── adapter_config.json
│   │   ├── adapter_model.bin     (~50MB)
│   │   └── tokenizer files
│   ├── checkpoint-500/           ← Intermediate checkpoints
│   ├── checkpoint-1000/
│   └── logs/
└── Mistral-7B-Instruct-v0.2/
    └── (same structure)
```

**QLoRA** (`results/qlora/`):
```
results/qlora/
├── Meta-Llama-3-8B-Instruct/
│   └── final_model/              ← Trained adapters
└── Mistral-7B-Instruct-v0.2/
    └── final_model/              ← Trained adapters
```

**ICL**: No training, nothing saved

### After Inference

**Predictions** (`results/predictions/`):
```
results/predictions/
├── Meta-Llama-3-8B-Instruct_lora_predictions.json
├── Meta-Llama-3-8B-Instruct_qlora_predictions.json
├── Meta-Llama-3-8B-Instruct_predictions.json      # ICL
├── Mistral-7B-Instruct-v0.2_lora_predictions.json
├── Mistral-7B-Instruct-v0.2_qlora_predictions.json
└── Mistral-7B-Instruct-v0.2_predictions.json      # ICL
```

**Prediction Format** (unified across all methods):
```json
[
  {
    "id": "test_0",
    "question": "What is the net income?",
    "predicted_program": "add(1234, 5678)",
    "predicted_answer": "6912",
    "gold_program": "add(1234, 5678)",
    "gold_answer": "6912",
    "raw_output": "Program: add(1234, 5678)\nAnswer: 6912"
  }
]
```

---

## Command Reference

### Quick Commands

```bash
# Full pipeline (train + infer)
./run_all_experiments.sh full                    # 12-17 hours

# Training only
./run_all_experiments.sh train-all               # 10-14 hours

# Inference only (after training)
./run_all_experiments.sh infer-all               # 2-3 hours

# Quick test
./run_all_experiments.sh test                    # ~5 minutes
```

### Individual Training

```bash
# LoRA
./run_all_experiments.sh train-lora-llama        # 2.5-3 hours
./run_all_experiments.sh train-lora-mistral      # 2.5-3 hours

# QLoRA
./run_all_experiments.sh train-qlora-llama       # 2.5-3.5 hours
./run_all_experiments.sh train-qlora-mistral     # 2.5-3.5 hours
```

### Individual Inference

```bash
# LoRA
./run_all_experiments.sh infer-lora-llama        # 20-30 minutes
./run_all_experiments.sh infer-lora-mistral      # 20-30 minutes

# QLoRA
./run_all_experiments.sh infer-qlora-llama       # 20-30 minutes
./run_all_experiments.sh infer-qlora-mistral     # 20-30 minutes

# ICL
./run_all_experiments.sh infer-icl-llama         # 20-30 minutes
./run_all_experiments.sh infer-icl-mistral       # 20-30 minutes
```

### Manual Commands (Advanced)

**Training**:
```bash
cd src

# LoRA
python train_lora.py \
    --model_name meta-llama/Meta-Llama-3-8B-Instruct \
    --data_dir ../data/simplified \
    --output_dir ../results/lora \
    --epochs 3 \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4

# QLoRA
python train_qlora.py \
    --model_name meta-llama/Meta-Llama-3-8B-Instruct \
    --data_dir ../data/simplified \
    --output_dir ../results/qlora \
    --epochs 3 \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4
```

**Inference**:
```bash
cd src

# LoRA/QLoRA
python inference.py \
    --model_name meta-llama/Meta-Llama-3-8B-Instruct \
    --adapter_path ../results/lora/Meta-Llama-3-8B-Instruct/final_model \
    --method lora \
    --data_dir ../data/simplified \
    --output_dir ../results/predictions \
    --temperature 0.1

# ICL
python icl_inference.py \
    --model_name meta-llama/Meta-Llama-3-8B-Instruct \
    --data_dir ../data/simplified \
    --output_dir ../results/predictions \
    --num_shots 5 \
    --temperature 0.1
```

**Evaluation**:
```bash
cd src

python evaluate.py \
    --predictions_file ../results/predictions/Meta-Llama-3-8B-Instruct_lora_predictions.json
```

---

## AWS Setup Guide

### Recommended Instance: g5.2xlarge

**Specs**:
- GPU: NVIDIA A10G (24GB VRAM)
- vCPUs: 8
- RAM: 32GB
- Storage: 200GB+ recommended
- Cost: $1.20/hr (or $0.36/hr with spot instances)

### Setup Steps

1. **Launch Instance**:
```bash
# Use AWS CLI or Console
# Select: g5.2xlarge, Deep Learning AMI, 200GB storage
```

2. **Install Dependencies**:
```bash
ssh -i your-key.pem ubuntu@your-instance-ip

# Clone repository
git clone https://github.com/your-repo/FinQA-Mini-Project.git
cd FinQA-Mini-Project

# Create environment
conda env create -f environment.yml
conda activate finqa-mini

# Or use pip
pip install -r requirements.txt
```

3. **Prepare Data**:
```bash
cd src
python data_loader_simplified.py
cd ..
```

4. **Run Experiments**:
```bash
# Quick test first
./run_all_experiments.sh test

# If successful, run full pipeline
./run_all_experiments.sh full
```

### Cost Estimates (g5.2xlarge @ $1.20/hr)

| Task | Duration | On-Demand Cost | Spot Cost (70% off) |
|------|----------|----------------|---------------------|
| LoRA Training (both) | 5-6 hrs | $6-7.20 | $1.80-2.16 |
| QLoRA Training (both) | 5-7 hrs | $6-8.40 | $1.80-2.52 |
| All Inference | 2-3 hrs | $2.40-3.60 | $0.72-1.08 |
| **TOTAL** | **12-17 hrs** | **$14-21** | **$4.32-6.30** |

**Recommendation**: Use spot instances for 70% savings!

### Monitoring Progress

```bash
# Check training progress
tail -f results/lora/Meta-Llama-3-8B-Instruct/logs/training_logs.txt

# Check GPU usage
nvidia-smi

# Check disk space
df -h
```

---

## Implementation Details

### Data Format

**Input** (simplified from FinQA):
```json
{
  "id": "example_0",
  "question": "What is the revenue growth?",
  "pre_text": ["Text context..."],
  "post_text": ["More context..."],
  "table": [["Header1", "Header2"], ["Value1", "Value2"]],
  "program": ["divide", "subtract", "..."],
  "answer": "0.234"
}
```

**Operations** (6 total):
```python
add(a, b)        # a + b
subtract(a, b)   # a - b
multiply(a, b)   # a * b
divide(a, b)     # a / b
greater(a, b)    # max(a, b)
exp(a, b)        # a ** b
```

### Model Configurations

**LoRA**:
```yaml
r: 8                    # Rank
lora_alpha: 16          # Scaling factor
target_modules:
  - q_proj
  - v_proj
  - k_proj
  - o_proj
lora_dropout: 0.05
bias: "none"
task_type: "CAUSAL_LM"
```

**QLoRA**:
```yaml
load_in_4bit: true
bnb_4bit_compute_dtype: bfloat16
bnb_4bit_use_double_quant: true
bnb_4bit_quant_type: "nf4"
+ LoRA config above
```

**ICL**:
```yaml
num_shots: 5
selection_strategy: "diverse"
temperature: 0.1
max_new_tokens: 256
```

### Training Parameters

```python
epochs: 3
batch_size: 4
gradient_accumulation_steps: 4
learning_rate: 2e-4
warmup_ratio: 0.03
weight_decay: 0.001
fp16: false
bf16: true
gradient_checkpointing: true
max_grad_norm: 0.3
```

---

## Troubleshooting

### Out of Memory (OOM)

**Problem**: `CUDA out of memory` error during training

**Solutions**:
1. Reduce batch size:
   ```bash
   python train_lora.py --batch_size 2 --gradient_accumulation_steps 8
   ```

2. Use gradient checkpointing (already enabled)

3. Reduce sequence length:
   ```bash
   python train_lora.py --max_length 1536
   ```

4. Switch to QLoRA (more memory efficient)

### Slow Training

**Problem**: Training taking much longer than expected

**Solutions**:
1. Check GPU utilization:
   ```bash
   nvidia-smi
   ```

2. Ensure using GPU:
   ```python
   import torch
   print(torch.cuda.is_available())  # Should be True
   ```

3. Reduce logging frequency:
   ```bash
   python train_lora.py --logging_steps 100
   ```

### Import Errors

**Problem**: `ModuleNotFoundError` for transformers, peft, etc.

**Solutions**:
1. Reinstall dependencies:
   ```bash
   pip install -r requirements.txt --force-reinstall
   ```

2. Check Python version (requires 3.10+):
   ```bash
   python --version
   ```

3. Verify environment activation:
   ```bash
   conda activate finqa-mini
   ```

### Adapter Not Found

**Problem**: `FileNotFoundError` when loading adapters for inference

**Solutions**:
1. Check adapter path exists:
   ```bash
   ls -la results/lora/Meta-Llama-3-8B-Instruct/final_model
   ```

2. Ensure training completed:
   ```bash
   # Check logs
   cat results/lora/Meta-Llama-3-8B-Instruct/logs/training_logs.txt
   ```

3. Use correct model name in inference script

### Poor Results

**Problem**: Low accuracy or nonsensical outputs

**Solutions**:
1. Check data loading:
   ```bash
   cd src
   python data_loader_simplified.py
   ```

2. Verify prediction format:
   ```bash
   cat results/predictions/*_predictions.json | head -50
   ```

3. Test with smaller sample:
   ```bash
   python train_lora.py --max_samples 100
   ```

4. Increase training epochs:
   ```bash
   python train_lora.py --epochs 5
   ```

---

## Tips and Best Practices

### Before Starting

- ✅ Test with 100 samples first (`--max_samples 100`)
- ✅ Verify data is loaded correctly
- ✅ Check disk space (need ~10GB)
- ✅ Use spot instances on AWS for cost savings
- ✅ Set up checkpointing for long runs

### During Training

- 📊 Monitor GPU usage with `nvidia-smi`
- 📁 Check logs regularly: `tail -f results/lora/.../logs/training_logs.txt`
- 💾 Verify checkpoints are being saved
- ⏰ Estimate remaining time from progress bars

### After Training

- ✓ Verify adapters saved: `ls results/lora/*/final_model`
- ✓ Check adapter file size (~50MB)
- ✓ Test inference on small sample first
- ✓ Compare outputs across methods

### Optimization

- **Parallel Training**: Train Llama and Mistral on separate GPUs if available
- **ICL First**: Run ICL inference while models are training (no dependencies)
- **Batch Inference**: Process multiple examples together for faster inference
- **Checkpoint Resume**: Save time by resuming from checkpoints if interrupted

---

## Timeline Summary

| Phase | Tasks | Duration | Cost (g5.2xlarge) |
|-------|-------|----------|-------------------|
| **Setup** | Environment, data prep | 30 min | $0.60 |
| **Training** | LoRA + QLoRA (2 models) | 10-14 hrs | $12-17 |
| **Inference** | All 6 configurations | 2-3 hrs | $2.40-3.60 |
| **Evaluation** | Calculate metrics | 30 min | $0.60 |
| **TOTAL** | | **13-18 hrs** | **$15-22** |

*Spot instances reduce cost by 70%: ~$5-7 total*

---

## Next Steps After Experiments

1. **Analyze Results**: Compare metrics across all 6 configurations
2. **Error Analysis**: Examine failed predictions to understand limitations
3. **Visualizations**: Create comparison charts and tables
4. **Report Writing**: Document findings and insights
5. **Optimization**: Fine-tune hyperparameters for better results

---

## Resources

- **FinQA Paper**: https://arxiv.org/abs/2109.00122
- **LoRA Paper**: https://arxiv.org/abs/2106.09685
- **QLoRA Paper**: https://arxiv.org/abs/2305.14314
- **Hugging Face PEFT**: https://github.com/huggingface/peft
- **Transformers Docs**: https://huggingface.co/docs/transformers
