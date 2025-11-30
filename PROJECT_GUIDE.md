# FinQA Project Guide - Complete Documentation

## Table of Contents
1. [Overview](#overview)
2. [Complete Workflow](#complete-workflow)
3. [Detailed Setup](#detailed-setup)
4. [Running Experiments](#running-experiments)
5. [Results Organization](#results-organization)
6. [Command Reference](#command-reference)
7. [AWS Setup Guide](#aws-setup-guide)
8. [Troubleshooting](#troubleshooting)
9. [Implementation Details](#implementation-details)

---

## Overview

This project compares two methods for financial reasoning on FinQA:

| Method | Description | Training Required | Key Advantage |
|--------|-------------|-------------------|---------------|
| **LoRA** | Low-Rank Adaptation fine-tuning | Yes (2.5-3 hrs) | Learns task-specific patterns |
| **ICL** | In-Context Learning (few-shot) | No | Zero training cost |

**Goal**: Determine which method produces better reasoning programs for numerical financial questions.

**Models**: 
- Llama-3-8B-Instruct (Meta)
- Mistral-7B-Instruct-v0.2 (Mistral AI)

**Dataset**: FinQA
- Train: 6,251 examples
- Dev: 883 examples
- Test: 1,147 examples

**Total Experiments**: 4 (2 models × 2 methods)

---

## Complete Workflow

```
┌─────────────────────────────────────────────────────────┐
│                    RAW FinQA DATASET                    │
│        (Loaded from HuggingFace datasets library)       │
└────────────────────┬────────────────────────────────────┘
                     │
                     │ data_loader_simplified.py
                     │ (Extracts essential fields only)
                     ▼
┌─────────────────────────────────────────────────────────┐
│               SIMPLIFIED DATASET FILES                  │
│   train_simplified.json / dev / test (in data/simplified/) │
└────────────┬────────────────────┬───────────────────────┘
             │                    │
             │                    │
    ┌────────▼────────┐    ┌──────▼──────┐
    │  LoRA TRAINING  │    │  ICL DIRECT │
    │   train_lora.py │    │  (No Train) │
    │   2.5-3 hours   │    │             │
    └────────┬────────┘    └──────┬──────┘
             │                    │
             ▼                    │
    ┌─────────────────┐           │
    │ LoRA ADAPTERS   │           │
    │   ~50MB each    │           │
    │ (results/lora/) │           │
    └────────┬────────┘           │
             │                    │
             ├────────────────────┘
             │
             ▼
    ┌─────────────────────────────────┐
    │        INFERENCE PHASE          │
    │  - inference.py (LoRA)          │
    │  - icl_inference.py (ICL)       │
    │  Test set: 1,147 examples       │
    │  Time: 20-30 min per model      │
    └────────┬────────────────────────┘
             │
             ▼
    ┌─────────────────────────────────┐
    │      PREDICTIONS SAVED          │
    │  LoRA: results/predictions/     │
    │  ICL: results/icl/{model}/...   │
    └────────┬────────────────────────┘
             │
             ▼
    ┌─────────────────────────────────┐
    │        EVALUATION               │
    │  evaluate.py                    │
    │  - Execution Accuracy           │
    │  - Program Accuracy             │
    └─────────────────────────────────┘
```

---

## Detailed Setup

### Prerequisites

1. **Python Environment**
   ```bash
   python --version  # Should be 3.10+
   ```

2. **GPU Requirements**
   - VRAM: 24GB+ recommended
   - CUDA: 11.8+
   - Tested on: NVIDIA A10G (AWS g5.2xlarge)

3. **Storage**
   - Code: ~500 MB
   - Data: ~50 MB
   - Models (cached): ~15 GB per model
   - Adapters: ~50 MB per model
   - Results: ~100 MB

### Installation Steps

#### Option 1: Conda (Recommended)
```bash
# Create environment from file
conda env create -f environment.yml
conda activate finqa-mini

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

#### Option 2: Pip + Virtual Environment
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

### Data Preparation

```bash
cd src
python data_loader_simplified.py
```

**What this does**:
1. Downloads FinQA from HuggingFace (automatic)
2. Extracts essential fields: id, question, pre_text, table, program, answer
3. Saves to `data/simplified/` as JSON files
4. Creates train/dev/test splits

**Output**:
```
data/simplified/
├── train_simplified.json  (6,251 examples, ~8 MB)
├── dev_simplified.json    (883 examples, ~1 MB)
└── test_simplified.json   (1,147 examples, ~1.5 MB)
```

---

## Running Experiments

### Using the Master Script

The `run_all_experiments.sh` script provides a unified interface for all experiments.

```bash
# Make executable (first time only)
chmod +x run_all_experiments.sh

# View all available commands
./run_all_experiments.sh
```

### Training Commands

#### Train LoRA - Mistral
```bash
./run_all_experiments.sh train-lora-mistral
```
- Duration: ~2.5-3 hours
- Memory: ~20 GB VRAM
- Output: `results/lora/Mistral-7B-Instruct-v0.2/final_model/`

#### Train LoRA - Llama
```bash
./run_all_experiments.sh train-lora-llama
```
- Duration: ~2.5-3 hours
- Memory: ~20 GB VRAM
- Settings: batch_size=1, gradient_accumulation=16 (OOM prevention)
- Output: `results/lora/Meta-Llama-3-8B-Instruct/final_model/`

### Inference Commands

#### LoRA Inference - Mistral
```bash
./run_all_experiments.sh infer-lora-mistral
```
- Duration: ~20-30 minutes
- Loads: `results/lora/Mistral-7B-Instruct-v0.2/final_model/`
- Output: `results/predictions/Mistral-7B-Instruct-v0.2_lora_predictions.json`

#### LoRA Inference - Llama
```bash
./run_all_experiments.sh infer-lora-llama
```
- Duration: ~20-30 minutes
- Loads: `results/lora/Meta-Llama-3-8B-Instruct/final_model/`
- Output: `results/predictions/Meta-Llama-3-8B-Instruct_lora_predictions.json`

#### ICL Inference - Mistral
```bash
./run_all_experiments.sh infer-icl-mistral
```
- Duration: ~20-30 minutes
- Config: `configs/icl_config_1.yaml` (5-shot diverse)
- Output: `results/icl/Mistral-7B-Instruct-v0.2/5shot_diverse/`

#### ICL Inference - Llama
```bash
./run_all_experiments.sh infer-icl-llama
```
- Duration: ~20-30 minutes
- Config: `configs/icl_config_1.yaml` (5-shot diverse)
- Output: `results/icl/Meta-Llama-3-8B-Instruct/5shot_diverse/`

### Running on AWS EC2 with nohup

For long-running processes, use nohup to keep running after disconnect:

```bash
# Activate environment and run in background
cd ~/FinQA-Mini-Project
source venv/bin/activate

# Training (long-running)
nohup ./run_all_experiments.sh train-lora-mistral > train_mistral.log 2>&1 &

# Monitor progress
tail -f train_mistral.log

# Check if still running
ps aux | grep train_lora
```

---

## Results Organization

### LoRA Results

```
results/
├── lora/
│   ├── Mistral-7B-Instruct-v0.2/
│   │   └── final_model/
│   │       ├── adapter_config.json
│   │       ├── adapter_model.bin (~50 MB)
│   │       └── README.md
│   └── Meta-Llama-3-8B-Instruct/
│       └── final_model/
│           ├── adapter_config.json
│           ├── adapter_model.bin (~50 MB)
│           └── README.md
│
└── predictions/
    ├── Mistral-7B-Instruct-v0.2_lora_predictions.json
    └── Meta-Llama-3-8B-Instruct_lora_predictions.json
```

### ICL Results

```
results/
└── icl/
    ├── Mistral-7B-Instruct-v0.2/
    │   └── 5shot_diverse/
    │       ├── predictions.json       # Same format as LoRA
    │       ├── config.yaml            # Exact config used
    │       └── metadata.json          # Run info
    └── Meta-Llama-3-8B-Instruct/
        └── 5shot_diverse/
            ├── predictions.json
            ├── config.yaml
            └── metadata.json
```

### Prediction Format (Both Methods)

```json
[
  {
    "id": "2951",
    "question": "what was the percentage change in total assets from 2016 to 2017?",
    "predicted_program": "divide(subtract(14280, 13292), 13292)",
    "predicted_answer": "0.0743",
    "gold_program": "divide(subtract(14280, 13292), 13292)",
    "gold_answer": "0.0743",
    "raw_output": "divide(subtract(14280, 13292), 13292)"
  }
]
```

### Metadata File (ICL Only)

```json
{
  "model": "mistralai/Mistral-7B-Instruct-v0.2",
  "num_shots": 5,
  "selection_method": "diverse",
  "temperature": 0.1,
  "num_predictions": 1147,
  "timestamp": "2025-11-30T17:45:23",
  "config_file": "../configs/icl_config_1.yaml"
}
```

---

## Command Reference

### Training

```bash
# LoRA training (manual)
cd src
python train_lora.py \
  --model_name mistralai/Mistral-7B-Instruct-v0.2 \
  --data_dir ../data/simplified \
  --output_dir ../results/lora \
  --epochs 3 \
  --batch_size 4 \
  --gradient_accumulation_steps 4 \
  --learning_rate 2e-4
```

### Inference

```bash
# LoRA inference (manual)
cd src
python inference.py \
  --model_name mistralai/Mistral-7B-Instruct-v0.2 \
  --adapter_path ../results/lora/Mistral-7B-Instruct-v0.2/final_model \
  --method lora \
  --data_dir ../data/simplified \
  --output_dir ../results/predictions \
  --temperature 0.1

# ICL inference (manual)
python icl_inference.py \
  --config ../configs/icl_config_1.yaml \
  --model_name mistralai/Mistral-7B-Instruct-v0.2 \
  --data_dir ../data/simplified \
  --output_dir ../results
```

### Evaluation

```bash
cd src

# Evaluate LoRA predictions
python evaluate.py \
  --predictions_file ../results/predictions/Mistral-7B-Instruct-v0.2_lora_predictions.json

# Evaluate ICL predictions
python evaluate.py \
  --predictions_file ../results/icl/Mistral-7B-Instruct-v0.2/5shot_diverse/predictions.json
```

**Output**:
```
Evaluation Results:
Total predictions: 1147
Execution accuracy: 0.4523 (correct numerical answer)
Program accuracy: 0.3891 (correct reasoning program)
```

---

## AWS Setup Guide

### Instance Selection

**Recommended**: g5.2xlarge
- GPU: NVIDIA A10G (24GB VRAM)
- vCPUs: 8
- RAM: 32 GB
- Cost: ~$1.20/hour (on-demand)
- Perfect for: LoRA training + inference

**Alternative**: g5.xlarge
- GPU: NVIDIA A10G (24GB VRAM)
- vCPUs: 4
- RAM: 16 GB
- Cost: ~$1.00/hour
- Works but slower (fewer CPUs for data loading)

### Setup Steps

1. **Launch Instance**
   ```bash
   # Use Deep Learning AMI (Ubuntu 20.04) - has CUDA pre-installed
   # Security group: Allow SSH (port 22) from your IP
   ```

2. **Connect to Instance**
   ```bash
   ssh -i your-key.pem ec2-user@your-instance-ip
   ```

3. **Clone Repository**
   ```bash
   git clone https://github.com/abhisvakil/FinQA-Mini-Project.git
   cd FinQA-Mini-Project
   ```

4. **Setup Environment**
   ```bash
   # Create virtual environment
   python3 -m venv venv
   source venv/bin/activate
   
   # Install dependencies
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

5. **Prepare Data**
   ```bash
   cd src
   python data_loader_simplified.py
   cd ..
   ```

6. **Run Experiments**
   ```bash
   # Make script executable
   chmod +x run_all_experiments.sh
   
   # Run with nohup (stays running after disconnect)
   source venv/bin/activate
   nohup ./run_all_experiments.sh train-lora-mistral > train.log 2>&1 &
   
   # Monitor
   tail -f train.log
   ```

### Cost Estimates

**g5.2xlarge ($1.20/hr)**:
- LoRA training (2 models): ~6 hours = $7.20
- Inference (all methods): ~2 hours = $2.40
- Total: ~8 hours = **$9.60**

**Tips to Save**:
1. Use Spot Instances (50-70% discount)
2. Stop instance when not in use
3. Use S3 for long-term storage (cheaper than EBS)
4. Set up billing alerts

---

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory (OOM)

**Symptoms**:
```
RuntimeError: CUDA out of memory. Tried to allocate 1.50 GiB
```

**Solutions**:
```bash
# For Llama training, use smaller batch size
# Edit run_all_experiments.sh or configs/lora_config.yaml
batch_size: 1
gradient_accumulation_steps: 16

# For inference, use 8-bit quantization
--load_in_8bit
```

#### 2. Module Not Found

**Symptoms**:
```
ModuleNotFoundError: No module named 'peft'
```

**Solution**:
```bash
# Make sure virtual environment is activated
source venv/bin/activate  # or conda activate finqa-mini

# Reinstall dependencies
pip install -r requirements.txt
```

#### 3. Permission Denied (AWS)

**Symptoms**:
```
./run_all_experiments.sh: Permission denied
```

**Solution**:
```bash
chmod +x run_all_experiments.sh
```

#### 4. Config File Not Found

**Symptoms**:
```
FileNotFoundError: [Errno 2] No such file or directory: '../configs/icl_config.yaml'
```

**Solution**:
```bash
# Use correct config file name
ls configs/  # Check available configs

# Update command to use existing config
python icl_inference.py --config ../configs/icl_config_1.yaml ...
```

#### 5. Slow Data Loading

**Symptoms**:
- Training stuck at "Loading data..."
- Takes >10 minutes to load dataset

**Solutions**:
```bash
# Use more data loader workers (in train_lora.py)
dataloader_num_workers: 4

# Or process data once and cache
cd src
python data_loader_simplified.py  # Creates cached version
```

### Debug Mode

Run scripts with verbose logging:

```bash
# Training
python train_lora.py --model_name ... --verbose

# Inference
python inference.py --model_name ... --verbose

# Check GPU usage
watch -n 1 nvidia-smi
```

---

## Implementation Details

### LoRA Configuration

```yaml
# configs/lora_config.yaml
lora:
  r: 8                      # Rank (bottleneck dimension)
  alpha: 16                 # Scaling factor (alpha/r = 2)
  target_modules:           # Which layers to adapt
    - q_proj                # Query projection
    - v_proj                # Value projection
    - k_proj                # Key projection
    - o_proj                # Output projection
  dropout: 0.05
  bias: none
  task_type: CAUSAL_LM

training:
  num_epochs: 3
  per_device_train_batch_size: 4    # Mistral: 4, Llama: 1
  gradient_accumulation_steps: 4     # Mistral: 4, Llama: 16
  learning_rate: 2e-4
  warmup_ratio: 0.03
  lr_scheduler_type: linear
  optim: paged_adamw_8bit           # Memory efficient optimizer
  save_strategy: epoch
  logging_steps: 50
```

**Why these settings?**
- r=8, alpha=16: Good balance of capacity and efficiency
- Target modules: Attention layers learn task patterns
- Dropout: Prevents overfitting
- Paged optimizer: Reduces memory usage

### ICL Configuration

```yaml
# configs/icl_config_1.yaml
model:
  model_name_or_path: "mistralai/Mistral-7B-Instruct-v0.2"
  torch_dtype: "bfloat16"
  device_map: "auto"
  load_in_8bit: false

generation:
  max_new_tokens: 256
  temperature: 0.1          # Low for deterministic output
  top_p: 0.95
  do_sample: true

icl:
  num_shots: 5              # Number of examples in prompt
  example_selection: diverse # diverse, random, or similarity

system_prompt: |
  You are a financial analyst assistant. Generate executable reasoning programs.
  
  Available Operations:
  - add(a, b), subtract(a, b), multiply(a, b)
  - divide(a, b), greater(a, b), exp(a, b)
```

**Few-shot Example Structure**:
```
Example 1:
Question: What was the percentage change?

Text:
- Total assets in 2016: $13,292 million
- Total assets in 2017: $14,280 million

Table:
| Year | Assets |
| 2016 | 13292  |
| 2017 | 14280  |

Program: divide(subtract(14280, 13292), 13292)
```

### Evaluation Metrics

**Execution Accuracy**:
- Execute predicted program
- Compare numerical result to gold answer
- Tolerance: ±0.001 for floating point

**Program Accuracy**:
- Exact string match after normalization
- Order of operations must match
- Variable names must match

**Code** (from `evaluate.py`):
```python
def execution_accuracy(pred_answer, gold_answer):
    """Check if predicted answer matches gold within tolerance."""
    try:
        pred_val = float(pred_answer)
        gold_val = float(gold_answer)
        return abs(pred_val - gold_val) < 0.001
    except:
        return pred_answer.strip() == gold_answer.strip()

def program_accuracy(pred_program, gold_program):
    """Check if predicted program matches gold exactly."""
    pred_norm = normalize_program(pred_program)
    gold_norm = normalize_program(gold_program)
    return pred_norm == gold_norm
```

---

## Best Practices

### Training
1. **Monitor GPU usage**: `watch -n 1 nvidia-smi`
2. **Check logs regularly**: `tail -f train.log`
3. **Save checkpoints**: Already configured (every epoch)
4. **Use gradient accumulation**: Simulate larger batches

### Inference
1. **Use low temperature** (0.1) for consistency
2. **Batch processing** for speed (if memory allows)
3. **Cache model** to avoid re-downloading
4. **Save raw outputs** for debugging

### Comparison
1. **Same test set** for all methods
2. **Same temperature** (0.1) for fairness
3. **Same base models** (no mixing versions)
4. **Multiple random seeds** for ICL (if needed)

---

## Additional Notes

### Model Caching

Models are cached in `~/.cache/huggingface/`:
- Llama-3-8B: ~15 GB
- Mistral-7B: ~14 GB

To change cache location:
```bash
export HF_HOME=/path/to/cache
export TRANSFORMERS_CACHE=/path/to/cache
```

### Memory Optimization

If hitting OOM:
1. Reduce batch size to 1
2. Increase gradient accumulation
3. Use 8-bit quantization (`load_in_8bit=True`)
4. Disable gradient checkpointing if not needed
5. Clear cache: `torch.cuda.empty_cache()`

### Extending to More Shots

To run ICL with different num_shots:

1. Copy config:
   ```bash
   cp configs/icl_config_1.yaml configs/icl_config_3shot.yaml
   ```

2. Edit `num_shots`:
   ```yaml
   icl:
     num_shots: 3  # or 10, 20, etc.
   ```

3. Run inference:
   ```bash
   cd src
   python icl_inference.py --config ../configs/icl_config_3shot.yaml ...
   ```

Results automatically saved to `results/icl/{model}/3shot_diverse/`

---

## Support & Resources

- **GitHub Issues**: Report bugs or ask questions
- **FinQA Paper**: https://arxiv.org/abs/2109.00122
- **LoRA Paper**: https://arxiv.org/abs/2106.09685
- **HuggingFace PEFT**: https://huggingface.co/docs/peft

---

**Last Updated**: November 30, 2025
