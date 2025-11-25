# FinQA Project: PEFT vs ICL Comparison

## Quick Reference Guide

### Project Goal
Compare **Parameter-Efficient Fine-Tuning (PEFT)** methods vs **In-Context Learning (ICL)** for financial reasoning using the FinQA dataset.

---

## Selected Models

### PEFT Models (for LoRA/QLoRA fine-tuning)

| Model | Size | Provider | HuggingFace Path | Use Case |
|-------|------|----------|------------------|----------|
| Llama-3-8B | 8B | Meta | `meta-llama/Meta-Llama-3-8B-Instruct` | Strong reasoning, popular |
| Mistral-7B | 7B | Mistral AI | `mistralai/Mistral-7B-Instruct-v0.2` | Efficient, high performance |

### ICL Models (for few-shot prompting)

| Model | Size | Provider | HuggingFace Path | Use Case |
|-------|------|----------|------------------|----------|
| Llama-3-70B | 70B | Meta | `meta-llama/Meta-Llama-3-70B-Instruct` | Large-scale reasoning |
| Qwen-2.5-72B | 72B | Alibaba | `Qwen/Qwen2.5-72B-Instruct` | Excellent math/reasoning |

---

## Three Approaches

### 1. LoRA (Low-Rank Adaptation) - 8B Models
- **What**: Fine-tune only small adapter layers
- **Pros**: Fast training, small model files (~50MB), good accuracy
- **Memory**: ~24GB GPU (RTX 3090/4090, A100)
- **Training time**: 2-4 hours
- **Models**: Llama-3-8B, Mistral-7B

### 2. QLoRA (Quantized LoRA) - 8B Models
- **What**: LoRA + 4-bit quantization
- **Pros**: 50% less memory, same accuracy as LoRA
- **Memory**: ~12GB GPU (RTX 3060, 4070)
- **Training time**: 3-5 hours (slightly slower)
- **Models**: Llama-3-8B, Mistral-7B

### 3. ICL (In-Context Learning) - 70B Models
- **What**: Few-shot prompting, no training
- **Pros**: No training needed, flexible, better reasoning
- **Memory**: ~40-48GB GPU (A100 80GB or multi-GPU) with 4-bit quantization
- **Inference time**: 3-5 seconds per example
- **Models**: Llama-3-70B, Qwen-2.5-72B

---

## Quick Start Commands

### 1. Setup Environment
```bash
# Create conda environment
conda env create -f environment.yml
conda activate finqa-mini

# Or use pip
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Explore Data
```bash
python src/data_loader.py
```

### 3. Train with LoRA
```bash
python src/peft_trainer.py \
  --model_name meta-llama/Meta-Llama-3-8B-Instruct \
  --method lora \
  --config configs/lora_config.yaml \
  --output_dir results/lora/llama3
```

### 4. Train with QLoRA
```bash
python src/peft_trainer.py \
  --model_name meta-llama/Meta-Llama-3-8B-Instruct \
  --method qlora \
  --config configs/qlora_config.yaml \
  --output_dir results/qlora/llama3
```

### 5. Run ICL Inference
```bash
python src/icl_inference.py \
  --model_name meta-llama/Meta-Llama-3-8B-Instruct \
  --config configs/icl_config.yaml \
  --output_dir results/icl/llama3
```

### 6. Evaluate Results
```bash
python src/evaluate.py \
  --predictions results/lora/llama3/predictions.json \
  --gold data/test.json
```

---

## Expected Results Structure

```
results/
├── lora/
│   ├── llama3/
│   │   ├── adapter_model.bin
│   │   ├── adapter_config.json
│   │   ├── predictions.json
│   │   └── metrics.json
│   └── mistral/
│       └── ...
├── qlora/
│   ├── llama3/
│   └── mistral/
└── icl/
    ├── llama3/
    └── mistral/
```

---

## Evaluation Metrics

1. **Execution Accuracy**: % of correct final answers
2. **Program Accuracy**: % of correct reasoning programs
3. **Training Time**: Hours to fine-tune (PEFT only)
4. **Inference Time**: Seconds per example
5. **Memory Usage**: Peak GPU memory (GB)
6. **Model Size**: Adapter size for PEFT (MB)

---

## Comparison Matrix Template

| Model | Method | Exec Acc | Prog Acc | Train Time | Inf Time | GPU Mem | Adapter Size |
|-------|--------|----------|----------|------------|----------|---------|--------------|
| Llama-3-8B | LoRA | ?% | ?% | ? hrs | ? s | ~24 GB | ~50 MB |
| Llama-3-8B | QLoRA | ?% | ?% | ? hrs | ? s | ~12 GB | ~50 MB |
| Mistral-7B | LoRA | ?% | ?% | ? hrs | ? s | ~24 GB | ~50 MB |
| Mistral-7B | QLoRA | ?% | ?% | ? hrs | ? s | ~12 GB | ~50 MB |
| Llama-3-70B | ICL | ?% | ?% | N/A | ? s | ~40 GB | N/A |
| Qwen-2.5-72B | ICL | ?% | ?% | N/A | ? s | ~48 GB | N/A |

---

## Key Files

### Source Code
- `src/data_loader.py` ✅ - Load FinQA dataset
- `src/executor.py` ✅ - Execute reasoning programs
- `src/evaluate.py` ✅ - Calculate metrics
- `src/peft_trainer.py` ⏳ - Train LoRA/QLoRA (to implement)
- `src/icl_inference.py` ⏳ - ICL inference (to implement)
- `src/model_utils.py` ⏳ - Model loading utilities (to implement)

### Configurations
- `configs/lora_config.yaml` ✅
- `configs/qlora_config.yaml` ✅
- `configs/icl_config.yaml` ✅

### Documentation
- `README.md` ✅ - Updated for PEFT/ICL
- `IMPLEMENTATION_PLAN_PEFT.md` ✅ - Detailed plan
- `requirements.txt` ✅ - Updated dependencies
- `environment.yml` ✅ - Conda environment

---

## Next Steps

### Immediate (Phase 2)
1. [ ] Implement `src/model_utils.py` - Model loading
2. [ ] Test model loading for chosen models
3. [ ] Run baseline zero-shot inference

### Short-term (Phase 3-4)
4. [ ] Implement `src/peft_trainer.py` - PEFT training
5. [ ] Format FinQA data for instruction tuning
6. [ ] Train LoRA on first model
7. [ ] Train QLoRA on first model

### Medium-term (Phase 5)
8. [ ] Implement `src/icl_inference.py` - ICL
9. [ ] Select few-shot examples
10. [ ] Run ICL on all models

### Final (Phase 6-7)
11. [ ] Collect all results
12. [ ] Generate comparison visualizations
13. [ ] Write analysis report
14. [ ] Create presentation

---

## Tips & Best Practices

### Memory Management
- Use QLoRA if GPU < 24GB
- Enable gradient checkpointing
- Reduce batch size if OOM
- Use `device_map="auto"` for multi-GPU

### Training Tips
- Start with small dataset sample (1000 examples) for debugging
- Monitor validation loss to detect overfitting
- Save checkpoints frequently
- Use learning rate warmup

### ICL Tips
- Start with 3-5 shot examples
- Use diverse examples (different question types)
- Keep examples concise
- Test different temperatures (0.1, 0.3, 0.7)

### Debugging
- Test on 10-100 examples first
- Check tokenizer output format
- Verify program parsing logic
- Test executor with known programs

---

## Troubleshooting

### CUDA Out of Memory
```bash
# Solution 1: Use QLoRA
# Solution 2: Reduce batch size to 1
# Solution 3: Enable gradient checkpointing
# Solution 4: Use 8-bit inference for ICL
```

### Model Download Issues
```bash
# Use HF token for gated models (Llama-3)
huggingface-cli login
```

### Slow Training
```bash
# Enable mixed precision training (bf16)
# Increase batch size with gradient accumulation
# Use faster optimizers (AdamW 8-bit)
```

---

## Resources

- **PEFT Library**: https://github.com/huggingface/peft
- **QLoRA Paper**: https://arxiv.org/abs/2305.14314
- **LoRA Paper**: https://arxiv.org/abs/2106.09685
- **FinQA Dataset**: https://github.com/czyssrs/FinQA
- **FinQA Paper**: https://arxiv.org/abs/2109.00122

---

## Contact & Collaboration

For questions or collaboration:
- Check GitHub issues
- Review documentation
- Test with small examples first
