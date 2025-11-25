# Implementation Plan: PEFT vs ICL for Financial Reasoning

## Project Overview

**Project Title**: Parameter-Efficient Fine-Tuning vs In-Context Learning for Financial Report Analysis

**Objective**: Compare PEFT methods (LoRA/QLoRA) against In-Context Learning (ICL) for answering numerical questions on financial reports. Use 2-4 open-source models to evaluate trade-offs between fine-tuning efficiency, accuracy, and zero-shot reasoning capabilities.

**Models for PEFT (LoRA/QLoRA)**:
1. **Llama-3-8B** (Meta) - `meta-llama/Meta-Llama-3-8B-Instruct` - Strong reasoning, 8B params
2. **Mistral-7B** (Mistral AI) - `mistralai/Mistral-7B-Instruct-v0.2` - Efficient, 7B params

**Models for ICL (Few-shot)**:
1. **Llama-3-70B** (Meta) - `meta-llama/Meta-Llama-3-70B-Instruct` - Large-scale reasoning, 70B params
2. **Qwen-2.5-72B** (Alibaba) - `Qwen/Qwen2.5-72B-Instruct` - Excellent math/reasoning, 72B params

**Methods**:
- **LoRA**: Low-rank adapter training (efficient fine-tuning) - 8B models
- **QLoRA**: 4-bit quantized LoRA (even more memory efficient) - 8B models
- **ICL**: Few-shot prompting (no fine-tuning) - 70B models

---

## Phase 1: Environment Setup and Data Preparation ✅

### 1.1 Environment Setup
- [x] Create Python virtual environment (Python 3.10+)
- [x] Install core dependencies:
  - PyTorch 1.7.1+
  - Hugging Face Transformers 4.4.2+
  - PEFT library for LoRA/QLoRA
  - bitsandbytes for quantization
  - accelerate for distributed training
- [x] Download FinQA dataset
- [x] Set up project directory structure
- [x] Implement data loader utilities

### 1.2 Dataset Preparation
- [ ] Prepare training data format for PEFT:
  - Input: Question + Context (text + table)
  - Output: Reasoning program
  - Format as instruction-tuning examples
- [ ] Create data preprocessing pipeline:
  - Linearize tables for LLM input
  - Combine pre_text and post_text appropriately
  - Format programs as executable code
- [ ] Prepare ICL prompt templates with few-shot examples
- [ ] Select diverse few-shot examples (3-10 examples)
- [ ] Optional: Sample subset for faster iteration during development

---

## Phase 2: Model Selection and Setup

### 2.1 Select Open-Source Models

**PEFT Models (8B - for LoRA/QLoRA fine-tuning):**
```python
peft_models = {
    "llama-3-8b": "meta-llama/Meta-Llama-3-8B-Instruct",
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.2"
}
```

**ICL Models (70B+ - for few-shot prompting):**
```python
icl_models = {
    "llama-3-70b": "meta-llama/Meta-Llama-3-70B-Instruct",
    "qwen-2.5-72b": "Qwen/Qwen2.5-72B-Instruct"
}
```

**Rationale:**
- 8B models: Optimal for efficient fine-tuning, fit in 24GB GPU for LoRA
- 70B models: Superior reasoning for ICL, can run quantized in 40-48GB GPU

### 2.2 Model Setup
- [ ] Create model loading utilities (`src/model_utils.py`)
- [ ] Download models from Hugging Face Hub
- [ ] Test model loading and basic inference
- [ ] Verify tokenizer compatibility with program format
- [ ] Test memory requirements (GPU/CPU)

### 2.3 Baseline Evaluation
- [ ] Run zero-shot inference on sample dev set
- [ ] Establish baseline metrics (execution accuracy)
- [ ] Identify model-specific quirks:
  - Prompt format requirements
  - Special tokens
  - Output parsing challenges

---

## Phase 3: LoRA Fine-tuning

### 3.1 LoRA Configuration
- [ ] Create LoRA config file (`configs/lora_config.yaml`):
  ```yaml
  lora:
    r: 8  # Rank (try 4, 8, 16)
    lora_alpha: 16
    target_modules:
      - q_proj
      - v_proj
      - k_proj  # Optional
      - o_proj  # Optional
    lora_dropout: 0.05
    bias: "none"
    task_type: "CAUSAL_LM"
  
  training:
    learning_rate: 2e-4
    batch_size: 4
    gradient_accumulation_steps: 4
    num_epochs: 3
    warmup_steps: 100
    max_seq_length: 2048
    save_steps: 500
    eval_steps: 250
  ```

### 3.2 Data Formatting for LoRA
- [ ] Format data as instruction-tuning examples:
  ```
  ### Instruction:
  Answer the following financial question by generating an executable reasoning program.
  Use operations: add(), subtract(), multiply(), divide(), table(row, col), const(value)
  
  ### Input:
  Question: [question]
  Context: [pre_text + table + post_text]
  
  ### Output:
  [program tokens]
  ```
- [ ] Create train/validation data loaders
- [ ] Implement data collator for batching

### 3.3 Training Implementation
- [ ] Implement training script (`src/peft_trainer.py`):
  - Load base model
  - Apply LoRA config with PEFT
  - Set up SFTTrainer or custom training loop
  - Monitor loss and validation metrics
  - Save checkpoints and adapters
- [ ] Train each selected model with LoRA
- [ ] Track training metrics (loss, learning rate, GPU memory)
- [ ] Save final LoRA adapters

### 3.4 LoRA Inference and Evaluation
- [ ] Load base model + LoRA adapters
- [ ] Run inference on dev and test sets
- [ ] Parse model outputs to extract programs
- [ ] Execute programs using executor
- [ ] Calculate metrics:
  - Execution accuracy
  - Program accuracy
  - Inference time per example
- [ ] Save results to `results/lora/`

---

## Phase 4: QLoRA Fine-tuning

### 4.1 QLoRA Configuration
- [ ] Create QLoRA config file (`configs/qlora_config.yaml`):
  ```yaml
  quantization:
    load_in_4bit: true
    bnb_4bit_quant_type: "nf4"
    bnb_4bit_compute_dtype: "bfloat16"
    bnb_4bit_use_double_quant: true
  
  lora:
    r: 8
    lora_alpha: 16
    target_modules:
      - q_proj
      - v_proj
    lora_dropout: 0.05
    bias: "none"
  
  training:
    learning_rate: 2e-4
    batch_size: 4
    gradient_accumulation_steps: 4
    num_epochs: 3
    warmup_steps: 100
  ```

### 4.2 QLoRA Training
- [ ] Implement 4-bit quantization with bitsandbytes
- [ ] Load models in 4-bit precision
- [ ] Apply LoRA to quantized model
- [ ] Train with same data as LoRA
- [ ] Monitor memory usage (should be ~50% of LoRA)
- [ ] Save QLoRA adapters

### 4.3 QLoRA Inference and Evaluation
- [ ] Load quantized model + QLoRA adapters
- [ ] Run inference on test set
- [ ] Compare with LoRA results:
  - Accuracy difference
  - Speed difference
  - Memory usage
- [ ] Save results to `results/qlora/`

---

## Phase 5: In-Context Learning (ICL)

### 5.1 Prompt Engineering
- [ ] Design ICL prompt template (`configs/icl_config.yaml`):
  ```yaml
  system_prompt: |
    You are a financial analyst. Answer numerical questions by generating 
    executable reasoning programs using these operations:
    - add(a, b), subtract(a, b), multiply(a, b), divide(a, b)
    - table(row, col): access table cell
    - const(value): constant value
  
  few_shot_examples: 3-5
  temperature: 0.1
  max_tokens: 256
  ```

### 5.2 Few-Shot Example Selection
- [ ] Select diverse examples from training set:
  - Different question types (percentage, ratio, growth, etc.)
  - Mix of table-only and text+table examples
  - Varying program complexity (simple to multi-step)
- [ ] Format few-shot examples clearly:
  ```
  Question: What was the revenue growth?
  Context: [text + table]
  Program: divide(subtract(table(2,1), table(1,1)), table(1,1))
  ```
- [ ] Test 3, 5, and 10-shot configurations

### 5.3 ICL Implementation
- [ ] Implement ICL inference script (`src/icl_inference.py`):
  - Load model (no fine-tuning)
  - Build prompt with few-shot examples
  - Generate program predictions
  - Parse outputs
- [ ] Run ICL on each selected model
- [ ] Experiment with:
  - Number of examples (3, 5, 10)
  - Example selection strategies
  - Temperature and sampling parameters

### 5.4 ICL Evaluation
- [ ] Run inference on test set
- [ ] Calculate same metrics as PEFT
- [ ] Save results to `results/icl/`
- [ ] Analyze prompt sensitivity

---

## Phase 6: Comparative Analysis

### 6.1 Quantitative Comparison
- [ ] Create comprehensive comparison table:
  ```
  | Model      | Method | Exec Acc | Prog Acc | Inf Time | GPU Mem | Training Time |
  |------------|--------|----------|----------|----------|---------|---------------|
  | Llama-3-8B | LoRA   | XX.X%    | XX.X%    | XX ms    | XX GB   | XX min        |
  | Llama-3-8B | QLoRA  | XX.X%    | XX.X%    | XX ms    | XX GB   | XX min        |
  | Llama-3-8B | ICL    | XX.X%    | XX.X%    | XX ms    | XX GB   | N/A           |
  | ...        | ...    | ...      | ...      | ...      | ...     | ...           |
  ```

- [ ] Generate visualizations:
  - Accuracy comparison bar charts
  - Efficiency scatter plots (accuracy vs time/memory)
  - Training curves for LoRA/QLoRA
  - Error rate by question type

### 6.2 Qualitative Analysis
- [ ] Case study analysis (20-30 examples):
  - Compare predictions across methods
  - Identify where each method excels
  - Analyze failure modes
- [ ] Error categorization:
  - Syntax errors
  - Semantic errors (wrong reasoning)
  - Arithmetic errors
  - Formatting issues
- [ ] Reasoning quality assessment:
  - Program interpretability
  - Logical coherence
  - Unnecessary complexity

### 6.3 Trade-off Analysis
- [ ] **Accuracy vs Efficiency**:
  - Training time cost
  - Inference speed
  - Memory requirements
  - Deployability

- [ ] **PEFT vs ICL Trade-offs**:
  - Task adaptation (fine-tuned vs prompted)
  - Data requirements
  - Maintenance and updates
  - Generalization to new question types

- [ ] **LoRA vs QLoRA**:
  - Accuracy gap
  - Memory savings
  - Training speed
  - Best use cases

### 6.4 Statistical Analysis
- [ ] Perform significance testing (t-test, Wilcoxon)
- [ ] Calculate confidence intervals
- [ ] Analyze variance across question types
- [ ] Identify statistically significant differences

---

## Phase 7: Documentation and Reporting

### 7.1 Code Documentation
- [ ] Add docstrings to all functions
- [ ] Create usage examples for each script
- [ ] Write configuration guide
- [ ] Document model-specific notes

### 7.2 Results Documentation
- [ ] Create final report with:
  - Executive summary
  - Methodology description
  - Results and metrics
  - Comparative analysis
  - Visualizations
  - Conclusions and recommendations
- [ ] Include example outputs:
  - Successful predictions
  - Failure case analysis
  - Side-by-side method comparisons

### 7.3 Reproducibility
- [ ] Document exact model versions
- [ ] Provide training commands
- [ ] Share config files
- [ ] List hardware specifications
- [ ] Document random seeds

---

## Project Timeline

- **Phase 1** (Environment & Data): 1-2 days ✅
- **Phase 2** (Model Setup): 1-2 days
- **Phase 3** (LoRA): 2-3 days
- **Phase 4** (QLoRA): 2-3 days
- **Phase 5** (ICL): 1-2 days
- **Phase 6** (Analysis): 2-3 days
- **Phase 7** (Documentation): 1-2 days

**Total**: ~2-3 weeks

---

## Key Files to Create

```
src/
├── model_utils.py         # Model loading utilities
├── peft_trainer.py        # LoRA/QLoRA training script
├── icl_inference.py       # ICL inference script
├── prompt_templates.py    # ICL prompt templates
└── analysis.py            # Comparison and analysis

configs/
├── lora_config.yaml       # LoRA hyperparameters
├── qlora_config.yaml      # QLoRA hyperparameters
└── icl_config.yaml        # ICL prompt config

notebooks/
├── data_exploration.ipynb
├── model_comparison.ipynb
└── error_analysis.ipynb
```

---

## Expected Outcomes

1. **Quantitative Results**: Accuracy, efficiency metrics for each method
2. **Qualitative Insights**: Understanding of trade-offs between methods
3. **Best Practices**: Guidelines for choosing PEFT vs ICL
4. **Model Recommendations**: Which models work best for financial reasoning
5. **Reproducible Pipeline**: Complete codebase for future experiments

---

## Next Steps

1. ✅ Environment setup complete
2. Create model loading utilities
3. Prepare data for PEFT training
4. Implement LoRA training pipeline
5. Run experiments and iterate

---

## References

- PEFT Library: https://github.com/huggingface/peft
- QLoRA Paper: https://arxiv.org/abs/2305.14314
- LoRA Paper: https://arxiv.org/abs/2106.09685
- FinQA Dataset: https://github.com/czyssrs/FinQA
- FinQA Paper: https://arxiv.org/abs/2109.00122
