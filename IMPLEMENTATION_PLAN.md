# Implementation Plan: Explainable Quantitative Reasoning for Financial Reports using FinQA

## Project Overview

**Project Title**: Explainable Quantitative Reasoning for Financial Reports using FinQA

**Objective**: Build a model to automate the analysis of complex financial reports containing both unstructured text and structured tables. The model should answer complex numerical questions by generating executable, step-by-step reasoning programs. Compare a finetuned specialist model against a large, general-purpose ICL model to evaluate trade-offs between accuracy and reasoning quality.

---

## Phase 1: Environment Setup and Data Preparation

### 1.1 Environment Setup
- [ ] Create Python virtual environment (Python 3.8+)
- [ ] Install core dependencies:
  - PyTorch 1.7.1+
  - Hugging Face Transformers 4.4.2+
  - Additional libraries: pandas, numpy, json, tqdm, scikit-learn
- [ ] Clone/download FinQA repository and dataset
- [ ] Set up project directory structure

### 1.2 Dataset Acquisition and Exploration
- [ ] Download FinQA dataset from GitHub: https://github.com/czyssrs/FinQA/tree/main
- [ ] Explore dataset structure:
  - Load train.json, dev.json, test.json
  - Understand data format: `pre_text`, `post_text`, `table`, `qa` fields
  - Analyze question types and program structures
- [ ] Perform data statistics:
  - Number of examples per split
  - Distribution of question types
  - Average program length
  - Table/text characteristics

### 1.3 Data Preprocessing
- [ ] Parse JSON files and extract components
- [ ] Convert tables to structured format (pandas DataFrames)
- [ ] Tokenize text data (pre_text, post_text)
- [ ] Normalize numerical values in tables
- [ ] Create data loaders for training/inference
- [ ] Handle edge cases (missing values, malformed data)

---

## Phase 2: Baseline Implementation (FinQANet Architecture)

### 2.1 Retriever Module
- [ ] Implement retriever architecture:
  - Use RoBERTa-large as encoder
  - Create passage-level and table-cell-level representations
  - Implement relevance scoring mechanism
- [ ] Train retriever:
  - Prepare training data with positive/negative examples
  - Set up training loop with appropriate loss function
  - Monitor retrieval accuracy on dev set
- [ ] Inference:
  - Generate top-k relevant facts for each question
  - Save retrieval results for generator training

### 2.2 Generator Module
- [ ] Implement program generator:
  - Sequence-to-sequence architecture (T5/BART or similar)
  - Generate executable programs token by token
  - Handle program syntax: `add()`, `subtract()`, `divide()`, `multiply()`, `table()`, etc.
- [ ] Train generator:
  - Use retrieved facts as input context
  - Train on program generation task
  - Monitor program accuracy and execution accuracy
- [ ] Program execution:
  - Implement program executor
  - Validate program syntax
  - Execute programs and get final answers

### 2.3 Evaluation Framework
- [ ] Implement evaluation metrics:
  - **Execution Accuracy**: Percentage of questions where executed program result matches gold answer
  - **Program Accuracy**: Exact match of generated program with gold program
- [ ] Create evaluation script compatible with FinQA format
- [ ] Test on dev set and establish baseline performance

---

## Phase 3: Fine-tuned Specialist Model

### 3.1 Model Selection and Architecture
- [ ] Choose base model:
  - Option A: Fine-tune T5/BART for program generation
  - Option B: Fine-tune GPT-2/CodeT5 for financial reasoning
  - Option C: Custom architecture combining retriever + generator
- [ ] Design model architecture:
  - Input: Question + Retrieved context (text + table)
  - Output: Executable reasoning program
  - Incorporate table understanding capabilities

### 3.2 Training Strategy
- [ ] Prepare training data:
  - Combine question, retrieved facts, and gold program
  - Format as sequence-to-sequence task
  - Create data augmentation if needed
- [ ] Hyperparameter tuning:
  - Learning rate (1e-5 to 5e-4)
  - Batch size (8-32)
  - Number of epochs (3-10)
  - Warmup steps
- [ ] Training:
  - Train on train split
  - Validate on dev split
  - Early stopping based on dev performance
  - Save best checkpoint

### 3.3 Evaluation
- [ ] Run inference on test set
- [ ] Calculate execution accuracy and program accuracy
- [ ] Analyze error cases:
  - Program syntax errors
  - Incorrect reasoning steps
  - Wrong numerical values
- [ ] Generate qualitative examples

---

## Phase 4: Large General-Purpose ICL Model

### 4.1 Model Selection
- [ ] Choose ICL-capable model:
  - Option A: GPT-3.5/GPT-4 via API
  - Option B: Claude (Anthropic)
  - Option C: Open-source: Llama-2, Mistral, or similar
- [ ] Set up API access or local model loading

### 4.2 Prompt Engineering
- [ ] Design prompt template:
  - System message explaining the task
  - Few-shot examples (3-5 examples)
  - Format specification for program output
  - Instructions for handling tables and text
- [ ] Create few-shot examples:
  - Select diverse examples from training set
  - Cover different question types
  - Include both text and table-based reasoning
- [ ] Iterate on prompt design:
  - Test different prompt formats
  - Optimize example selection
  - Refine instructions

### 4.3 Inference Pipeline
- [ ] Implement batch inference:
  - Process questions in batches
  - Handle API rate limits (if using API)
  - Error handling and retries
- [ ] Post-processing:
  - Parse model output to extract program
  - Validate program format
  - Handle malformed outputs
- [ ] Execute programs and get answers

### 4.4 Evaluation
- [ ] Run inference on test set
- [ ] Calculate same metrics as specialist model
- [ ] Compare with baseline and specialist model
- [ ] Analyze reasoning quality differences

---

## Phase 5: Comparative Analysis

### 5.1 Quantitative Comparison
- [ ] Create comparison table:
  - Execution Accuracy
  - Program Accuracy
  - Inference time per example
  - Training time (if applicable)
  - Model size
  - Cost (if using API)
- [ ] Statistical significance testing
- [ ] Performance by question type

### 5.2 Qualitative Analysis
- [ ] Case study analysis:
  - Select 10-20 representative examples
  - Compare reasoning steps between models
  - Identify strengths/weaknesses of each approach
- [ ] Error analysis:
  - Categorize error types
  - Identify common failure modes
  - Analyze which model handles which cases better

### 5.3 Trade-off Analysis
- [ ] Accuracy vs. Cost:
  - Specialist model: Training cost, inference cost
  - ICL model: API costs or compute requirements
- [ ] Accuracy vs. Explainability:
  - Quality of reasoning steps
  - Interpretability of programs
  - User trust and understanding
- [ ] Scalability:
  - Ability to handle new question types
  - Fine-tuning requirements
  - Maintenance overhead

---

## Phase 6: Documentation and Reporting

### 6.1 Code Documentation
- [ ] Write docstrings for all functions/classes
- [ ] Create README with setup instructions
- [ ] Document model architectures
- [ ] Include example usage scripts

### 6.2 Results Documentation
- [ ] Create results report:
  - Executive summary
  - Methodology description
  - Results and metrics
  - Comparative analysis
  - Visualizations (accuracy plots, error distributions)
- [ ] Include example outputs:
  - Successful cases
  - Failure cases with analysis
  - Side-by-side comparisons

### 6.3 Final Deliverables
- [ ] Complete codebase with all models
- [ ] Trained model checkpoints (specialist model)
- [ ] Evaluation results and metrics
- [ ] Final report/documentation
- [ ] Presentation slides (if required)

---

## Project Structure

```
LLM_Mini_Project/
├── data/
│   ├── train.json
│   ├── dev.json
│   ├── test.json
│   └── processed/          # Preprocessed data
├── models/
│   ├── retriever/          # Retriever model code
│   ├── generator/          # Generator model code
│   ├── specialist/         # Fine-tuned specialist model
│   └── icl/                # ICL model implementation
├── src/
│   ├── data_loader.py
│   ├── preprocess.py
│   ├── executor.py         # Program executor
│   └── evaluate.py         # Evaluation metrics
├── notebooks/
│   ├── data_exploration.ipynb
│   ├── baseline_training.ipynb
│   └── analysis.ipynb
├── results/
│   ├── baseline/
│   ├── specialist/
│   └── icl/
├── configs/
│   ├── retriever_config.yaml
│   └── generator_config.yaml
├── requirements.txt
├── README.md
└── IMPLEMENTATION_PLAN.md
```

---

## Key Metrics to Track

1. **Execution Accuracy**: % of questions with correct final answer
2. **Program Accuracy**: % of questions with correct reasoning program
3. **Inference Time**: Average time per question
4. **Training Time**: Total training time (specialist model)
5. **Cost**: API costs or compute costs
6. **Reasoning Quality**: Qualitative assessment of program clarity

---

## Timeline Estimate

- **Phase 1** (Environment & Data): 2-3 days
- **Phase 2** (Baseline): 5-7 days
- **Phase 3** (Specialist Model): 4-5 days
- **Phase 4** (ICL Model): 3-4 days
- **Phase 5** (Analysis): 2-3 days
- **Phase 6** (Documentation): 2-3 days

**Total**: ~3-4 weeks

---

## Next Steps

1. Start with Phase 1: Set up environment and explore the dataset
2. Get familiar with FinQA baseline code structure
3. Begin implementing retriever module
4. Iterate and refine based on initial results

---

## References

- FinQA Dataset: https://github.com/czyssrs/FinQA
- FinQA Paper: https://arxiv.org/abs/2109.00122
- FinQA Leaderboard: https://finllm-leaderboard.readthedocs.io/

