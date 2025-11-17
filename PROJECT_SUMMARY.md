# Project Summary: Explainable Quantitative Reasoning for Financial Reports

## Overview

This project implements and compares two approaches for automated financial report analysis:

1. **Fine-tuned Specialist Model**: A model specifically trained on the FinQA dataset
2. **Large General-Purpose ICL Model**: A large language model using in-context learning

Both models answer numerical questions by generating executable reasoning programs.

## Key Deliverables

### 1. Implementation Plan (`IMPLEMENTATION_PLAN.md`)
- Comprehensive 6-phase implementation roadmap
- Detailed task breakdown with checkboxes
- Timeline estimates (~3-4 weeks)
- Project structure and organization

### 2. Technical Guide (`TECHNICAL_GUIDE.md`)
- Detailed technical specifications
- Architecture designs for retriever and generator
- Code examples and implementation patterns
- Common challenges and solutions

### 3. Starter Code
- **`src/data_loader.py`**: Dataset loading and parsing utilities
- **`src/executor.py`**: Program execution engine
- **`src/evaluate.py`**: Evaluation metrics (execution & program accuracy)

### 4. Project Infrastructure
- **`requirements.txt`**: All necessary dependencies
- **`setup.sh`**: Automated setup script
- **`README.md`**: Project documentation
- **`.gitignore`**: Git ignore rules

## Quick Start

### Step 1: Setup Environment
```bash
# Run setup script
./setup.sh

# Or manually:
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Step 2: Download Dataset
```bash
# Clone FinQA repository
git clone https://github.com/czyssrs/FinQA.git

# Copy dataset files
cp FinQA/dataset/*.json data/
```

### Step 3: Explore Data
```python
from src.data_loader import FinQADataLoader

loader = FinQADataLoader(data_dir="data")
train_data = loader.load_split("train")
stats = loader.get_statistics(train_data)
print(stats)
```

### Step 4: Test Executor
```python
from src.executor import ProgramExecutor

table = [["Year", "Revenue"], ["2020", "1000"]]
executor = ProgramExecutor(table)
result = executor.execute(['add(', 'const(5)', 'const(3)', ')'])
print(result)  # Should output 8.0
```

## Implementation Phases

### Phase 1: Setup & Data (Days 1-3)
- ✅ Environment setup
- ✅ Dataset download and exploration
- ✅ Data preprocessing pipeline

### Phase 2: Baseline (Days 4-10)
- [ ] Implement retriever module
- [ ] Implement generator module
- [ ] Train baseline model
- [ ] Establish baseline metrics

### Phase 3: Specialist Model (Days 11-15)
- [ ] Fine-tune T5/BART on FinQA
- [ ] Hyperparameter optimization
- [ ] Evaluate performance

### Phase 4: ICL Model (Days 16-19)
- [ ] Set up GPT-3.5/4 or open-source LLM
- [ ] Design and optimize prompts
- [ ] Run inference on test set

### Phase 5: Comparison (Days 20-22)
- [ ] Quantitative comparison
- [ ] Qualitative analysis
- [ ] Trade-off evaluation

### Phase 6: Documentation (Days 23-25)
- [ ] Write final report
- [ ] Create visualizations
- [ ] Prepare presentation

## Evaluation Metrics

1. **Execution Accuracy**: % of questions with correct final answer
   - Target: >60% (baseline FinQANet: ~61%)

2. **Program Accuracy**: % of questions with correct reasoning program
   - Target: >55% (baseline FinQANet: ~59%)

3. **Inference Time**: Average time per question
   - Specialist: ~100-500ms
   - ICL: ~1-5s (API) or ~500ms-2s (local)

4. **Reasoning Quality**: Qualitative assessment
   - Clarity of reasoning steps
   - Correctness of intermediate calculations
   - Interpretability

## Expected Outcomes

### Specialist Model
- **Pros**: 
  - Higher accuracy on FinQA dataset
  - Faster inference
  - Lower cost (after training)
- **Cons**:
  - Requires training data
  - Less generalizable
  - Training time and compute

### ICL Model
- **Pros**:
  - No training required
  - More generalizable
  - Easy to update with new examples
- **Cons**:
  - Potentially lower accuracy
  - Slower inference (API latency)
  - Higher cost (API usage)
  - Less control over model behavior

## Key Files Reference

| File | Purpose |
|------|---------|
| `IMPLEMENTATION_PLAN.md` | Step-by-step implementation guide |
| `TECHNICAL_GUIDE.md` | Technical specifications and code patterns |
| `src/data_loader.py` | Dataset loading utilities |
| `src/executor.py` | Program execution engine |
| `src/evaluate.py` | Evaluation metrics |
| `README.md` | Project overview and setup |
| `requirements.txt` | Python dependencies |

## Next Steps

1. **Review Documentation**: Read `IMPLEMENTATION_PLAN.md` and `TECHNICAL_GUIDE.md`
2. **Set Up Environment**: Run `./setup.sh` or follow manual setup
3. **Download Dataset**: Clone FinQA repo and copy dataset files
4. **Explore Data**: Use `src/data_loader.py` to understand the dataset
5. **Start Implementation**: Begin with Phase 1 tasks

## Resources

- **FinQA Repository**: https://github.com/czyssrs/FinQA
- **FinQA Paper**: https://arxiv.org/abs/2109.00122
- **FinQA Leaderboard**: https://finllm-leaderboard.readthedocs.io/
- **Hugging Face Transformers**: https://huggingface.co/transformers/

## Support

For questions or issues:
1. Check the documentation files
2. Review FinQA repository README
3. Consult the technical guide for implementation details

---

**Good luck with your mini-project! 🚀**

