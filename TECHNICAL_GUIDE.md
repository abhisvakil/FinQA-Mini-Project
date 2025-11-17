# Technical Implementation Guide: FinQA Project

## Detailed Technical Specifications

### 1. Data Format Understanding

#### FinQA Dataset Structure
```python
{
    "pre_text": ["sentence1", "sentence2", ...],  # Text before table
    "post_text": ["sentence1", "sentence2", ...], # Text after table
    "table": [
        ["header1", "header2", ...],              # Table headers
        ["row1_col1", "row1_col2", ...],          # Table rows
        ...
    ],
    "id": "unique_example_id",
    "qa": {
        "question": "What is the net income?",
        "program": ["add(", "revenue", "expenses", ")"],
        "gold_inds": [0, 1, 5, 10],              # Indices of supporting facts
        "exe_ans": "12345.67",                    # Execution result
        "program_re": {...}                       # Nested program format
    }
}
```

#### Program Operations
- `add(a, b)`: Addition
- `subtract(a, b)`: Subtraction
- `multiply(a, b)`: Multiplication
- `divide(a, b)`: Division
- `table(row, col)`: Access table cell
- `const(value)`: Constant value
- `max(a, b)`: Maximum
- `min(a, b)`: Minimum

---

### 2. Retriever Architecture

#### Model: RoBERTa-large based
```python
class Retriever(nn.Module):
    def __init__(self):
        self.encoder = RobertaModel.from_pretrained('roberta-large')
        self.scorer = nn.Linear(768, 1)
    
    def forward(self, question, passages):
        # Encode question
        q_emb = self.encoder(question)
        
        # Encode passages (text + table cells)
        p_emb = self.encoder(passages)
        
        # Score relevance
        scores = self.scorer(q_emb @ p_emb.T)
        return scores
```

#### Training Objective
- Positive examples: Gold supporting facts
- Negative examples: Random facts from same document
- Loss: Binary cross-entropy or contrastive loss

---

### 3. Generator Architecture

#### Option A: T5-based Generator
```python
class ProgramGenerator(nn.Module):
    def __init__(self):
        self.model = T5ForConditionalGeneration.from_pretrained('t5-base')
    
    def forward(self, question, context, program):
        # Format input: "question: {q} context: {c}"
        input_text = f"question: {question} context: {context}"
        
        # Generate program tokens
        outputs = self.model(
            input_ids=input_text,
            labels=program
        )
        return outputs
```

#### Option B: GPT-2/CodeT5 for Code Generation
- Better for structured program generation
- Can leverage code-specific pre-training

---

### 4. ICL Model Implementation

#### Prompt Template
```
You are a financial analyst. Given a financial report with text and tables, 
answer numerical questions by generating an executable reasoning program.

Program Format:
- Use operations: add(), subtract(), multiply(), divide()
- Access table cells: table(row, col)
- Use constants: const(value)

Examples:

Question: What is the total revenue?
Context: 
  Text: "The company reported revenue of $1000."
  Table: [["Year", "Revenue"], ["2020", "1000"], ["2021", "1200"]]
Program: add(const(1000), table(1, 1))
Answer: 2200

[Few more examples...]

Now answer:
Question: {question}
Context: {context}
Program:
```

#### API Usage (GPT-3.5/4)
```python
import openai

def generate_program_icl(question, context, examples):
    prompt = build_prompt(question, context, examples)
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1  # Low temperature for deterministic output
    )
    
    return parse_program(response.choices[0].message.content)
```

---

### 5. Program Executor

```python
class ProgramExecutor:
    def __init__(self, table, pre_text, post_text):
        self.table = table
        self.text = pre_text + post_text
        self.variables = {}
    
    def execute(self, program_tokens):
        # Parse program tokens into AST
        ast = self.parse(program_tokens)
        
        # Execute AST
        result = self.eval(ast)
        return result
    
    def parse(self, tokens):
        # Convert token sequence to abstract syntax tree
        # Handle nested operations
        pass
    
    def eval(self, node):
        if node.type == 'add':
            return self.eval(node.left) + self.eval(node.right)
        elif node.type == 'table':
            row, col = node.row, node.col
            return float(self.table[row][col])
        elif node.type == 'const':
            return float(node.value)
        # ... other operations
```

---

### 6. Evaluation Metrics

#### Execution Accuracy
```python
def execution_accuracy(predictions, gold_answers):
    correct = 0
    for pred, gold in zip(predictions, gold_answers):
        if abs(float(pred) - float(gold)) < 1e-6:
            correct += 1
    return correct / len(predictions)
```

#### Program Accuracy
```python
def program_accuracy(predicted_programs, gold_programs):
    exact_matches = 0
    for pred, gold in zip(predicted_programs, gold_programs):
        if pred == gold:
            exact_matches += 1
    return exact_matches / len(predicted_programs)
```

---

### 7. Training Configuration

#### Retriever Training
```yaml
model: roberta-large
batch_size: 16
learning_rate: 2e-5
epochs: 10
warmup_steps: 500
max_seq_length: 512
top_k: 5  # Retrieve top 5 facts
```

#### Generator Training
```yaml
model: t5-base  # or t5-large
batch_size: 8
learning_rate: 5e-5
epochs: 15
warmup_steps: 1000
max_input_length: 1024
max_output_length: 256
beam_size: 5  # For inference
```

---

### 8. Implementation Checklist

#### Week 1: Setup & Baseline
- [ ] Clone FinQA repository
- [ ] Set up environment
- [ ] Load and explore dataset
- [ ] Implement data preprocessing pipeline
- [ ] Set up retriever training infrastructure
- [ ] Train baseline retriever

#### Week 2: Generator & Specialist Model
- [ ] Implement program generator
- [ ] Train baseline generator
- [ ] Implement program executor
- [ ] Evaluate baseline end-to-end
- [ ] Fine-tune specialist model (T5/BART)
- [ ] Optimize hyperparameters

#### Week 3: ICL Model & Comparison
- [ ] Set up ICL model (API or local)
- [ ] Design and test prompts
- [ ] Run ICL inference on test set
- [ ] Implement comparison framework
- [ ] Generate quantitative results

#### Week 4: Analysis & Documentation
- [ ] Error analysis
- [ ] Qualitative case studies
- [ ] Create visualizations
- [ ] Write final report
- [ ] Prepare presentation

---

### 9. Common Challenges & Solutions

#### Challenge 1: Table Representation
**Problem**: Tables need to be converted to text format for language models
**Solution**: 
- Convert to linearized format: "row 0: header1, header2 | row 1: val1, val2"
- Or use special tokens: `<table>` ... `</table>`

#### Challenge 2: Program Syntax Errors
**Problem**: Generated programs may have syntax errors
**Solution**:
- Use constrained decoding
- Post-process to fix common errors
- Validate before execution

#### Challenge 3: Numerical Precision
**Problem**: Floating point comparisons
**Solution**:
- Use tolerance-based comparison (1e-6)
- Normalize numerical formats

#### Challenge 4: Long Context
**Problem**: Financial reports can be very long
**Solution**:
- Truncate or chunk long documents
- Use sliding window approach
- Focus on retrieved relevant parts

---

### 10. Useful Libraries

```python
# Core ML
import torch
import transformers
from transformers import (
    RobertaModel, RobertaTokenizer,
    T5ForConditionalGeneration, T5Tokenizer,
    GPT2LMHeadModel, GPT2Tokenizer
)

# Data Processing
import pandas as pd
import numpy as np
import json

# Evaluation
from sklearn.metrics import accuracy_score
import re

# Utilities
from tqdm import tqdm
import logging
```

---

### 11. Testing Strategy

#### Unit Tests
- Test program executor with known programs
- Test data loading and preprocessing
- Test evaluation metrics

#### Integration Tests
- Test retriever + generator pipeline
- Test end-to-end inference
- Validate output format

#### Validation
- Monitor dev set performance during training
- Early stopping to prevent overfitting
- Cross-validation if dataset allows

---

## Quick Start Commands

```bash
# Setup
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install torch transformers pandas numpy tqdm

# Download dataset
git clone https://github.com/czyssrs/FinQA.git
cd FinQA

# Train retriever
cd retriever
python train.py

# Train generator
cd ../generator
python train.py

# Evaluate
python evaluate.py predictions.json test.json
```

---

## Next Steps

1. Review this technical guide
2. Set up development environment
3. Start with data exploration notebook
4. Implement retriever module first
5. Iterate based on results

