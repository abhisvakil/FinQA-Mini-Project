# LangChain Tool Calling for LoRA Models

## Overview

This implementation replaces the feedback loop approach with proper LangChain tool calling. The model can now call tools (add, subtract, multiply, divide, etc.) incrementally in a multi-turn conversation instead of generating entire programs and retrying on errors.

## Installation

First, install the new dependencies:

```bash
pip install -r requirements.txt
```

This will install:
- `langchain>=0.1.0`
- `langchain-community>=0.0.20`
- `pydantic>=2.0.0`

## Usage

### Quick Test (10 samples)

```bash
# Edit config to set max_samples: 10
python src/lora_tool_calling_langchain.py --config configs/lora_tool_calling_config.yaml
```

### Full Test Set

Edit `configs/lora_tool_calling_config.yaml`:
```yaml
data:
  max_samples: null  # Process all samples
```

Then run:
```bash
python src/lora_tool_calling_langchain.py --config configs/lora_tool_calling_config.yaml
```

### Using Different Models

You can override model settings via command line:

```bash
# Use Mistral instead of Llama
python src/lora_tool_calling_langchain.py \
  --config configs/lora_tool_calling_config.yaml \
  --model_name "mistralai/Mistral-7B-Instruct-v0.2" \
  --adapter_path "results/lora/Mistral-7B-Instruct-v0.2/final_model"
```

## Configuration

Edit `configs/lora_tool_calling_config.yaml`:

```yaml
model:
  model_name_or_path: "meta-llama/Meta-Llama-3-8B-Instruct"
  adapter_path: "results/lora/Meta-Llama-3-8B-Instruct/final_model"
  torch_dtype: "bfloat16"
  device_map: "auto"
  load_in_8bit: false

agent:
  max_iterations: 10  # Max tool calls per question
  verbose: true        # Print agent actions

data:
  data_dir: "data/simplified"
  test_file: "test_simplified.json"
  output_dir: "results/predictions"
  max_samples: null    # Set to number for testing
```

## How It Works

1. **Model receives question + context** (table, text)
2. **LangChain agent decides which tool to call** (e.g., `add(5, 3)`)
3. **Tool executes** using EnhancedExecutor
4. **Result returned to agent** (e.g., `8`)
5. **Agent uses result for next step** or provides final answer
6. **Repeat** until final answer or max iterations

## Output Format

Predictions are saved in the same format as existing inference:

```json
{
  "id": "2951",
  "question": "what was the percentage change...",
  "predicted_answer": "0.0743",
  "gold_program": "divide(subtract(14280, 13292), 13292)",
  "gold_answer": "0.0743",
  "tool_calls": 3,
  "raw_output": "...",
  "intermediate_steps": "..."
}
```

## Available Tools

- `add(a, b)` - Add two numbers
- `subtract(a, b)` - Subtract two numbers
- `multiply(a, b)` - Multiply two numbers
- `divide(a, b)` - Divide two numbers
- `greater(a, b)` - Return maximum
- `exp(a, b)` - Exponentiation
- `const(value)` - Constant value
- `table_access(row, col)` - Access table cell

## Comparison with Feedback Loop

**Feedback Loop (Old):**
- Model generates entire program: `"subtract(750, 500), divide(#0, 500)"`
- Executor runs program
- If error → feedback → model retries
- Limited to retry-based correction

**Tool Calling (New):**
- Model calls: `add(750, 500)` → gets `1250`
- Model calls: `divide(1250, 500)` → gets `2.5`
- Model provides final answer: `2.5`
- Incremental execution with immediate feedback

## Troubleshooting

### Import Errors

If you get import errors for LangChain:
```bash
pip install --upgrade langchain langchain-community
```

### Model Loading Issues

Make sure adapter paths are correct:
```bash
ls results/lora/Meta-Llama-3-8B-Instruct/final_model/
```

### Memory Issues

If running out of memory, enable 8-bit loading:
```yaml
model:
  load_in_8bit: true
```

## Next Steps

1. Test with small sample set (10 samples)
2. Compare results with feedback loop approach
3. Evaluate on full test set
4. Analyze tool call patterns and success rates

