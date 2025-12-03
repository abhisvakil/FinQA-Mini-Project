"""
Unified In-Context Learning (ICL) inference script for FinQA.
Uses YAML config and produces predictions in the same format as LoRA inference.
"""

import os
import sys
import json
import yaml
import torch
import random
import argparse
from typing import List, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# Configure tqdm for nohup and terminal output
tqdm.monitor_interval = 0

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def select_few_shot_examples(train_data: List[Dict], num_examples: int = 5, 
                             selection_method: str = "diverse") -> List[Dict]:
    """
    Select few-shot examples from training data.
    
    Args:
        train_data: Training examples
        num_examples: Number of examples to select
        selection_method: "random" or "diverse"
        
    Returns:
        Selected examples
    """
    if selection_method == "random":
        return random.sample(train_data, min(num_examples, len(train_data)))
    
    elif selection_method == "diverse":
        # Select diverse examples based on program length
        examples = []
        sorted_by_program = sorted(train_data, key=lambda x: len(x.get('program', [])))
        
        n = len(sorted_by_program)
        indices = [n // 4, n // 2, 3 * n // 4]
        
        for idx in indices:
            if idx < len(sorted_by_program):
                examples.append(sorted_by_program[idx])
        
        # Fill remaining with random
        remaining = [ex for ex in train_data if ex not in examples]
        if remaining and len(examples) < num_examples:
            examples.extend(random.sample(remaining, min(num_examples - len(examples), len(remaining))))
        
        return examples[:num_examples]
    
    return train_data[:num_examples]


def finqa_to_yaml_examples(selected: List[Dict]) -> List[Dict]:
    """
    Convert FinQA training examples to format expected by prompt template.
    Includes both program and answer.
    """
    out = []
    for ex in selected:
        # Build context with pre_text and table
        context_str = ""
        
        # Add pre_text if present
        if ex.get("pre_text"):
            context_str += " ".join(ex["pre_text"][:10]) + "\n"
        
        # Add table as markdown string
        if ex.get("table"):
            context_str += "Table:\n"
            header = " | ".join(ex["table"][0])
            context_str += f"| {header} |\n"
            for row in ex["table"][1:]:
                row_str = " | ".join(str(cell) for cell in row)
                context_str += f"| {row_str} |\n"
        
        # Extract program and answer
        program = " ".join(ex['program']) if isinstance(ex.get('program'), list) else ex.get('program', '')
        
        # Remove const_ prefixes from program to match desired format
        import re
        program = re.sub(r'const_(\d+\.?\d*)', r'\1', program)
        program = re.sub(r'const_', '', program)
        
        answer = str(ex.get('answer', ''))
        
        out.append({
            "question": ex["question"],
            "context": context_str.strip(),
            "program": program,
            "answer": answer
        })
    return out


def format_few_shot_examples(few_shots: List[Dict]) -> str:
    """
    Format few-shot examples for inclusion in prompt template.
    Includes both Program and Answer fields.
    """
    formatted = []
    for ex in few_shots:
        # Include both program and answer
        answer_part = f"\nAnswer: {ex.get('answer', '')}" if ex.get('answer') else ""
        formatted.append(
            f"Question: {ex['question']}\n{ex['context']}\nProgram: {ex['program']}{answer_part}"
        )
    return "\n\n".join(formatted)


def create_prompt_from_config(config: dict, example: dict) -> str:
    """
    Create prompt using config's prompt_template.
    This ensures the prompt format matches exactly what's in the config.
    """
    template = config["prompt_template"]
    system_prompt = config["system_prompt"]
    few_shot_block = format_few_shot_examples(config.get("few_shot_examples", []))

    # Compose context from example fields (FinQA specific)
    context = ""
    if example.get("pre_text"):
        context += " ".join(example.get("pre_text", [])[:10]) + "\n"
    if example.get("post_text"):
        context += " ".join(example.get("post_text", [])[:10]) + "\n"
    if example.get("table"):
        context += "Table:\n"
        table = example["table"]
        if len(table) > 0:
            header = " | ".join(table[0])
            context += f"| {header} |\n"
            for row in table[1:]:
                row_str = " | ".join(str(cell) for cell in row)
                context += f"| {row_str} |\n"
    context = context.strip()
    return template.format(
        system_prompt=system_prompt,
        few_shot_examples=few_shot_block,
        question=example["question"],
        context=context,
        )


def parse_model_output(output: str) -> tuple:
    """
    Parse model output to extract program and answer.
    Handles various edge cases like extra text, spacing issues, etc.
    
    Args:
        output: Raw model output
        
    Returns:
        (program, answer) tuple
    """
    import re
    
    program = ""
    answer = ""
    
    # Clean up output
    output = output.strip()
    
    # Strategy 1: Look for "Program:" and "Answer:" markers
    if "Program:" in output:
        program_part = output.split("Program:")[1]
        if "Answer:" in program_part:
            # Extract program (everything before Answer:)
            program = program_part.split("Answer:")[0].strip()
            # Extract answer (everything after Answer:, but clean it up)
            answer_part = program_part.split("Answer:")[1].strip()
            
            # Clean answer: take first line, remove extra text
            answer = answer_part.split('\n')[0].strip()
            # Remove common artifacts like "assistant", "|", etc.
            answer = answer.split('assistant')[0].split('Assistant')[0].split('|')[0].strip()
            # Remove any trailing punctuation that shouldn't be there
            answer = answer.rstrip('.,;:!?')
        else:
            # Only program, no explicit answer - try to extract from next lines
            program = program_part.split('\n')[0].strip()
    
    # Strategy 2: Look for program-like patterns at the start of lines
    if not program:
        # Match lines that start with operation names (add, subtract, etc.)
        lines = output.strip().split('\n')
        for line in lines:
            line = line.strip()
            # Check if line starts with a valid operation
            if re.match(r'^(add|subtract|multiply|divide|greater|exp)\(', line):
                program = line
                break
        
        # If still no program, try to find after "### Final" marker
        if not program and "### Final" in output:
            after_final = output.split("### Final")[1].strip()
            lines = after_final.split('\n')
            for line in lines:
                line = line.strip()
                if re.match(r'^(add|subtract|multiply|divide|greater|exp)\(', line):
                    program = line
                    break
    
    # Strategy 3: Extract answer from "Answer:" marker (case-insensitive)
    if not answer:
        # Look for Answer: marker (case-insensitive)
        answer_match = re.search(r'Answer:\s*([^\n]+)', output, re.IGNORECASE)
        if answer_match:
            answer = answer_match.group(1).strip()
            # Remove trailing explanation text
            answer = answer.split('This')[0].split('Therefore')[0].split('So')[0].strip()
            # Remove trailing punctuation
            answer = answer.rstrip('.,;:!?%')
    
    # Strategy 4: If no answer found, try to find a number after the program
    if not answer and program and output:
        # Look for numbers that appear after the program line
        program_idx = output.find(program)
        if program_idx != -1:
            after_program = output[program_idx + len(program):]
            # Look for standalone numbers
            number_pattern = r'\b(-?\d+\.?\d*(?:e[+-]?\d+)?)\b'
            matches = re.findall(number_pattern, after_program)
            if matches:
                # Take the first number found
                answer = matches[0]
    
    # Clean up program: remove spaces between characters if present
    if program and ' ' in program and len(program.split()) > 10:
        # Likely has spaces between characters, try to fix common patterns
        program = program.replace(' s u b t r a c t ', ' subtract ').replace('s u b t r a c t', 'subtract')
        program = program.replace(' a d d ', ' add ').replace('a d d', 'add')
        program = program.replace(' m u l t i p l y ', ' multiply ').replace('m u l t i p l y', 'multiply')
        program = program.replace(' d i v i d e ', ' divide ').replace('d i v i d e', 'divide')
        program = program.replace(' g r e a t e r ', ' greater ').replace('g r e a t e r', 'greater')
        program = program.replace(' e x p ', ' exp ').replace('e x p', 'exp')
        # Clean up multiple spaces
        program = ' '.join(program.split())
    
    # Remove const_ prefixes from program (model may still output them despite instructions)
    if program:
        import re
        program = re.sub(r'const_(\d+\.?\d*)', r'\1', program)
        program = re.sub(r'const_', '', program)
    
    return program, answer


def run_inference(model, tokenizer, test_data: List[Dict], config: dict) -> List[Dict]:
    """
    Run ICL inference on test data using config template.
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        test_data: Test examples
        config: Configuration dictionary
        
    Returns:
        List of predictions in same format as LoRA inference
    """
    gen_config = config['generation']
    predictions = []
    
    # Configure progress bar for nohup compatibility
    pbar = tqdm(
        test_data, 
        desc="Running ICL inference",
        ncols=100,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
        file=sys.stdout,
        dynamic_ncols=False
    )
    
    for example in pbar:
        # Clear CUDA cache periodically to prevent fragmentation
        if len(predictions) % 10 == 0:
            torch.cuda.empty_cache()
        
        # Create prompt using config template
        prompt = create_prompt_from_config(config, example)
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=8192)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=gen_config['max_new_tokens'],
                temperature=gen_config['temperature'],
                do_sample=gen_config.get('do_sample', True),
                top_p=gen_config.get('top_p', 0.95),
                num_beams=gen_config.get('num_beams', 1),
                repetition_penalty=gen_config.get('repetition_penalty', 1.1),
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode
        generated_text = tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        # Free memory immediately
        del inputs, outputs
        
        # Parse output
        program, answer = parse_model_output(generated_text)
        
        # Store prediction in same format as LoRA inference
        predictions.append({
            'id': example.get('id', None),
            'question': example['question'],
            'predicted_program': program,
            'predicted_answer': answer,
            'gold_program': ' '.join(example['program']) if isinstance(example.get('program'), list) else example.get('program', ''),
            'gold_answer': str(example.get('answer', '')),
            'raw_output': generated_text
        })
    
    return predictions


def main():
    parser = argparse.ArgumentParser(description="Unified ICL Inference for FinQA")
    parser.add_argument("--config", type=str, default="../configs/icl_config.yaml",
                        help="Path to ICL config YAML file")
    parser.add_argument("--model_name", type=str, default=None,
                        help="Override model name from config")
    parser.add_argument("--data_dir", type=str, default="/data/simplified",
                        help="Data directory")
    parser.add_argument("--output_dir", type=str, default="../results/predictions",
                        help="Output directory for predictions")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max test samples (for quick testing)")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Override temperature from config")
    parser.add_argument("--num_shots", type=int, default=5,
                        help="Number of few-shot examples (e.g., 0, 3, 5, 10)")
    
    args = parser.parse_args()
    
    # Load config
    print("=" * 80)
    print("UNIFIED ICL INFERENCE FOR FINQA")
    print("=" * 80)
    print(f"Config: {args.config}")
    
    config = load_config(args.config)
    
    # Override model if specified
    if args.model_name:
        config['model']['model_name_or_path'] = args.model_name
    
    # Override temperature if specified
    if args.temperature is not None:
        config['generation']['temperature'] = args.temperature
    
    model_name = config['model']['model_name_or_path']
    print(f"Model: {model_name}")
    print(f"Few-shot examples: {config['icl']['num_shots']}")
    print(f"Temperature: {config['generation']['temperature']}")
    print(f"Output: {args.output_dir}")
    print("=" * 80)
    
    # Load model and tokenizer
    print("\n[1/5] Loading model and tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=getattr(torch, config['model'].get('torch_dtype', 'bfloat16')),
        device_map=config['model'].get('device_map', 'auto'),
        load_in_8bit=True,  # Force 8-bit to save memory
        max_memory={0: "20GiB"},  # Limit GPU 0 memory usage
        trust_remote_code=True
    )
    model.eval()
    print(f"  Model loaded on device: {model.device}", flush=True)
    
    # Load data
    print("\n[2/5] Loading data...", flush=True)
    train_path = os.path.join(args.data_dir, "train_simplified.json")
    test_path = os.path.join(args.data_dir, "test_simplified.json")
    
    with open(train_path, 'r') as f:
        train_data = json.load(f)
    with open(test_path, 'r') as f:
        test_data = json.load(f)
    
    if args.max_samples:
        test_data = test_data[:args.max_samples]
    
    print(f"  Train examples: {len(train_data)}", flush=True)
    print(f"  Test examples: {len(test_data)}", flush=True)
    
    # Select and format few-shot examples
    print(f"\n[3/5] Selecting and formatting few-shot examples...", flush=True)
    num_shots = args.num_shots
    selection_method = config['icl'].get('example_selection', 'diverse')
    few_shot_examples = select_few_shot_examples(train_data, num_shots, selection_method)
    
    # Convert to format expected by prompt template (includes program and answer)
    config['few_shot_examples'] = finqa_to_yaml_examples(few_shot_examples)
    
    for i, ex in enumerate(config['few_shot_examples'], 1):
        print(f"  Example {i}: {ex['question'][:60]}...", flush=True)
    
    # Run inference
    print("\n[4/5] Running inference...", flush=True)
    sys.stdout.flush()  # Extra flush before progress bar starts
    predictions = run_inference(model, tokenizer, test_data, config)
    
    # Save predictions
    print("\n[5/5] Saving predictions...", flush=True)
    
    # Determine output directory based on config file name
    config_basename = os.path.basename(args.config)
    if "config_1" in config_basename or "config1" in config_basename:
        output_dir = "../results/icl_config_1"
    elif "config_2" in config_basename or "config2" in config_basename:
        output_dir = "../results/icl_config_2"
    else:
        output_dir = args.output_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    model_name_short = model_name.split("/")[-1]
    output_filename = f"{model_name_short}_icl_predictions_latest.json"
    output_path = os.path.join(output_dir, output_filename)
    
    with open(output_path, 'w') as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)
    
    print(f"  Saved to: {output_path}", flush=True)
    
    # Quick preview
    print("\n" + "=" * 80)
    print("PREVIEW OF PREDICTIONS")
    print("=" * 80)
    print(f"Total predictions: {len(predictions)}")
    if predictions:
        print(f"\nFirst prediction:")
        print(f"  Question: {predictions[0]['question']}")
        print(f"  Predicted program: {predictions[0]['predicted_program'][:100]}...")
        print(f"  Predicted answer: {predictions[0]['predicted_answer']}")
        print(f"  Gold program: {predictions[0]['gold_program'][:100]}...")
        print(f"  Gold answer: {predictions[0]['gold_answer']}")
    print("=" * 80)
    print(f"\n✓ ICL inference complete! Results saved to {output_path}")


if __name__ == "__main__":
    main()
