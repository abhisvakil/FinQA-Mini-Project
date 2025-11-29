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


def format_few_shot_examples(examples: List[Dict], system_prompt: str) -> str:
    """
    Format few-shot examples into a prompt.
    
    Args:
        examples: List of example dictionaries
        system_prompt: System instructions
        
    Returns:
        Formatted few-shot prompt
    """
    prompt_parts = [system_prompt, "\nHere are some examples:\n"]
    
    for i, ex in enumerate(examples, 1):
        prompt_parts.append(f"\nExample {i}:")
        prompt_parts.append(f"Question: {ex['question']}")
        
        # Add table if present
        if ex.get('table'):
            prompt_parts.append("\nTable:")
            table = ex['table']
            if len(table) > 0:
                header = " | ".join(table[0])
                prompt_parts.append(f"| {header} |")
                for row in table[1:min(4, len(table))]:  # Limit rows
                    row_str = " | ".join(str(cell) for cell in row)
                    prompt_parts.append(f"| {row_str} |")
        
        # Add program
        program_str = " ".join(ex['program']) if isinstance(ex['program'], list) else ex['program']
        prompt_parts.append(f"\nProgram: {program_str}\n")
    
    return "\n".join(prompt_parts)


def create_icl_prompt(example: Dict, few_shot_prompt: str) -> str:
    """
    Create full ICL prompt with few-shot examples and current question.
    
    Args:
        example: Test example
        few_shot_prompt: Pre-formatted few-shot examples
        
    Returns:
        Complete prompt
    """
    prompt_parts = [few_shot_prompt, "\nNow answer this question:\n"]
    
    # Add current question
    prompt_parts.append(f"Question: {example['question']}")
    
    # Add context
    if example.get('pre_text'):
        prompt_parts.append("\nText:")
        for sent in example['pre_text'][:10]:  # Limit length
            prompt_parts.append(sent)
    
    # Add table
    if example.get('table'):
        prompt_parts.append("\nTable:")
        table = example['table']
        if len(table) > 0:
            header = " | ".join(table[0])
            prompt_parts.append(f"| {header} |")
            for row in table[1:]:
                row_str = " | ".join(str(cell) for cell in row)
                prompt_parts.append(f"| {row_str} |")
    
    prompt_parts.append("\nProgram:")
    
    return "\n".join(prompt_parts)


def parse_model_output(output: str) -> tuple:
    """
    Parse model output to extract program and answer.
    Same format as LoRA inference.
    
    Args:
        output: Raw model output
        
    Returns:
        (program, answer) tuple
    """
    program = ""
    answer = ""
    
    lines = output.strip().split('\n')
    
    for line in lines:
        if line.strip().startswith("Program:"):
            program = line.replace("Program:", "").strip()
        elif line.strip().startswith("Answer:"):
            answer = line.replace("Answer:", "").strip()
    
    # If no markers, assume first line is program
    if not program and lines:
        program = lines[0].strip()
    
    # Try to extract answer if not found
    if not answer and len(lines) > 1:
        for line in lines[1:]:
            if line.strip() and not line.startswith("Program"):
                answer = line.strip()
                break
    
    return program, answer


def run_inference(model, tokenizer, test_data: List[Dict], few_shot_prompt: str,
                 config: dict) -> List[Dict]:
    """
    Run ICL inference on test data.
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        test_data: Test examples
        few_shot_prompt: Pre-formatted few-shot examples
        config: Configuration dictionary
        
    Returns:
        List of predictions in same format as LoRA inference
    """
    gen_config = config['generation']
    predictions = []
    
    for example in tqdm(test_data, desc="Running ICL inference"):
        # Create prompt
        prompt = create_icl_prompt(example, few_shot_prompt)
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=gen_config['max_new_tokens'],
                temperature=gen_config['temperature'],
                do_sample=gen_config.get('do_sample', True),
                top_p=gen_config.get('top_p', 0.95),
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode
        generated_text = tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        # Parse output
        program, answer = parse_model_output(generated_text)
        
        # Store prediction in same format as LoRA inference
        predictions.append({
            'id': example['id'],
            'question': example['question'],
            'predicted_program': program,
            'predicted_answer': answer,
            'gold_program': ' '.join(example['program']) if isinstance(example['program'], list) else example['program'],
            'gold_answer': str(example['answer']),
            'raw_output': generated_text
        })
    
    return predictions


def main():
    parser = argparse.ArgumentParser(description="Unified ICL Inference for FinQA")
    parser.add_argument("--config", type=str, default="../configs/icl_config.yaml",
                        help="Path to ICL config YAML file")
    parser.add_argument("--model_name", type=str, default=None,
                        help="Override model name from config")
    parser.add_argument("--data_dir", type=str, default="../data/simplified",
                        help="Data directory")
    parser.add_argument("--output_dir", type=str, default="../results/predictions",
                        help="Output directory for predictions")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max test samples (for quick testing)")
    
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
    
    model_name = config['model']['model_name_or_path']
    print(f"Model: {model_name}")
    print(f"Few-shot examples: {config['icl']['num_shots']}")
    print(f"Temperature: {config['generation']['temperature']}")
    print(f"Output: {args.output_dir}")
    print("=" * 80)
    
    # Load model and tokenizer
    print("\n[1/5] Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=getattr(torch, config['model'].get('torch_dtype', 'bfloat16')),
        device_map=config['model'].get('device_map', 'auto'),
        load_in_8bit=config['model'].get('load_in_8bit', False),
        trust_remote_code=True
    )
    model.eval()
    print(f"  Model loaded on device: {model.device}")
    
    # Load data
    print("\n[2/5] Loading data...")
    train_path = os.path.join(args.data_dir, "train_simplified.json")
    test_path = os.path.join(args.data_dir, "test_simplified.json")
    
    with open(train_path, 'r') as f:
        train_data = json.load(f)
    with open(test_path, 'r') as f:
        test_data = json.load(f)
    
    if args.max_samples:
        test_data = test_data[:args.max_samples]
    
    print(f"  Train examples: {len(train_data)}")
    print(f"  Test examples: {len(test_data)}")
    
    # Select few-shot examples
    print(f"\n[3/5] Selecting few-shot examples...")
    num_shots = config['icl']['num_shots']
    selection_method = config['icl'].get('example_selection', 'diverse')
    few_shot_examples = select_few_shot_examples(train_data, num_shots, selection_method)
    
    for i, ex in enumerate(few_shot_examples, 1):
        print(f"  Example {i}: {ex['question'][:60]}...")
    
    # Create few-shot prompt
    system_prompt = config.get('system_prompt', '')
    few_shot_prompt = format_few_shot_examples(few_shot_examples, system_prompt)
    
    # Run inference
    print("\n[4/5] Running inference...")
    predictions = run_inference(model, tokenizer, test_data, few_shot_prompt, config)
    
    # Save predictions
    print("\n[5/5] Saving predictions...")
    os.makedirs(args.output_dir, exist_ok=True)
    
    model_name_short = model_name.split("/")[-1]
    output_filename = f"{model_name_short}_icl_predictions.json"
    output_path = os.path.join(args.output_dir, output_filename)
    
    with open(output_path, 'w') as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)
    
    print(f"  Saved to: {output_path}")
    
    # Quick preview
    print("\n" + "=" * 80)
    print("PREVIEW OF PREDICTIONS")
    print("=" * 80)
    print(f"Total predictions: {len(predictions)}")
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
