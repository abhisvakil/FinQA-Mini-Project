"""
In-Context Learning (ICL) inference script for FinQA.
Uses few-shot prompting with Llama-3-8B or Mistral-7B without fine-tuning.
"""

import os
import sys
import json
import torch
import argparse
from typing import List, Dict, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import random

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_loader_simplified import FinQASimplifiedLoader


def create_few_shot_prompt(question: str, pre_text: List[str], post_text: List[str], 
                           table: List[List[str]], examples: List[Dict]) -> str:
    """
    Create a few-shot prompt with examples.
    
    Args:
        question: The question to answer
        pre_text: Context before table
        post_text: Context after table  
        table: Financial table
        examples: Few-shot examples
        
    Returns:
        Formatted prompt string
    """
    system_prompt = """You are a financial analyst. Answer numerical questions about financial reports by generating executable reasoning programs.

Available Operations:
- add(a, b): Add two numbers
- subtract(a, b): Subtract b from a  
- multiply(a, b): Multiply two numbers
- divide(a, b): Divide a by b
- greater(a, b): Return the greater of two numbers
- exp(a, b): Calculate a raised to the power of b

Here are some examples:

"""
    
    # Add few-shot examples
    example_texts = []
    for i, ex in enumerate(examples, 1):
        ex_text = f"Example {i}:\n"
        ex_text += f"Question: {ex['question']}\n"
        
        # Add table from example
        if ex['table']:
            ex_text += "Table:\n"
            table_data = ex['table']
            if len(table_data) > 0:
                header = " | ".join(table_data[0])
                ex_text += f"| {header} |\n"
                for row in table_data[1:min(4, len(table_data))]:  # Limit rows
                    row_str = " | ".join(str(cell) for cell in row)
                    ex_text += f"| {row_str} |\n"
        
        # Add program and answer
        program_str = " ".join(ex['program']) if ex['program'] else ""
        ex_text += f"\nProgram: {program_str}\n"
        ex_text += f"Answer: {ex['answer']}\n"
        
        example_texts.append(ex_text)
    
    few_shot_examples = "\n".join(example_texts)
    
    # Add current question
    current_question = "\nNow answer this question:\n\n"
    current_question += f"Question: {question}\n"
    
    # Add context
    if pre_text:
        current_question += "Text Context:\n"
        for sent in pre_text[:5]:  # Limit sentences
            current_question += f"{sent}\n"
    
    # Add table
    if table:
        current_question += "\nTable:\n"
        if len(table) > 0:
            header = " | ".join(table[0])
            current_question += f"| {header} |\n"
            for row in table[1:]:
                row_str = " | ".join(str(cell) for cell in row)
                current_question += f"| {row_str} |\n"
    
    current_question += "\nProgram:"
    
    return system_prompt + few_shot_examples + current_question


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
        # Select diverse examples based on program length and table size
        examples = []
        
        # Sort by program length
        sorted_by_program = sorted(train_data, key=lambda x: len(x.get('program', [])))
        
        # Select from different quartiles
        n = len(sorted_by_program)
        indices = [
            n // 4,           # Short program
            n // 2,           # Medium program
            3 * n // 4,       # Long program
        ]
        
        # Add some random ones
        for idx in indices:
            if idx < len(sorted_by_program):
                examples.append(sorted_by_program[idx])
        
        # Fill remaining with random
        remaining = [ex for ex in train_data if ex not in examples]
        if remaining and len(examples) < num_examples:
            examples.extend(random.sample(remaining, min(num_examples - len(examples), len(remaining))))
        
        return examples[:num_examples]
    
    return train_data[:num_examples]


def parse_model_output(output: str) -> tuple[str, str]:
    """
    Parse model output to extract program and answer.
    
    Args:
        output: Raw model output
        
    Returns:
        (program, answer) tuple
    """
    program = ""
    answer = ""
    
    # Try to extract program
    if "Program:" in output:
        program_part = output.split("Program:")[1]
        if "Answer:" in program_part:
            program = program_part.split("Answer:")[0].strip()
            answer = program_part.split("Answer:")[1].strip()
        else:
            program = program_part.strip()
    else:
        # If no "Program:" marker, try to extract from beginning
        lines = output.strip().split('\n')
        if lines:
            program = lines[0].strip()
    
    return program, answer


def run_inference(model, tokenizer, test_data: List[Dict], few_shot_examples: List[Dict],
                 max_new_tokens: int = 256, temperature: float = 0.1) -> List[Dict]:
    """
    Run ICL inference on test data.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        test_data: Test examples
        few_shot_examples: Few-shot examples to use
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        
    Returns:
        List of predictions
    """
    predictions = []
    
    for example in tqdm(test_data, desc="Running inference"):
        # Create prompt
        prompt = create_few_shot_prompt(
            example['question'],
            example['pre_text'],
            example['post_text'],
            example['table'],
            few_shot_examples
        )
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode
        generated_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        # Parse output
        program, answer = parse_model_output(generated_text)
        
        # Store prediction
        predictions.append({
            'id': example['id'],
            'question': example['question'],
            'predicted_program': program,
            'predicted_answer': answer,
            'gold_program': ' '.join(example['program']),
            'gold_answer': example['answer'],
            'raw_output': generated_text
        })
    
    return predictions


def main():
    parser = argparse.ArgumentParser(description="ICL Inference for FinQA")
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct",
                        help="Model name or path")
    parser.add_argument("--data_dir", type=str, default="../data/simplified",
                        help="Data directory")
    parser.add_argument("--output_dir", type=str, default="../results/icl",
                        help="Output directory")
    parser.add_argument("--num_shots", type=int, default=5,
                        help="Number of few-shot examples")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max test samples (for quick testing)")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="Sampling temperature")
    parser.add_argument("--max_new_tokens", type=int, default=256,
                        help="Maximum tokens to generate")
    parser.add_argument("--load_in_8bit", action="store_true",
                        help="Load model in 8-bit for memory efficiency")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("ICL INFERENCE FOR FINQA")
    print("=" * 80)
    print(f"Model: {args.model_name}")
    print(f"Few-shot examples: {args.num_shots}")
    print(f"Temperature: {args.temperature}")
    print(f"Output: {args.output_dir}")
    print("=" * 80)
    
    # Load model and tokenizer
    print("\n[1/5] Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        load_in_8bit=args.load_in_8bit,
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
    print(f"\n[3/5] Selecting {args.num_shots} few-shot examples...")
    few_shot_examples = select_few_shot_examples(train_data, args.num_shots, "diverse")
    
    for i, ex in enumerate(few_shot_examples, 1):
        print(f"  Example {i}: {ex['question'][:60]}...")
    
    # Run inference
    print("\n[4/5] Running inference...")
    predictions = run_inference(
        model, tokenizer, test_data, few_shot_examples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature
    )
    
    # Save predictions
    print("\n[5/5] Saving predictions...")
    os.makedirs(args.output_dir, exist_ok=True)
    
    model_name_short = args.model_name.split("/")[-1]
    output_path = os.path.join(args.output_dir, f"{model_name_short}_ICL_predictions.json")
    
    with open(output_path, 'w') as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)
    
    print(f"  Saved to: {output_path}")
    
    # Quick stats
    print("\n" + "=" * 80)
    print("QUICK STATISTICS")
    print("=" * 80)
    print(f"Total predictions: {len(predictions)}")
    print(f"Sample prediction:")
    print(f"  Question: {predictions[0]['question']}")
    print(f"  Predicted: {predictions[0]['predicted_program'][:100]}...")
    print(f"  Gold: {predictions[0]['gold_program'][:100]}...")
    print("=" * 80)
    print(f"\n✓ Inference complete! Results saved to {output_path}")


if __name__ == "__main__":
    main()
