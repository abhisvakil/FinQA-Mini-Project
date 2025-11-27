"""
Unified inference script for FinQA.
Runs inference on trained LoRA/QLoRA models or base models (ICL).
Saves predictions in a consistent format for evaluation.
"""

import os
import sys
import json
import torch
import argparse
from typing import List, Dict, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_loader_simplified import FinQASimplifiedLoader


def load_model_and_tokenizer(model_name: str, adapter_path: Optional[str] = None, 
                             load_in_8bit: bool = False):
    """
    Load model and tokenizer, optionally with LoRA/QLoRA adapters.
    
    Args:
        model_name: Base model name or path
        adapter_path: Path to LoRA/QLoRA adapters (None for base model)
        load_in_8bit: Whether to load in 8-bit
        
    Returns:
        (model, tokenizer) tuple
    """
    print(f"Loading model: {model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        load_in_8bit=load_in_8bit,
        trust_remote_code=True
    )
    
    # Load adapters if provided
    if adapter_path:
        print(f"Loading adapters from: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()  # Merge adapters into base model
    
    model.eval()
    print(f"Model loaded on device: {model.device}")
    
    return model, tokenizer


def create_inference_prompt(example: Dict, method: str = "direct") -> str:
    """
    Create prompt for inference.
    
    Args:
        example: Test example
        method: "direct" for LoRA/QLoRA, "icl" for few-shot (handled separately)
        
    Returns:
        Formatted prompt
    """
    instruction = """Answer the following financial question by generating a reasoning program and the final answer.

Available Operations:
- add(a, b): Add two numbers
- subtract(a, b): Subtract b from a
- multiply(a, b): Multiply two numbers
- divide(a, b): Divide a by b
- greater(a, b): Return the greater of two numbers
- exp(a, b): Calculate a raised to the power of b

"""
    
    # Add context
    context_parts = []
    
    if example['pre_text']:
        context_parts.append("Text:")
        context_parts.extend(example['pre_text'][:10])  # Limit length
    
    if example['table']:
        context_parts.append("\nTable:")
        table = example['table']
        if len(table) > 0:
            header = " | ".join(table[0])
            context_parts.append(f"| {header} |")
            for row in table[1:]:
                row_str = " | ".join(str(cell) for cell in row)
                context_parts.append(f"| {row_str} |")
    
    context = "\n".join(context_parts)
    
    prompt = f"{instruction}Question: {example['question']}\n\nContext:\n{context}\n\nGenerate the reasoning program and final answer:\n\nProgram:"
    
    return prompt


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


def run_inference(model, tokenizer, test_data: List[Dict], 
                 max_new_tokens: int = 256, temperature: float = 0.1,
                 batch_size: int = 1) -> List[Dict]:
    """
    Run inference on test data.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        test_data: Test examples
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        batch_size: Batch size for inference (1 recommended)
        
    Returns:
        List of predictions
    """
    predictions = []
    
    for example in tqdm(test_data, desc="Running inference"):
        # Create prompt
        prompt = create_inference_prompt(example, method="direct")
        
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
        
        # Store prediction
        predictions.append({
            'id': example['id'],
            'question': example['question'],
            'predicted_program': program,
            'predicted_answer': answer,
            'gold_program': ' '.join(example['program']),
            'gold_answer': str(example['answer']),
            'raw_output': generated_text
        })
    
    return predictions


def main():
    parser = argparse.ArgumentParser(description="Unified Inference for FinQA")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Base model name (e.g., meta-llama/Meta-Llama-3-8B-Instruct)")
    parser.add_argument("--adapter_path", type=str, default=None,
                        help="Path to LoRA/QLoRA adapters (optional, for fine-tuned models)")
    parser.add_argument("--method", type=str, default="lora", choices=["lora", "qlora", "base"],
                        help="Method used (for naming output files)")
    parser.add_argument("--data_dir", type=str, default="../data/simplified",
                        help="Data directory")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for predictions")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max test samples (for quick testing)")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="Sampling temperature")
    parser.add_argument("--max_new_tokens", type=int, default=256,
                        help="Maximum tokens to generate")
    parser.add_argument("--load_in_8bit", action="store_true",
                        help="Load model in 8-bit")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("UNIFIED INFERENCE FOR FINQA")
    print("=" * 80)
    print(f"Model: {args.model_name}")
    print(f"Adapters: {args.adapter_path if args.adapter_path else 'None (base model)'}")
    print(f"Method: {args.method}")
    print(f"Temperature: {args.temperature}")
    print(f"Output: {args.output_dir}")
    print("=" * 80)
    
    # Load model and tokenizer
    print("\n[1/4] Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(
        args.model_name,
        args.adapter_path,
        args.load_in_8bit
    )
    
    # Load test data
    print("\n[2/4] Loading test data...")
    test_path = os.path.join(args.data_dir, "test_simplified.json")
    
    with open(test_path, 'r') as f:
        test_data = json.load(f)
    
    if args.max_samples:
        test_data = test_data[:args.max_samples]
    
    print(f"  Test examples: {len(test_data)}")
    
    # Run inference
    print("\n[3/4] Running inference...")
    predictions = run_inference(
        model, tokenizer, test_data,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature
    )
    
    # Save predictions
    print("\n[4/4] Saving predictions...")
    os.makedirs(args.output_dir, exist_ok=True)
    
    model_name_short = args.model_name.split("/")[-1]
    output_filename = f"{model_name_short}_{args.method}_predictions.json"
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
    print(f"\n✓ Inference complete! Results saved to {output_path}")


if __name__ == "__main__":
    main()
