#!/usr/bin/env python3
"""
In-Context Learning inference with Enhanced Executor and Retry Logic.
Optimized for A100 GPU with batch processing and BF16 precision.
"""

import json
import os
import argparse
import yaml
from pathlib import Path
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from enhanced_executor import EnhancedExecutor
import re


def extract_program_and_answer(text: str):
    """
    Extract program and answer from model output.
    
    Expected format:
        Program: subtract(750, 500)
        Answer: 250
    """
    program = ""
    answer = ""
    
    lines = text.strip().split('\n')
    for line in lines:
        line_lower = line.lower().strip()
        if line_lower.startswith('program:'):
            program = line.split(':', 1)[1].strip()
        elif line_lower.startswith('answer:'):
            answer = line.split(':', 1)[1].strip()
    
    # If not found in structured format, try to extract
    if not program:
        # Look for operation patterns
        for line in lines:
            if any(op in line for op in ['add', 'subtract', 'multiply', 'divide', 'greater', 'exp']):
                # Extract everything that looks like operations
                program = line.strip()
                break
    
    if not answer:
        # Try to find numbers
        for line in lines:
            numbers = re.findall(r'-?\d+\.?\d*', line)
            if numbers:
                answer = numbers[-1]
                break
    
    return program, answer


def format_table(table):
    """Format table for display."""
    if not table:
        return ""
    
    headers = table[0] if table else []
    rows = table[1:] if len(table) > 1 else []
    
    table_str = "Table:\n"
    if headers:
        table_str += "| " + " | ".join(str(h) for h in headers) + " |\n"
        table_str += "|" + "|".join(["---"] * len(headers)) + "|\n"
    
    for row in rows:
        table_str += "| " + " | ".join(str(cell) for cell in row) + " |\n"
    
    return table_str


def create_prompt(config, question, context_data, feedback=None):
    """
    Create prompt for the model.
    
    Args:
        config: Configuration dict
        question: Question text
        context_data: Dict with pre_text, post_text, table
        feedback: Optional feedback from previous attempt
    """
    # Format context
    pre_text = context_data.get('pre_text', [])
    post_text = context_data.get('post_text', [])
    table = context_data.get('table', [])
    
    pre_text_str = " ".join(pre_text) if isinstance(pre_text, list) else str(pre_text)
    post_text_str = " ".join(post_text) if isinstance(post_text, list) else str(post_text)
    table_str = format_table(table)
    
    context = f"{pre_text_str}\n\n{table_str}\n\n{post_text_str}".strip()
    
    # Add feedback if this is a retry
    if feedback:
        system_prompt = config.get('system_prompt', '')
        system_prompt += f"\n\n⚠️ FEEDBACK FROM PREVIOUS ATTEMPT:\n{feedback}\n"
        system_prompt += "Please revise your answer based on this feedback.\n"
    else:
        system_prompt = config.get('system_prompt', '')
    
    # Format prompt using template
    prompt = config['prompt_template'].format(
        system_prompt=system_prompt,
        few_shot_examples="",  # TODO: Add few-shot examples if needed
        question=question,
        context=context
    )
    
    return prompt


def process_single_sample(sample, model, tokenizer, config, executor_config):
    """
    Process a single sample with executor and retry logic.
    
    Args:
        sample: Test sample
        model: Language model
        tokenizer: Tokenizer
        config: Configuration
        executor_config: Executor configuration
    
    Returns:
        Result dictionary
    """
    question = sample.get('qa', {}).get('question', '')
    gold_program = sample.get('qa', {}).get('program', '')
    gold_answer = sample.get('qa', {}).get('exe_ans', '')
    
    context_data = {
        'pre_text': sample.get('pre_text', []),
        'post_text': sample.get('post_text', []),
        'table': sample.get('table', [])
    }
    
    # Initialize executor
    executor = EnhancedExecutor(table=context_data.get('table'))
    
    max_retries = executor_config.get('max_retries', 2)
    feedback = None
    
    attempt_history = []
    
    for attempt in range(max_retries + 1):
        # Create prompt (with feedback if retry)
        prompt = create_prompt(config, question, context_data, feedback)
        
        # Generate
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=config['generation']['max_new_tokens'],
                temperature=config['generation']['temperature'],
                top_p=config['generation']['top_p'],
                top_k=config['generation']['top_k'],
                do_sample=config['generation']['do_sample'],
                num_beams=config['generation']['num_beams'],
                repetition_penalty=config['generation'].get('repetition_penalty', 1.0),
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
            )
        
        # Decode
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the generated part (after prompt)
        if prompt in generated_text:
            generated_answer = generated_text.split(prompt)[-1].strip()
        else:
            generated_answer = generated_text[len(prompt):].strip()
        
        # Extract program and answer
        predicted_program, predicted_answer = extract_program_and_answer(generated_answer)
        
        # Execute program
        exec_result = executor.execute(predicted_program)
        
        # Store attempt
        attempt_info = {
            'attempt': attempt,
            'predicted_program': predicted_program,
            'predicted_answer': predicted_answer,
            'executor_result': exec_result,
            'feedback': feedback,
            'raw_output': generated_answer
        }
        attempt_history.append(attempt_info)
        
        # Check if execution was successful
        if exec_result['success']:
            # Use executor's answer if available
            if exec_result['answer'] is not None:
                predicted_answer = str(round(exec_result['answer'], 5))
            
            # Validate result
            is_valid, warnings = executor.validate_result(exec_result['answer'], question)
            
            if is_valid or attempt == max_retries:
                # Success or out of retries
                break
            else:
                # Warnings found, provide feedback for retry
                feedback = "⚠️ Program executed but result seems unusual:\n"
                feedback += "\n".join(warnings)
                feedback += "\n\nPlease review your logic and revise the program."
        else:
            # Execution failed
            if attempt < max_retries:
                # Provide feedback for retry
                feedback = exec_result['feedback']
            else:
                # Out of retries
                break
    
    # Return final result
    return {
        'id': sample.get('id', ''),
        'question': question,
        'predicted_program': predicted_program,
        'predicted_answer': predicted_answer,
        'gold_program': gold_program,
        'gold_answer': str(gold_answer),
        'executor_success': exec_result['success'],
        'executor_answer': str(exec_result['answer']) if exec_result['answer'] is not None else '',
        'attempts': len(attempt_history),
        'attempt_history': attempt_history,
        'raw_output': generated_answer
    }


def run_icl_with_executor(config_path):
    """Run ICL inference with executor and retry logic."""
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("="*60)
    print("ICL Inference with Enhanced Executor")
    print("="*60)
    
    # Load test data
    test_file = config['data']['test_file']
    print(f"\n📂 Loading test data from: {test_file}")
    
    with open(test_file, 'r') as f:
        test_data = json.load(f)
    
    max_samples = config['data'].get('max_samples')
    if max_samples:
        test_data = test_data[:max_samples]
        print(f"🔢 Processing first {max_samples} samples")
    
    print(f"✅ Loaded {len(test_data)} test examples")
    
    # Load model
    print(f"\n🤖 Loading model: {config['model']['model_name_or_path']}")
    
    tokenizer = AutoTokenizer.from_pretrained(config['model']['model_name_or_path'])
    
    # Use bfloat16 for A100
    torch_dtype = torch.bfloat16 if config['model'].get('torch_dtype') == 'bfloat16' else torch.float16
    
    model = AutoModelForCausalLM.from_pretrained(
        config['model']['model_name_or_path'],
        torch_dtype=torch_dtype,
        device_map=config['model']['device_map']
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"✅ Model loaded on {model.device}")
    print(f"   Precision: {torch_dtype}")
    
    # Run inference
    print("\n🔮 Running ICL inference with executor...")
    
    executor_config = config.get('executor', {})
    results = []
    
    for sample in tqdm(test_data, desc="Processing"):
        result = process_single_sample(sample, model, tokenizer, config, executor_config)
        results.append(result)
    
    # Save results
    output_file = config['data']['output_file']
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    print(f"\n💾 Saving results to: {output_file}")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print statistics
    print("\n" + "="*60)
    print("Statistics")
    print("="*60)
    
    total = len(results)
    executor_success = sum(1 for r in results if r['executor_success'])
    retry_counts = {}
    for r in results:
        attempts = r['attempts']
        retry_counts[attempts] = retry_counts.get(attempts, 0) + 1
    
    print(f"Total samples: {total}")
    print(f"Executor successful: {executor_success} ({executor_success/total*100:.1f}%)")
    print(f"\nRetry distribution:")
    for attempts in sorted(retry_counts.keys()):
        count = retry_counts[attempts]
        print(f"  {attempts} attempt(s): {count} samples ({count/total*100:.1f}%)")
    
    # Sample results
    print("\n" + "="*60)
    print("Sample Results (first 3)")
    print("="*60)
    
    for i, result in enumerate(results[:3], 1):
        print(f"\n--- Example {i} ---")
        print(f"Question: {result['question'][:80]}...")
        print(f"Attempts: {result['attempts']}")
        print(f"Predicted Program: {result['predicted_program'][:60]}...")
        print(f"Gold Program: {result['gold_program'][:60]}...")
        print(f"Executor Success: {'✅' if result['executor_success'] else '❌'}")
        if result['executor_answer']:
            print(f"Executor Answer: {result['executor_answer']}")
        print(f"Gold Answer: {result['gold_answer']}")
    
    print("\n" + "="*60)
    print("✅ ICL Inference with Executor Complete!")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Run ICL inference with executor")
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to configuration file'
    )
    
    args = parser.parse_args()
    run_icl_with_executor(args.config)


if __name__ == '__main__':
    main()

