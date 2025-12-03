#!/usr/bin/env python3
"""
Script to evaluate LoRA predictions for multiple models by comparing:
1. predicted_program vs gold_program
2. predicted_answer vs gold_answer (after cleaning and rounding)
"""

import json
import re
import csv
from pathlib import Path


def clean_predicted_answer(answer_str, remove_assistant_suffix=False):
    """
    Clean predicted answer:
    - Optionally remove 'assistant:...' suffix from predicted answer
    - Extract the numeric value before any extra text
    """
    if not answer_str:
        return ""
    
    answer_str = str(answer_str).strip()
    
    # Remove everything after 'assistant:' if flag is set
    if remove_assistant_suffix and 'assistant:' in answer_str.lower():
        answer_str = re.split(r'assistant:', answer_str, flags=re.IGNORECASE)[0]
        answer_str = answer_str.strip()
    
    # Try to extract the first numeric value (including negative and decimal)
    # This handles cases like "0.107142857 > 0.107142857"
    match = re.search(r'-?\d+\.?\d*', answer_str)
    if match:
        return match.group(0)
    
    return answer_str


def normalize_program(program_str):
    """
    Normalize program string by removing all spaces for robust comparison.
    This handles cases like "5 8 2 9,   5 7 3 5" vs "5 8 2 9 ,   5 7 3 5"
    """
    if not program_str:
        return ""
    # Remove all spaces for comparison
    normalized = ''.join(program_str.split())
    return normalized


def safe_float_round(value_str, decimals=2):
    """
    Safely convert string to float and round to specified decimals.
    Returns None if conversion fails.
    """
    try:
        return round(float(value_str), decimals)
    except (ValueError, TypeError):
        return None


def evaluate_predictions(predictions_file, output_csv, remove_assistant_suffix=False):
    """
    Evaluate predictions and save results to CSV.
    
    Args:
        predictions_file: Path to predictions JSON file
        output_csv: Path to output CSV file
        remove_assistant_suffix: Whether to remove 'assistant:' suffix from predicted answers
    """
    # Load predictions
    with open(predictions_file, 'r') as f:
        predictions = json.load(f)
    
    model_name = Path(predictions_file).stem.replace('_lora_predictions', '')
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name}")
    print(f"{'='*60}")
    print(f"Loaded {len(predictions)} predictions from {predictions_file}")
    
    # Initialize counters
    total_count = len(predictions)
    program_correct = 0
    answer_correct = 0
    
    # Store detailed results
    results = []
    
    for idx, pred in enumerate(predictions):
        pred_id = pred.get('id', f'unknown_{idx}')
        question = pred.get('question', '')
        
        # Extract fields
        predicted_program = pred.get('predicted_program', '')
        gold_program = pred.get('gold_program', '')
        predicted_answer = pred.get('predicted_answer', '')
        gold_answer = pred.get('gold_answer', '')
        
        # Normalize programs
        pred_prog_norm = normalize_program(predicted_program)
        gold_prog_norm = normalize_program(gold_program)
        
        # Check program match
        program_match = (pred_prog_norm == gold_prog_norm)
        if program_match:
            program_correct += 1
        
        # Clean and round answers
        cleaned_pred_answer = clean_predicted_answer(predicted_answer, remove_assistant_suffix)
        pred_answer_rounded = safe_float_round(cleaned_pred_answer, 2)
        gold_answer_rounded = safe_float_round(gold_answer, 2)
        
        # Check answer match
        answer_match = False
        if pred_answer_rounded is not None and gold_answer_rounded is not None:
            answer_match = (pred_answer_rounded == gold_answer_rounded)
            if answer_match:
                answer_correct += 1
        
        # Store result
        results.append({
            'id': pred_id,
            'question': question,
            'predicted_program': predicted_program,
            'gold_program': gold_program,
            'program_match': program_match,
            'predicted_answer_raw': predicted_answer,
            'predicted_answer_cleaned': cleaned_pred_answer,
            'predicted_answer_rounded': pred_answer_rounded if pred_answer_rounded is not None else '',
            'gold_answer': gold_answer,
            'gold_answer_rounded': gold_answer_rounded if gold_answer_rounded is not None else '',
            'answer_match': answer_match
        })
    
    # Calculate accuracies
    program_accuracy = (program_correct / total_count * 100) if total_count > 0 else 0
    answer_accuracy = (answer_correct / total_count * 100) if total_count > 0 else 0
    
    # Print summary
    print(f"\nTotal predictions: {total_count}")
    print(f"Program matches: {program_correct}")
    print(f"Program accuracy: {program_accuracy:.2f}%")
    print(f"Answer matches: {answer_correct}")
    print(f"Answer accuracy: {answer_accuracy:.2f}%")
    print("="*60)
    
    # Save to CSV
    csv_file = output_csv
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        fieldnames = [
            'id', 'question', 
            'predicted_program', 'gold_program', 'program_match',
            'predicted_answer_raw', 'predicted_answer_cleaned', 
            'predicted_answer_rounded', 'gold_answer', 'gold_answer_rounded', 'answer_match'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"Detailed results saved to: {csv_file}")
    
    # Also save summary to a separate CSV
    summary_csv = str(output_csv).replace('.csv', '_summary.csv')
    with open(summary_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Model', model_name])
        writer.writerow(['Metric', 'Count', 'Accuracy (%)'])
        writer.writerow(['Total Predictions', total_count, ''])
        writer.writerow(['Program Correct', program_correct, f'{program_accuracy:.2f}'])
        writer.writerow(['Answer Correct', answer_correct, f'{answer_accuracy:.2f}'])
    
    print(f"Summary saved to: {summary_csv}\n")
    
    return {
        'model': model_name,
        'total': total_count,
        'program_correct': program_correct,
        'program_accuracy': program_accuracy,
        'answer_correct': answer_correct,
        'answer_accuracy': answer_accuracy
    }


def main():
    # Base directory
    predictions_dir = Path('/Users/abhivakil/Desktop/FinQA/FinQA-Mini-Project/results/predictions')
    
    # Model configurations: (filename, remove_assistant_suffix)
    models = [
        ('Meta-Llama-3-8B-Instruct_lora_predictions.json', True),  # Has assistant: suffix
        ('Mistral-7B-Instruct-v0.2_lora_predictions.json', False)  # No assistant: suffix
    ]
    
    all_results = []
    
    # Evaluate each model
    for model_file, remove_suffix in models:
        predictions_file = predictions_dir / model_file
        output_csv = predictions_dir / model_file.replace('_predictions.json', '_evaluation.csv')
        
        if predictions_file.exists():
            result = evaluate_predictions(predictions_file, output_csv, remove_suffix)
            all_results.append(result)
        else:
            print(f"⚠️  File not found: {predictions_file}")
    
    # Create a combined summary CSV
    if all_results:
        combined_csv = predictions_dir / 'combined_evaluation_summary.csv'
        with open(combined_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Model', 'Total Predictions', 'Program Correct', 'Program Accuracy (%)', 
                           'Answer Correct', 'Answer Accuracy (%)'])
            for result in all_results:
                writer.writerow([
                    result['model'],
                    result['total'],
                    result['program_correct'],
                    f"{result['program_accuracy']:.2f}",
                    result['answer_correct'],
                    f"{result['answer_accuracy']:.2f}"
                ])
        
        print(f"\n{'='*60}")
        print("COMBINED SUMMARY")
        print(f"{'='*60}")
        print(f"Combined summary saved to: {combined_csv}")
        print(f"{'='*60}\n")
    
    print("✅ All evaluations completed successfully!")


if __name__ == '__main__':
    main()

