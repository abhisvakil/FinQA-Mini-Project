#!/usr/bin/env python3
"""
Script to evaluate LoRA predictions by comparing:
1. predicted_program vs gold_program
2. predicted_answer vs gold_answer (after cleaning and rounding)
"""

import json
import re
import csv
from pathlib import Path


def clean_predicted_answer(answer_str):
    """
    Remove 'assistant:...' suffix from predicted answer.
    Also extract the numeric value before 'assistant:' if present.
    """
    if not answer_str:
        return ""
    
    # Remove everything after 'assistant:' (case insensitive)
    cleaned = re.split(r'assistant:', str(answer_str), flags=re.IGNORECASE)[0]
    cleaned = cleaned.strip()
    
    return cleaned


def normalize_program(program_str):
    """
    Normalize program string by removing extra spaces for comparison.
    Also normalize commas and parentheses spacing.
    """
    if not program_str:
        return ""
    # Remove all spaces and then compare to be more robust
    # This handles cases like "5 8 2 9,   5 7 3 5" vs "5 8 2 9 ,   5 7 3 5"
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


def evaluate_predictions(predictions_file, output_csv):
    """
    Evaluate predictions and save results to CSV.
    """
    # Load predictions
    with open(predictions_file, 'r') as f:
        predictions = json.load(f)
    
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
        cleaned_pred_answer = clean_predicted_answer(predicted_answer)
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
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Total predictions: {total_count}")
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
    
    print(f"\nDetailed results saved to: {csv_file}")
    
    # Also save summary to a separate CSV
    summary_csv = str(output_csv).replace('.csv', '_summary.csv')
    with open(summary_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Count', 'Accuracy (%)'])
        writer.writerow(['Total Predictions', total_count, ''])
        writer.writerow(['Program Correct', program_correct, f'{program_accuracy:.2f}'])
        writer.writerow(['Answer Correct', answer_correct, f'{answer_accuracy:.2f}'])
    
    print(f"Summary saved to: {summary_csv}")
    
    return {
        'total': total_count,
        'program_correct': program_correct,
        'program_accuracy': program_accuracy,
        'answer_correct': answer_correct,
        'answer_accuracy': answer_accuracy
    }


def main():
    # Paths
    predictions_file = Path('/Users/abhivakil/Desktop/FinQA/FinQA-Mini-Project/results/predictions/Meta-Llama-3-8B-Instruct_lora_predictions.json')
    output_csv = Path('/Users/abhivakil/Desktop/FinQA/FinQA-Mini-Project/results/predictions/Meta-Llama-3-8B-Instruct_lora_evaluation.csv')
    
    # Evaluate
    results = evaluate_predictions(predictions_file, output_csv)
    
    print("\n✅ Evaluation completed successfully!")


if __name__ == '__main__':
    main()

