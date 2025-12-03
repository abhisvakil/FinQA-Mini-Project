#!/usr/bin/env python3
"""
Evaluate predictions from ICL with Executor.
"""

import json
import sys
import csv


def normalize_program(program_str):
    """Normalize program string by removing all spaces."""
    if not program_str:
        return ""
    return ''.join(program_str.split())


def safe_float(value, decimals=2):
    """Safely convert to float and round."""
    try:
        return round(float(value), decimals)
    except (ValueError, TypeError):
        return None


def evaluate_executor_predictions(predictions_file, output_csv=None):
    """Evaluate executor predictions with detailed statistics."""
    # Load predictions
    with open(predictions_file, 'r') as f:
        predictions = json.load(f)
    
    print("="*70)
    print(f"Evaluating ICL with Executor: {predictions_file}")
    print("="*70)
    print(f"Total predictions: {len(predictions)}\n")
    
    # Statistics
    total = len(predictions)
    program_correct = 0
    answer_correct = 0
    executor_success_count = 0
    
    retry_stats = {
        1: 0,  # First attempt
        2: 0,  # One retry
        3: 0   # Two retries
    }
    
    # Detailed results
    results_data = []
    
    print("="*70)
    print("Detailed Results")
    print("="*70)
    
    for i, pred in enumerate(predictions, 1):
        pred_prog = normalize_program(pred.get('predicted_program', ''))
        gold_prog = normalize_program(pred.get('gold_program', ''))
        
        # Use executor answer if available and successful
        if pred.get('executor_success') and pred.get('executor_answer'):
            pred_ans = safe_float(pred['executor_answer'])
        else:
            pred_ans = safe_float(pred.get('predicted_answer', ''))
        
        gold_ans = safe_float(pred.get('gold_answer', ''))
        
        prog_match = (pred_prog == gold_prog)
        ans_match = (pred_ans is not None and gold_ans is not None and pred_ans == gold_ans)
        
        if prog_match:
            program_correct += 1
        if ans_match:
            answer_correct += 1
        if pred.get('executor_success'):
            executor_success_count += 1
        
        # Track retry attempts
        attempts = pred.get('attempts', 1)
        if attempts in retry_stats:
            retry_stats[attempts] += 1
        
        # Print result
        print(f"\n{'─'*70}")
        print(f"Example {i}:")
        print(f"  ID: {pred.get('id', 'N/A')}")
        print(f"  Question: {pred.get('question', 'N/A')[:65]}...")
        print(f"  Attempts: {attempts}")
        print(f"  Executor Success: {'✅' if pred.get('executor_success') else '❌'}")
        print(f"\n  Predicted Program: {pred.get('predicted_program', 'N/A')[:60]}...")
        print(f"  Gold Program:      {pred.get('gold_program', 'N/A')[:60]}...")
        print(f"  Program Match: {'✅' if prog_match else '❌'}")
        
        if pred.get('executor_answer'):
            print(f"\n  Executor Answer: {pred.get('executor_answer', 'N/A')}")
        print(f"  Predicted Answer: {pred.get('predicted_answer', 'N/A')}")
        print(f"  Gold Answer:      {pred.get('gold_answer', 'N/A')}")
        print(f"  Answer Match: {'✅' if ans_match else '❌'}")
        
        # Show retry history if there were retries
        if attempts > 1:
            print(f"\n  Retry History:")
            for attempt_info in pred.get('attempt_history', []):
                attempt_num = attempt_info.get('attempt', 0)
                exec_success = attempt_info.get('executor_result', {}).get('success', False)
                print(f"    Attempt {attempt_num + 1}: {'✅ Success' if exec_success else '❌ Failed'}")
                if not exec_success:
                    error = attempt_info.get('executor_result', {}).get('error', 'Unknown')
                    print(f"      Error: {error[:50]}...")
        
        # Store for CSV
        results_data.append({
            'id': pred.get('id', ''),
            'question': pred.get('question', ''),
            'attempts': attempts,
            'executor_success': pred.get('executor_success', False),
            'predicted_program': pred.get('predicted_program', ''),
            'gold_program': pred.get('gold_program', ''),
            'program_match': prog_match,
            'executor_answer': pred.get('executor_answer', ''),
            'predicted_answer': pred.get('predicted_answer', ''),
            'gold_answer': pred.get('gold_answer', ''),
            'answer_match': ans_match
        })
    
    # Calculate accuracies
    prog_acc = (program_correct / total * 100) if total > 0 else 0
    ans_acc = (answer_correct / total * 100) if total > 0 else 0
    exec_success_rate = (executor_success_count / total * 100) if total > 0 else 0
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    print(f"Total Predictions:        {total}")
    print(f"\nProgram Accuracy:         {program_correct}/{total} ({prog_acc:.2f}%)")
    print(f"Answer Accuracy:          {answer_correct}/{total} ({ans_acc:.2f}%)")
    print(f"Executor Success Rate:    {executor_success_count}/{total} ({exec_success_rate:.2f}%)")
    
    print(f"\n{'─'*70}")
    print("Retry Statistics:")
    print(f"{'─'*70}")
    for attempts, count in sorted(retry_stats.items()):
        percentage = (count / total * 100) if total > 0 else 0
        retry_label = f"{attempts} attempt(s)"
        print(f"  {retry_label:20s}: {count:4d} ({percentage:5.1f}%)")
    
    print("="*70)
    
    # Save to CSV if requested
    if output_csv:
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            fieldnames = [
                'id', 'question', 'attempts', 'executor_success',
                'predicted_program', 'gold_program', 'program_match',
                'executor_answer', 'predicted_answer', 'gold_answer', 'answer_match'
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results_data)
        
        print(f"\n💾 Detailed results saved to: {output_csv}")
        
        # Save summary
        summary_csv = output_csv.replace('.csv', '_summary.csv')
        with open(summary_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Metric', 'Count', 'Percentage'])
            writer.writerow(['Total Predictions', total, ''])
            writer.writerow(['Program Correct', program_correct, f'{prog_acc:.2f}%'])
            writer.writerow(['Answer Correct', answer_correct, f'{ans_acc:.2f}%'])
            writer.writerow(['Executor Success', executor_success_count, f'{exec_success_rate:.2f}%'])
            writer.writerow(['', '', ''])
            writer.writerow(['Retry Distribution', '', ''])
            for attempts, count in sorted(retry_stats.items()):
                percentage = (count / total * 100) if total > 0 else 0
                writer.writerow([f'{attempts} attempts', count, f'{percentage:.2f}%'])
        
        print(f"💾 Summary saved to: {summary_csv}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python evaluate_executor_predictions.py <predictions_file> [output_csv]")
        print("Example: python evaluate_executor_predictions.py results/icl_executor_predictions.json results/executor_eval.csv")
        sys.exit(1)
    
    predictions_file = sys.argv[1]
    output_csv = sys.argv[2] if len(sys.argv) > 2 else None
    
    evaluate_executor_predictions(predictions_file, output_csv)


if __name__ == '__main__':
    main()

