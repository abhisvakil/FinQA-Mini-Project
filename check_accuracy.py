#!/usr/bin/env python3
"""Quick script to check accuracy of predictions"""

import json
import sys

# Load predictions from file
if len(sys.argv) > 1:
    pred_file = sys.argv[1]
else:
    # Default to latest test results
    pred_file = 'results_test/Mistral-7B-Instruct-v0.2_icl_predictions_latest.json'

try:
    with open(pred_file, 'r') as f:
        predictions = json.load(f)
    print(f"Loaded {len(predictions)} predictions from: {pred_file}")
except FileNotFoundError:
    print(f"Error: File not found: {pred_file}")
    print("Usage: python check_accuracy.py [path/to/predictions.json]")
    sys.exit(1)

print("Analyzing predictions:")
print("=" * 80)

correct_programs = 0
correct_answers = 0

for i, pred in enumerate(predictions, 1):
    pred_prog = pred['predicted_program'].strip()
    gold_prog = pred['gold_program'].strip()
    
    pred_ans = str(pred['predicted_answer']).strip()
    gold_ans = str(pred['gold_answer']).strip()
    
    # Normalize programs for comparison (remove spaces, lowercase)
    pred_prog_norm = pred_prog.replace(' ', '').lower()
    gold_prog_norm = gold_prog.replace(' ', '').lower()
    
    program_match = pred_prog_norm == gold_prog_norm
    
    # Check answer (round to 2 decimals for comparison)
    pred_rounded = None
    gold_rounded = None
    try:
        pred_num = float(pred_ans.split()[0]) if pred_ans else None
        gold_num = float(gold_ans)
        # Round both to 2 decimal places
        if pred_num is not None:
            pred_rounded = round(pred_num, 2)
            gold_rounded = round(gold_num, 2)
            answer_match = pred_rounded == gold_rounded
        else:
            answer_match = False
    except (ValueError, AttributeError):
        # For non-numeric answers (like True/yes, False/no)
        pred_lower = pred_ans.lower().strip()
        gold_lower = gold_ans.lower().strip()
        
        # Handle boolean equivalents
        true_values = ['true', 'yes', '1']
        false_values = ['false', 'no', '0']
        
        if pred_lower in true_values and gold_lower in true_values:
            answer_match = True
        elif pred_lower in false_values and gold_lower in false_values:
            answer_match = True
        else:
            answer_match = pred_lower in gold_lower or gold_lower in pred_lower
    
    if program_match:
        correct_programs += 1
    
    if answer_match:
        correct_answers += 1
    
    print(f"\n{i}. {pred['id']}")
    print(f"   Question: {pred['question'][:70]}...")
    print(f"   Program match: {'✓' if program_match else '✗'}")
    print(f"   Answer match: {'✓' if answer_match else '✗'}")
    if pred_rounded is not None and gold_rounded is not None:
        print(f"   Pred: {pred_ans} → {pred_rounded} | Gold: {gold_ans} → {gold_rounded}")
    if not program_match:
        print(f"   Predicted: {pred_prog}")
        print(f"   Gold:      {gold_prog}")
    if not answer_match and (pred_rounded is None or gold_rounded is None):
        print(f"   Pred answer: {pred_ans}")
        print(f"   Gold answer: {gold_ans}")

print("\n" + "=" * 80)
total = len(predictions)
print(f"Results: {correct_programs}/{total} correct programs ({correct_programs*100//total}%)")
print(f"         {correct_answers}/{total} correct answers ({correct_answers*100//total}%)")
print("=" * 80)
