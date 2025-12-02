#!/usr/bin/env python3
"""Script to check accuracy of predictions using official FinQA evaluation"""

import json
import sys
import os

# Add src directory to path for evaluate module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from evaluate import evaluate_result, program_tokenization

def convert_predictions_format(predictions, output_file):
    """
    Convert our prediction format to the format expected by evaluate.py
    Our format: {id, question, predicted_program, predicted_answer, gold_program, gold_answer, raw_output}
    Expected format: {id, predicted: [program tokens]}
    """
    converted = []
    for pred in predictions:
        # Convert program string to token list
        program_str = pred['predicted_program']
        
        # Split by comma to get individual operations
        operations = [op.strip() for op in program_str.split(',') if op.strip()]
        
        # Tokenize each operation
        tokens = []
        for op in operations:
            # Parse operation like "subtract(5829, 5735)"
            if '(' in op and ')' in op:
                func_name = op.split('(')[0].strip()
                args = op.split('(')[1].rstrip(')').strip()
                arg_list = [arg.strip() for arg in args.split(',')]
                
                # Build token list: ["subtract", "(", "5829", "5735", ")"]
                tokens.append(func_name)
                tokens.append('(')
                tokens.extend(arg_list)
                tokens.append(')')
            else:
                # Handle cases where it might just be a number or reference
                if op:
                    tokens.append(op)
        
        # Add EOF token
        tokens.append('EOF')
        
        converted.append({
            'id': pred['id'],
            'predicted': tokens
        })
    
    # Write converted predictions
    with open(output_file, 'w') as f:
        json.dump(converted, f, indent=2)
    
    return output_file

# Load predictions from file
if len(sys.argv) > 1:
    pred_file = sys.argv[1]
else:
    # Default to latest test results
    pred_file = 'results_test/Mistral-7B-Instruct-v0.2_icl_predictions_latest.json'

# Get test file path
if len(sys.argv) > 2:
    test_file = sys.argv[2]
else:
    test_file = 'data/simplified/test.json'

try:
    with open(pred_file, 'r') as f:
        predictions = json.load(f)
    print(f"Loaded {len(predictions)} predictions from: {pred_file}")
except FileNotFoundError:
    print(f"Error: File not found: {pred_file}")
    print("Usage: python check_accuracy.py [path/to/predictions.json] [path/to/test.json]")
    sys.exit(1)

# Check if test file exists
if not os.path.exists(test_file):
    print(f"Error: Test file not found: {test_file}")
    sys.exit(1)

print("=" * 80)
print("Converting predictions to official format...")
print("=" * 80)

# Convert predictions to expected format
converted_file = pred_file.replace('.json', '_converted.json')
convert_predictions_format(predictions, converted_file)

print("\n" + "="*80)
print("Running official FinQA evaluation...")
print("="*80 + "\n")

# Run evaluation
try:
    exe_acc, prog_acc = evaluate_result(converted_file, test_file)
    
    print("\n" + "="*80)
    print(f"Final Results:")
    print(f"  Total predictions: {len(predictions)}")
    print(f"  Execution Accuracy: {exe_acc:.4f} ({exe_acc*100:.2f}%)")
    print(f"  Program Accuracy: {prog_acc:.4f} ({prog_acc*100:.2f}%)")
    print("="*80)
except Exception as e:
    print(f"Error during evaluation: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
