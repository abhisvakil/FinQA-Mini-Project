#!/usr/bin/env python3
"""Script to check accuracy of predictions using official FinQA evaluation"""

import json
import sys
import os

# Import from finqa_evaluate module in project root
sys.path.insert(0, os.path.dirname(__file__))
from finqa_evaluate import evaluate_result, program_tokenization

def convert_predictions_format(predictions, output_file, test_data_dict=None):
    """
    Convert our prediction format to the format expected by evaluate.py
    Our format: {id, question, predicted_program, predicted_answer, gold_program, gold_answer, raw_output}
    Expected format: {id, predicted: [program tokens]}
    
    If test_data_dict is None, we'll create a mock test file from gold data in predictions
    """
    converted = []
    test_data = []
    
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
                # No commas - just op, (, args..., )
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
        
        # If no test_data_dict provided, create test data from predictions
        if test_data_dict is None:
            test_data.append({
                'id': pred['id'],
                'qa': {
                    'program': pred['gold_program'],
                    'exe_ans': pred['gold_answer']
                },
                'table': []  # Empty table since we can't execute without it
            })
    
    # Write converted predictions
    with open(output_file, 'w') as f:
        json.dump(converted, f, indent=2)
    
    # If we created test_data, write it too
    if test_data_dict is None and test_data:
        test_file = output_file.replace('_converted.json', '_test.json')
        with open(test_file, 'w') as f:
            json.dump(test_data, f, indent=2)
        return output_file, test_file
    
    return output_file, None

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
    test_file = None  # Will use gold data from predictions file

try:
    with open(pred_file, 'r') as f:
        predictions = json.load(f)
    print(f"Loaded {len(predictions)} predictions from: {pred_file}")
except FileNotFoundError:
    print(f"Error: File not found: {pred_file}")
    print("Usage: python check_accuracy.py [path/to/predictions.json] [path/to/test.json]")
    sys.exit(1)

# Check if test file exists, if not we'll use gold data from predictions
test_data_dict = None
if test_file and os.path.exists(test_file):
    print(f"Using test data from: {test_file}")
    with open(test_file, 'r') as f:
        test_data = json.load(f)
        test_data_dict = {item['id']: item for item in test_data}
else:
    print("No test file provided - using gold data from predictions file")
    print("Note: Cannot execute programs without table data, only comparing program structure")

print("=" * 80)
print("Converting predictions to official format...")
print("=" * 80)

# Convert predictions to expected format
converted_file = pred_file.replace('.json', '_converted.json')
result = convert_predictions_format(predictions, converted_file, test_data_dict)

if isinstance(result, tuple):
    converted_file, generated_test_file = result
    if generated_test_file:
        test_file = generated_test_file
        print(f"Generated test file from predictions: {test_file}")
else:
    converted_file = result

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
