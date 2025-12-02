#!/usr/bin/env python3
"""
Wrapper script to evaluate predictions using the official FinQA evaluator.
Converts our prediction format to the format expected by evaluate.py
Usage: python evaluate_predictions.py <predictions_json> <test_json>
"""

import json
import sys
import os

def convert_predictions_format(predictions_file, output_file):
    """
    Convert our prediction format to the format expected by evaluate.py
    Our format: {id, question, predicted_program, predicted_answer, gold_program, gold_answer, raw_output}
    Expected format: {id, predicted: [program tokens]}
    """
    with open(predictions_file, 'r') as f:
        predictions = json.load(f)
    
    converted = []
    for pred in predictions:
        # Convert program string to token list
        # Our format: "subtract(5829, 5735)" or "add(1, 2), subtract(#0, 3)"
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
    
    print(f"Converted {len(converted)} predictions")
    print(f"Saved to: {output_file}")
    return output_file

def main():
    if len(sys.argv) < 3:
        print("Usage: python evaluate_predictions.py <predictions_json> <test_json>")
        print("Example: python evaluate_predictions.py results_test/Mistral-7B-Instruct-v0.2_icl_predictions_latest.json data/simplified/test.json")
        sys.exit(1)
    
    predictions_file = sys.argv[1]
    test_file = sys.argv[2]
    
    # Check files exist
    if not os.path.exists(predictions_file):
        print(f"Error: Predictions file not found: {predictions_file}")
        sys.exit(1)
    
    if not os.path.exists(test_file):
        print(f"Error: Test file not found: {test_file}")
        sys.exit(1)
    
    # Convert predictions to expected format
    converted_file = predictions_file.replace('.json', '_converted.json')
    convert_predictions_format(predictions_file, converted_file)
    
    # Import and run the official evaluator
    print("\n" + "="*80)
    print("Running official FinQA evaluation...")
    print("="*80 + "\n")
    
    # Import the evaluate module
    sys.path.insert(0, os.path.dirname(__file__))
    from evaluate import evaluate_result
    
    # Run evaluation
    exe_acc, prog_acc = evaluate_result(converted_file, test_file)
    
    print("\n" + "="*80)
    print(f"Final Results:")
    print(f"  Execution Accuracy: {exe_acc:.4f} ({exe_acc*100:.2f}%)")
    print(f"  Program Accuracy: {prog_acc:.4f} ({prog_acc*100:.2f}%)")
    print("="*80)
    
    # Clean up converted file
    # os.remove(converted_file)

if __name__ == '__main__':
    main()
