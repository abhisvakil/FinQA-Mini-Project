"""
Evaluation metrics for FinQA task.
"""

import json
import sys
from typing import List, Dict, Any
from src.executor import ProgramExecutor


def execution_accuracy(predictions: List[Dict], gold_data: List[Dict], tolerance: float = 1e-6) -> float:
    """
    Calculate execution accuracy.
    
    Args:
        predictions: List of predictions with 'id' and 'predicted' (program tokens)
        gold_data: List of gold examples with 'id', 'qa' containing 'exe_ans' and table/text
        tolerance: Tolerance for floating point comparison
        
    Returns:
        Execution accuracy (0-1)
    """
    # Create mapping from id to gold data
    gold_map = {ex['id']: ex for ex in gold_data}
    
    correct = 0
    total = 0
    errors = []
    
    for pred in predictions:
        pred_id = pred['id']
        if pred_id not in gold_map:
            continue
        
        gold_ex = gold_map[pred_id]
        total += 1
        
        try:
            # Get context for execution
            table = gold_ex.get('table', [])
            pre_text = gold_ex.get('pre_text', [])
            post_text = gold_ex.get('post_text', [])
            
            # Execute predicted program
            executor = ProgramExecutor(table, pre_text, post_text)
            pred_result = executor.execute(pred['predicted'])
            
            # Get gold answer
            gold_ans = gold_ex.get('qa', {}).get('exe_ans', '')
            
            # Compare results
            if isinstance(pred_result, (int, float)) and isinstance(gold_ans, (int, float, str)):
                try:
                    gold_float = float(str(gold_ans).replace(',', '').replace('$', ''))
                    if abs(pred_result - gold_float) < tolerance:
                        correct += 1
                    else:
                        errors.append({
                            'id': pred_id,
                            'predicted': pred_result,
                            'gold': gold_float,
                            'program': pred['predicted']
                        })
                except (ValueError, TypeError):
                    errors.append({
                        'id': pred_id,
                        'error': 'Cannot parse gold answer',
                        'program': pred['predicted']
                    })
            elif str(pred_result) == str(gold_ans):
                correct += 1
            else:
                errors.append({
                    'id': pred_id,
                    'predicted': pred_result,
                    'gold': gold_ans,
                    'program': pred['predicted']
                })
        
        except Exception as e:
            errors.append({
                'id': pred_id,
                'error': str(e),
                'program': pred['predicted']
            })
    
    accuracy = correct / total if total > 0 else 0.0
    return accuracy, errors


def program_accuracy(predictions: List[Dict], gold_data: List[Dict]) -> float:
    """
    Calculate program accuracy (exact match).
    
    Args:
        predictions: List of predictions with 'id' and 'predicted' (program tokens)
        gold_data: List of gold examples with 'id' and 'qa' containing 'program'
        
    Returns:
        Program accuracy (0-1)
    """
    gold_map = {ex['id']: ex for ex in gold_data}
    
    correct = 0
    total = 0
    
    for pred in predictions:
        pred_id = pred['id']
        if pred_id not in gold_map:
            continue
        
        total += 1
        gold_program = gold_map[pred_id].get('qa', {}).get('program', [])
        pred_program = pred['predicted']
        
        # Remove EOF token if present
        if pred_program and pred_program[-1] == 'EOF':
            pred_program = pred_program[:-1]
        
        if pred_program == gold_program:
            correct += 1
    
    accuracy = correct / total if total > 0 else 0.0
    return accuracy


def evaluate(predictions_file: str, gold_file: str) -> Dict[str, Any]:
    """
    Evaluate predictions against gold data.
    
    Args:
        predictions_file: Path to predictions JSON file
        gold_file: Path to gold data JSON file
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Load predictions
    with open(predictions_file, 'r', encoding='utf-8') as f:
        predictions = json.load(f)
    
    # Load gold data
    with open(gold_file, 'r', encoding='utf-8') as f:
        gold_data = json.load(f)
    
    # Calculate metrics
    exec_acc, exec_errors = execution_accuracy(predictions, gold_data)
    prog_acc = program_accuracy(predictions, gold_data)
    
    results = {
        'execution_accuracy': exec_acc,
        'program_accuracy': prog_acc,
        'num_examples': len(predictions),
        'execution_errors': exec_errors[:10]  # First 10 errors for inspection
    }
    
    return results


def main():
    """Command-line interface for evaluation."""
    if len(sys.argv) != 3:
        print("Usage: python evaluate.py <predictions.json> <gold.json>")
        sys.exit(1)
    
    predictions_file = sys.argv[1]
    gold_file = sys.argv[2]
    
    results = evaluate(predictions_file, gold_file)
    
    print("=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"Number of examples: {results['num_examples']}")
    print(f"Execution Accuracy: {results['execution_accuracy']:.4f} ({results['execution_accuracy']*100:.2f}%)")
    print(f"Program Accuracy: {results['program_accuracy']:.4f} ({results['program_accuracy']*100:.2f}%)")
    print()
    
    if results['execution_errors']:
        print("Sample Errors:")
        print("-" * 60)
        for i, error in enumerate(results['execution_errors'][:5], 1):
            print(f"Error {i}:")
            print(f"  ID: {error['id']}")
            if 'error' in error:
                print(f"  Error: {error['error']}")
            else:
                print(f"  Predicted: {error.get('predicted', 'N/A')}")
                print(f"  Gold: {error.get('gold', 'N/A')}")
            print(f"  Program: {' '.join(error['program'][:10])}...")
            print()


if __name__ == "__main__":
    main()

