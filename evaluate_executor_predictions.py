#!/usr/bin/env python3
"""
Evaluate predictions from ICL with Executor.
"""

import json
import sys
import csv
from sympy import simplify


all_ops = [
    "add",
    "subtract",
    "multiply",
    "divide",
    "exp",
    "greater",
    "table_max",
    "table_min",
    "table_sum",
    "table_average",
]


def normalize_program(program_str):
    """Normalize program string by removing all spaces."""
    if not program_str:
        return ""
    return ''.join(program_str.split())


def program_tokenization(original_program):
    """Convert program string to token list (FinQA-style)."""
    original_program = original_program.split(', ')
    program = []
    for tok in original_program:
        cur_tok = ''
        for c in tok:
            if c == ')':
                if cur_tok != '':
                    program.append(cur_tok)
                    cur_tok = ''
                program.append(c)
            elif c == '(':
                if cur_tok != '':
                    program.append(cur_tok)
                    cur_tok = ''
                program.append(c)
            elif c == ' ':
                if cur_tok != '':
                    program.append(cur_tok)
                    cur_tok = ''
            else:
                cur_tok += c
        if cur_tok != '':
            program.append(cur_tok)
    program.append('EOF')
    return program


def equal_program(program_gold: str, program_pred: str) -> bool:
    """
    Symbolically compare two programs for equivalence.
    This mirrors the logic in check_accuracy_simple.py.
    """
    program_gold = (program_gold or "").strip()
    program_pred = (program_pred or "").strip()

    # Quick exact match
    if program_gold == program_pred:
        return True

    # Empty or obviously bad predicted program
    if not program_pred:
        return False

    try:
        # Tokenize both programs
        prog1_tokens = program_tokenization(program_gold)
        prog2_tokens = program_tokenization(program_pred)

        sym_map = {}

        # Build symbolic map from gold program
        prog1_tokens = prog1_tokens[:-1]  # remove EOF
        program1_str = "|".join(prog1_tokens)
        steps1 = program1_str.split(")")[:-1]

        sym_ind = 0
        step_dict_1 = {}

        for ind, step in enumerate(steps1):
            step = step.strip()
            if len(step.split("(")) > 2:
                return False

            op = step.split("(")[0].strip("|").strip()
            args = step.split("(")[1].strip("|").strip()

            arg1 = args.split("|")[0].strip()
            arg2 = args.split("|")[1].strip()

            step_dict_1[ind] = step

            if "table" in op:
                if step not in sym_map:
                    sym_map[step] = "a" + str(sym_ind)
                    sym_ind += 1
            else:
                if "#" not in arg1:
                    if arg1 not in sym_map:
                        sym_map[arg1] = "a" + str(sym_ind)
                        sym_ind += 1
                if "#" not in arg2:
                    if arg2 not in sym_map:
                        sym_map[arg2] = "a" + str(sym_ind)
                        sym_ind += 1

        # Check predicted program structure
        step_dict_2 = {}
        prog2_tokens = prog2_tokens[:-1]  # remove EOF

        # Validate structure: op ( arg1 arg2 )
        for ind, token in enumerate(prog2_tokens):
            if ind % 4 == 0:
                if token.strip("(") not in all_ops:
                    return False
            if (ind + 1) % 4 == 0:
                if token != ")":
                    return False

        program2_str = "|".join(prog2_tokens)
        steps2 = program2_str.split(")")[:-1]

        for ind, step in enumerate(steps2):
            step = step.strip()
            if len(step.split("(")) > 2:
                return False

            op = step.split("(")[0].strip("|").strip()
            args = step.split("(")[1].strip("|").strip()

            arg1 = args.split("|")[0].strip()
            arg2 = args.split("|")[1].strip()

            step_dict_2[ind] = step

            if "table" in op:
                if step not in sym_map:
                    return False
            else:
                if "#" not in arg1:
                    if arg1 not in sym_map:
                        return False
                else:
                    if int(arg1.strip("#")) >= ind:
                        return False

                if "#" not in arg2:
                    if arg2 not in sym_map:
                        return False
                else:
                    if int(arg2.strip("#")) >= ind:
                        return False

        def symbol_recur(step, step_dict):
            step = step.strip()
            op = step.split("(")[0].strip("|").strip()
            args = step.split("(")[1].strip("|").strip()

            arg1 = args.split("|")[0].strip()
            arg2 = args.split("|")[1].strip()

            if "table" in op:
                return sym_map[step]

            if "#" in arg1:
                arg1_ind = int(arg1.replace("#", ""))
                arg1_part = symbol_recur(step_dict[arg1_ind], step_dict)
            else:
                arg1_part = sym_map[arg1]

            if "#" in arg2:
                arg2_ind = int(arg2.replace("#", ""))
                arg2_part = symbol_recur(step_dict[arg2_ind], step_dict)
            else:
                arg2_part = sym_map[arg2]

            if op == "add":
                return f"( {arg1_part} + {arg2_part} )"
            elif op == "subtract":
                return f"( {arg1_part} - {arg2_part} )"
            elif op == "multiply":
                return f"( {arg1_part} * {arg2_part} )"
            elif op == "divide":
                return f"( {arg1_part} / {arg2_part} )"
            elif op == "exp":
                return f"( {arg1_part} ** {arg2_part} )"
            elif op == "greater":
                return f"( {arg1_part} > {arg2_part} )"
            elif op == "less":
                return f"( {arg1_part} < {arg2_part} )"
            return ""

        sym_prog1 = symbol_recur(steps1[-1], step_dict_1)
        sym_prog2 = symbol_recur(steps2[-1], step_dict_2)

        # For boolean operations, compare strings directly
        if any(tok in sym_prog1 for tok in [">", "<"]) or any(
            tok in sym_prog2 for tok in [">", "<"]
        ):
            return sym_prog1 == sym_prog2

        # For arithmetic, compare via sympy
        sym_prog1_s = simplify(sym_prog1, evaluate=False)
        sym_prog2_s = simplify(sym_prog2, evaluate=False)
        return sym_prog1_s == sym_prog2_s

    except Exception:
        # If anything goes wrong, fall back to False
        return False


def compare_answers(pred_ans: str, gold_ans: str) -> bool:
    """
    Compare answers using the same tolerance as check_accuracy_simple.py:
    - numeric: compare floats rounded to 2 decimals
    - non-numeric: relaxed boolean/string matching
    """
    pred_ans = (pred_ans or "").strip()
    gold_ans = (gold_ans or "").strip()

    try:
        pred_num = float(pred_ans.split()[0]) if pred_ans else None
        gold_num = float(gold_ans)
        if pred_num is not None:
            return round(pred_num, 2) == round(gold_num, 2)
        return False
    except (ValueError, AttributeError):
        pred_lower = pred_ans.lower()
        gold_lower = gold_ans.lower()

        true_values = ['true', 'yes', '1']
        false_values = ['false', 'no', '0']

        if pred_lower in true_values and gold_lower in true_values:
            return True
        if pred_lower in false_values and gold_lower in false_values:
            return True

        # Fallback: substring match
        return pred_lower in gold_lower or gold_lower in pred_lower


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
        # Use raw programs for symbolic comparison (no space stripping)
        pred_prog_raw = pred.get('predicted_program', '')
        gold_prog_raw = pred.get('gold_program', '')
        
        # Normalized versions (kept for potential debugging / CSV)
        pred_prog = normalize_program(pred_prog_raw)
        gold_prog = normalize_program(gold_prog_raw)
        
        # Use executor answer if available and successful (raw strings)
        if pred.get('executor_success') and pred.get('executor_answer'):
            pred_ans_raw = str(pred['executor_answer'])
        else:
            pred_ans_raw = str(pred.get('predicted_answer', ''))
        
        gold_ans_raw = str(pred.get('gold_answer', ''))
        
        # Program match via symbolic comparison
        prog_match = equal_program(gold_prog_raw, pred_prog_raw)

        # Answer match using tolerant comparison
        ans_match = compare_answers(pred_ans_raw, gold_ans_raw)
        
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

