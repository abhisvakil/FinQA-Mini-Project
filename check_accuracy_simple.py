#!/usr/bin/env python3
"""Simple script to check accuracy of predictions with symbolic program comparison"""

import json
import sys
from sympy import simplify

all_ops = ["add", "subtract", "multiply", "divide", "exp", "greater", "table_max",
           "table_min", "table_sum", "table_average"]


def program_tokenization(original_program):
    """Convert program string to token list"""
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


def equal_program(program1, program2):
    """
    Check if two programs are symbolically equal
    program1: gold (string)
    program2: pred (string)
    """
    try:
        # Tokenize both programs
        prog1_tokens = program_tokenization(program1)
        prog2_tokens = program_tokenization(program2)
        
        sym_map = {}
        
        prog1_tokens = prog1_tokens[:-1]  # remove EOF
        program1_str = "|".join(prog1_tokens)
        steps = program1_str.split(")")[:-1]
        
        sym_ind = 0
        step_dict_1 = {}
        
        # Build symbolic map from gold program
        for ind, step in enumerate(steps):
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
        
        # Validate structure
        for ind, token in enumerate(prog2_tokens):
            if ind % 4 == 0:
                if token.strip("(") not in all_ops:
                    return False
            if (ind + 1) % 4 == 0:
                if token != ")":
                    return False
        
        program2_str = "|".join(prog2_tokens)
        steps = program2_str.split(")")[:-1]
        
        for ind, step in enumerate(steps):
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
                return "( " + arg1_part + " + " + arg2_part + " )"
            elif op == "subtract":
                return "( " + arg1_part + " - " + arg2_part + " )"
            elif op == "multiply":
                return "( " + arg1_part + " * " + arg2_part + " )"
            elif op == "divide":
                return "( " + arg1_part + " / " + arg2_part + " )"
            elif op == "exp":
                return "( " + arg1_part + " ** " + arg2_part + " )"
            elif op == "greater":
                return "( " + arg1_part + " > " + arg2_part + " )"
            elif op == "less":
                return "( " + arg1_part + " < " + arg2_part + " )"
            return ""
        
        # Derive symbolic expressions
        steps1 = program1_str.split(")")[:-1]
        sym_prog1 = symbol_recur(steps1[-1], step_dict_1)
        
        steps2 = program2_str.split(")")[:-1]
        sym_prog2 = symbol_recur(steps2[-1], step_dict_2)
        
        # For boolean operations (greater/less), compare strings directly
        # For arithmetic operations, use sympy simplify
        if ">" in sym_prog1 or ">" in sym_prog2 or "<" in sym_prog1 or "<" in sym_prog2:
            return sym_prog1 == sym_prog2
        else:
            sym_prog1 = simplify(sym_prog1, evaluate=False)
            sym_prog2 = simplify(sym_prog2, evaluate=False)
            return sym_prog1 == sym_prog2
        
    except Exception as e:
        # If any error in symbolic comparison, fall back to False
        # Uncomment below for debugging:
        # print(f"ERROR in equal_program: {e}")
        # import traceback; traceback.print_exc()
        return False

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
    print("Usage: python check_accuracy_simple.py [path/to/predictions.json]")
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
    
    # Use symbolic program comparison
    program_match = equal_program(gold_prog, pred_prog)
    
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
    
    if i <= 10:  # Show first 10
        print(f"\n{i}. {pred['id']}")
        print(f"   Question: {pred['question'][:70]}...")
        print(f"   Program match: {'✓' if program_match else '✗'}")
        print(f"   Answer match: {'✓' if answer_match else '✗'}")
        if not program_match:
            print(f"   Predicted: {pred_prog}")
            print(f"   Gold:      {gold_prog}")

print("\n" + "=" * 80)
total = len(predictions)
print(f"Results: {correct_programs}/{total} correct programs ({correct_programs*100/total:.1f}%)")
print(f"         {correct_answers}/{total} correct answers ({correct_answers*100/total:.1f}%)")
print("=" * 80)
