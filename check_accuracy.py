#!/usr/bin/env python3
"""Quick script to check accuracy of predictions"""

import json
import sys

# Sample predictions from the output
predictions = [
  {
    "id": "ETR/2016/page_23.pdf-2",
    "question": "what is the net change in net revenue during 2015 for entergy corporation?",
    "predicted_program": "add(const_5735, 187), add(#0, 95), subtract(#0, 32), subtract(#0, 35), subtract(#0, 107), add(#0, 14)",
    "predicted_answer": "5829.0 - 32.0 - 35.0 + 14.0 = 5762.0",
    "gold_program": "subtract(5829, 5735)",
    "gold_answer": "94.0"
  },
  {
    "id": "INTC/2015/page_41.pdf-4",
    "question": "what percentage of total facilities as measured in square feet are leased?",
    "predicted_program": "divide(const_8_1, const_56_0), multiply(#0, 100)",
    "predicted_answer": "0.14516129",
    "gold_program": "divide(8.1, 56.0)",
    "gold_answer": "0.14464"
  },
  {
    "id": "ADI/2011/page_61.pdf-2",
    "question": "what is the percentage change in cash flow hedges in 2011 compare to the 2010?",
    "predicted_program": "subtract(const_20692, 2014)",
    "predicted_answer": "20488.0",
    "gold_program": "subtract(153.7, 139.9), divide(#0, 139.9)",
    "gold_answer": "0.09864"
  },
  {
    "id": "FIS/2010/page_70.pdf-2",
    "question": "what portion of total purchase price is related to stock awards?",
    "predicted_program": "divide(const_121.4, const_4187.8)",
    "predicted_answer": "0.028974418604651165",
    "gold_program": "divide(121.4, 4187.8)",
    "gold_answer": "0.02899"
  },
  {
    "id": "MAS/2017/page_27.pdf-2",
    "question": "what was the difference in percentage cumulative total shareholder return on masco common stock versus the s&p 500 index for the five year period ended 2017?",
    "predicted_program": "subtract(divide(const_318.46, const_206.49), 1)",
    "predicted_answer": "0.6152954594817461",
    "gold_program": "subtract(318.46, const_100), divide(#0, const_100), subtract(206.49, const_100), divide(#2, const_100), subtract(#1, #3)",
    "gold_answer": "1.1197"
  },
  {
    "id": "SYY/2006/page_71.pdf-1",
    "question": "what was the percentage change in total rental expense under operating leases from july 2 , 2005 to july 1 , 2006?",
    "predicted_program": "divide(const_25666000, const_33329000), multiply(#0, 100)",
    "predicted_answer": "0.76842157",
    "gold_program": "subtract(92710000, 86842000), divide(#0, 86842000)",
    "gold_answer": "0.06757"
  },
  {
    "id": "AES/2010/page_227.pdf-3",
    "question": "what percent of total recourse debt is current?",
    "predicted_program": "divide(const_497, const_4612), multiply(#0, 100)",
    "predicted_answer": "0.10335823585304683",
    "gold_program": "divide(463, 4612)",
    "gold_answer": "0.10039"
  },
  {
    "id": "GS/2015/page_188.pdf-2",
    "question": "what percentage of future minimum rental payments are due in 2018?",
    "predicted_program": "divide(const_301, sum(const_317, const_313, const_301, const_258, const_226, const_1160)) * 100",
    "predicted_answer": "0.011715729166666667",
    "gold_program": "divide(301, 2575)",
    "gold_answer": "0.11689"
  },
  {
    "id": "MO/2016/page_19.pdf-1",
    "question": "did altria outperform the s&p 500?",
    "predicted_program": "greater(286.61, 198.09)",
    "predicted_answer": "True",
    "gold_program": "greater(286.61, 198.09)",
    "gold_answer": "yes"
  },
  {
    "id": "CMCSA/2015/page_112.pdf-2",
    "question": "what was the change in unrecognized tax benefits from the end of 2014 to the end of 2015?",
    "predicted_program": "subtract(const_1171, const_1136)",
    "predicted_answer": "35.0",
    "gold_program": "subtract(1136, 1171)",
    "gold_answer": "-35.0"
  }
]

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
    if not program_match:
        print(f"   Predicted: {pred_prog}")
        print(f"   Gold:      {gold_prog}")
    if not answer_match:
        print(f"   Pred answer: {pred_ans}")
        print(f"   Gold answer: {gold_ans}")

print("\n" + "=" * 80)
print(f"Results: {correct_programs}/10 correct programs ({correct_programs*10}%)")
print(f"         {correct_answers}/10 correct answers ({correct_answers*10}%)")
print("=" * 80)
