#!/usr/bin/env python3
"""
Evaluate LoRA predictions by:
- Checking program equality (like check_accuracy_simple / evaluate_lora_predictions)
- For examples where the program matches the gold program, executing that program
  with a simple FinQA-style executor to compute a numeric/boolean answer
- Comparing the executed answer to the gold answer

Usage:
    python check_lora_program_execution.py \
        --predictions results/predictions/Mistral-7B-Instruct-v0.2_lora_predictions.json \
        --output_csv results/predictions/Mistral-7B-Instruct-v0.2_lora_program_exec_eval.csv
"""

import argparse
import json
import csv
from pathlib import Path
from typing import Tuple

from program_executor_tool import execute_program


def normalize_program(program_str: str) -> str:
    """
    Normalize program string by removing all whitespace.

    This mirrors the logic used in evaluate_lora_predictions.py so that
    we get the same notion of "program match".
    """
    if not program_str:
        return ""
    return "".join(str(program_str).split())


def compare_answers(exec_answer, gold_answer: str) -> Tuple[bool, str, str]:
    """
    Compare executed answer vs gold answer.

    Returns:
        (match, exec_str, gold_str)
        - match: bool, True if answers are considered equal
        - exec_str: string representation of executed answer
        - gold_str: string representation of gold answer
    """
    exec_str = "" if exec_answer is None else str(exec_answer)
    gold_str = "" if gold_answer is None else str(gold_answer)

    # Try numeric comparison first (rounded to 2 decimals)
    try:
        exec_num = float(str(exec_answer).split()[0])
        gold_num = float(gold_str)
        return round(exec_num, 2) == round(gold_num, 2), exec_str, gold_str
    except (ValueError, TypeError):
        # Fallback to simple boolean/string comparison
        e = exec_str.strip().lower()
        g = gold_str.strip().lower()

        true_vals = {"true", "yes", "1"}
        false_vals = {"false", "no", "0"}

        if e in true_vals and g in true_vals:
            return True, exec_str, gold_str
        if e in false_vals and g in false_vals:
            return True, exec_str, gold_str

        return e == g, exec_str, gold_str


def evaluate_lora_program_execution(predictions_file: Path, output_csv: Path):
    """Main evaluation routine."""
    with predictions_file.open("r") as f:
        predictions = json.load(f)

    print(f"Loaded {len(predictions)} predictions from {predictions_file}")

    total = len(predictions)
    total_prog_match = 0
    total_executed = 0
    total_answer_match = 0

    rows = []

    for idx, pred in enumerate(predictions, start=1):
        pred_id = pred.get("id", f"example_{idx}")
        question = pred.get("question", "")
        pred_prog = pred.get("predicted_program", "")
        gold_prog = pred.get("gold_program", "")
        gold_ans = pred.get("gold_answer", "")

        pred_norm = normalize_program(pred_prog)
        gold_norm = normalize_program(gold_prog)

        program_match = pred_norm == gold_norm
        if program_match:
            total_prog_match += 1

            success, exec_answer = execute_program(pred_prog)
            if success:
                total_executed += 1
                answer_match, exec_str, gold_str = compare_answers(exec_answer, gold_ans)
                if answer_match:
                    total_answer_match += 1
            else:
                answer_match = False
                exec_str = ""
                gold_str = str(gold_ans)
        else:
            success = False
            exec_str = ""
            gold_str = str(gold_ans)
            answer_match = False

        rows.append(
            {
                "id": pred_id,
                "question": question,
                "program_match": program_match,
                "program": gold_prog if program_match else pred_prog,
                "execution_success": success if program_match else False,
                "executed_answer": exec_str,
                "gold_answer": gold_str,
                "answer_match": answer_match,
            }
        )

    # Write CSV
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "id",
            "question",
            "program_match",
            "program",
            "execution_success",
            "executed_answer",
            "gold_answer",
            "answer_match",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Print summary
    print("\n" + "=" * 70)
    print("LoRA Program Execution Evaluation")
    print("=" * 70)
    print(f"Total predictions:                 {total}")
    print(f"Program matches:                   {total_prog_match}")
    print(f"Executed (on matching programs):   {total_executed}")
    print(f"Executed answers matching gold:    {total_answer_match}")
    if total_prog_match > 0:
        print(
            f"\nAnswer accuracy (within matching-program subset): "
            f"{total_answer_match}/{total_prog_match} "
            f"({total_answer_match / total_prog_match * 100:.2f}%)"
        )
    print("=" * 70)
    print(f"Detailed results saved to: {output_csv}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate LoRA predictions by executing matching programs."
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("results/predictions/Mistral-7B-Instruct-v0.2_lora_predictions.json"),
        help="Path to predictions JSON file.",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=Path(
            "results/predictions/Mistral-7B-Instruct-v0.2_lora_program_exec_eval.csv"
        ),
        help="Path to output CSV file.",
    )

    args = parser.parse_args()
    evaluate_lora_program_execution(args.predictions, args.output_csv)


if __name__ == "__main__":
    main()


