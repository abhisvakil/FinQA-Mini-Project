#!/usr/bin/env python3
"""
Simple FinQA-style program executor used as a tool by evaluation scripts.

This script defines `execute_program(program_str)` which takes a FinQA-style
program string (e.g. "subtract(5829,5735), divide(#0,5735)") and returns
its final computed answer.

Supported operations (mirrors the FinQA operators):
- add(a, b):        a + b
- subtract(a, b):   a - b
- multiply(a, b):   a * b
- divide(a, b):     a / b
- exp(a, b):        a ** b
- greater(a, b):    "yes" if a > b else "no"

Arguments can be:
- numeric literals, possibly with commas, "$", or "%" (e.g. "1,234", "$153.7", "9.5%")
- references to previous steps: "#0", "#1", ...
- constants like "const_100" (treated as 100.0)

Table operations (table_max, table_min, table_sum, table_average) are NOT
handled here; if a program uses them, execution will fail with a flag.
"""

from typing import Tuple, Union
import re


NumberOrStr = Union[float, str]


def clean_program_for_execution(program_str: str) -> str:
    """
    Clean a raw model program string into a form suitable for execution.

    Handles common formatting issues:
    - Strips leading/trailing whitespace
    - If multiple lines, prefers a line starting with 'Program:' or the first line
      that looks like an operation (contains '(')
    - Removes leading 'Program:' label
    - Removes any trailing 'Answer: ...' or explanation text on the same line
    - Removes Markdown code fences (``` and optional language tags)

    Note: we intentionally do NOT remove internal spaces between characters here;
    `execute_program` will normalize by removing all spaces so that patterns like
    's u b t r a c t ( 5 8 2 9 ,  5 7 3 5 )' become 'subtract(5829,5735)'.
    """
    if not program_str:
        return ""

    text = str(program_str).strip()

    # Drop markdown code fences if present
    if "```" in text:
        parts = text.split("```")
        # Heuristic: keep the content inside the first code block if it exists,
        # otherwise keep everything before the first fence.
        if len(parts) >= 3:
            text = parts[1].strip()
        else:
            text = parts[0].strip()

    # Split into non-empty lines
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return ""

    # Prefer an explicit Program: line
    program_line = None
    for ln in lines:
        if ln.lower().startswith("program:"):
            program_line = ln
            break

    # Otherwise, pick the first line that looks like an operation (has '(')
    if program_line is None:
        for ln in lines:
            if "(" in ln and ")" in ln:
                program_line = ln
                break

    # Fallback: just take the first non-empty line
    if program_line is None:
        program_line = lines[0]

    # Remove leading "Program:" label if still present
    lowered = program_line.lower()
    if lowered.startswith("program:"):
        program_line = program_line.split(":", 1)[1].strip()

    # Remove any inline "Answer:" section on the same line
    if "Answer:" in program_line:
        program_line = program_line.split("Answer:", 1)[0].strip()
    if "answer:" in program_line:
        program_line = program_line.split("answer:", 1)[0].strip()

    return program_line


def _str_to_num(token: str) -> float:
    """
    Convert a string token to a float, handling commas, $, %, and const_ prefix.
    Raises ValueError if conversion fails.
    """
    s = str(token).strip()
    s = s.replace(",", "")
    # Remove currency symbol
    s = s.replace("$", "")

    # Handle const_ prefix (e.g., const_100 -> 100)
    if s.startswith("const_"):
        s = s[len("const_") :]

    # Handle percentages (e.g., "12.5%")
    if s.endswith("%"):
        base = s[:-1]
        try:
            return float(base) / 100.0
        except ValueError as e:
            raise ValueError(f"Cannot parse percentage token '{token}': {e}")

    try:
        return float(s)
    except ValueError as e:
        raise ValueError(f"Cannot parse numeric token '{token}': {e}")


def execute_program(program_str: str) -> Tuple[bool, NumberOrStr]:
    """
    Execute a FinQA-style program string and return (success, answer).

    Args:
        program_str: Raw program string as produced by the model. This may
                     include labels like 'Program:', code fences, extra
                     whitespace, or spaced-out tokens.

    Returns:
        (success, answer)
        - success: bool, True if execution completed without errors
        - answer:  final result (float or 'yes'/'no') if success is True,
                   otherwise None
    """
    cleaned = clean_program_for_execution(program_str)
    if not cleaned:
        return False, None

    # Normalize by removing all spaces to handle forms like "s u b t r a c t"
    canonical = "".join(cleaned.split())

    # Extract operations of the form op(arg1,arg2) using regex so that
    # argument commas do not break the splitting between operations.
    # Example:
    #   "subtract(750,500),divide(#0,500),multiply(#1,100)"
    # becomes matches:
    #   [("subtract", "750,500"), ("divide", "#0,500"), ("multiply", "#1,100")]
    matches = re.findall(r"([a-z_]+)\(([^)]*)\)", canonical)
    if not matches:
        return False, None

    results = {}  # step index -> value
    final_result: NumberOrStr = None

    try:
        for idx, (name, args_str) in enumerate(matches):
            # Reject unsupported table operations
            if name.startswith("table_"):
                raise ValueError(f"Table operation '{name}' not supported in this tool")

            # We expect exactly two arguments separated by a comma
            raw_args = [a for a in args_str.split(",") if a != ""]
            if len(raw_args) != 2:
                raise ValueError(
                    f"Operation '{name}' expects 2 arguments, got {len(raw_args)} in '{op_str}'"
                )

            values = []
            for raw_arg in raw_args:
                token = raw_arg.strip()
                if token.startswith("#"):
                    # Reference to previous step
                    ref_idx = int(token[1:])
                    if ref_idx not in results:
                        raise ValueError(
                            f"Reference '{token}' not found. Available: {list(results.keys())}"
                        )
                    values.append(results[ref_idx])
                else:
                    values.append(_str_to_num(token))

            a, b = values

            if name == "add":
                final_result = a + b
            elif name == "subtract":
                final_result = a - b
            elif name == "multiply":
                final_result = a * b
            elif name == "divide":
                if abs(b) < 1e-12:
                    raise ValueError("Division by zero")
                final_result = a / b
            elif name == "exp":
                final_result = a ** b
            elif name == "greater":
                final_result = "yes" if a > b else "no"
            else:
                raise ValueError(f"Unknown operation '{name}'")

            results[idx] = final_result

        return True, final_result

    except Exception:
        # For analysis purposes we just mark failure and return None
        return False, None


if __name__ == "__main__":
    # Simple manual test cases (optional)
    examples = [
        "subtract(750,500)",
        "s u b t r a c t ( 7 5 0 ,  5 0 0 )",
        "subtract(750,500),divide(#0,500),multiply(#1,100)",
        "s u b t r a c t ( 7 5 0 ,  5 0 0 ) , d i v i d e ( # 0 ,  5 0 0 ) , m u l t i p l y ( # 1 ,  1 0 0 )",
        "greater(286.61,198.09)",
    ]

    for prog in examples:
        ok, ans = execute_program(prog)
        print(f"Program: {prog}")
        print(f"  Success: {ok}, Answer: {ans}")


