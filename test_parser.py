#!/usr/bin/env python3
"""Test the improved parse_model_output function"""

import re

def parse_model_output(output: str) -> tuple:
    """
    Parse model output to extract program and answer.
    Handles various edge cases like extra text, spacing issues, etc.
    
    Args:
        output: Raw model output
        
    Returns:
        (program, answer) tuple
    """
    program = ""
    answer = ""
    
    # Clean up output
    output = output.strip()
    
    # Strategy 1: Look for "Program:" and "Answer:" markers
    if "Program:" in output:
        program_part = output.split("Program:")[1]
        if "Answer:" in program_part:
            # Extract program (everything before Answer:)
            program = program_part.split("Answer:")[0].strip()
            # Extract answer (everything after Answer:, but clean it up)
            answer_part = program_part.split("Answer:")[1].strip()
            
            # Clean answer: take first line, remove extra text
            answer = answer_part.split('\n')[0].strip()
            # Remove common artifacts like "assistant", "|", etc.
            answer = answer.split('assistant')[0].split('Assistant')[0].split('|')[0].strip()
            # Remove any trailing punctuation that shouldn't be there
            answer = answer.rstrip('.,;:!?')
        else:
            # Only program, no explicit answer - try to extract from next lines
            program = program_part.split('\n')[0].strip()
    
    # Strategy 2: Look for program-like patterns at the start of lines
    if not program:
        # Match lines that start with operation names (add, subtract, etc.)
        lines = output.strip().split('\n')
        for line in lines:
            line = line.strip()
            # Check if line starts with a valid operation
            if re.match(r'^(add|subtract|multiply|divide|greater|exp)\(', line):
                program = line
                break
        
        # If still no program, try to find after "### Final" marker
        if not program and "### Final" in output:
            after_final = output.split("### Final")[1].strip()
            lines = after_final.split('\n')
            for line in lines:
                line = line.strip()
                if re.match(r'^(add|subtract|multiply|divide|greater|exp)\(', line):
                    program = line
                    break
    
    # Strategy 3: Extract answer from "Answer:" marker (case-insensitive)
    if not answer:
        # Look for Answer: marker (case-insensitive)
        answer_match = re.search(r'Answer:\s*([^\n]+)', output, re.IGNORECASE)
        if answer_match:
            answer = answer_match.group(1).strip()
            # Remove trailing explanation text
            answer = answer.split('This')[0].split('Therefore')[0].split('So')[0].strip()
            # Remove trailing punctuation
            answer = answer.rstrip('.,;:!?%')
    
    # Strategy 4: If no answer found, try to find a number after the program
    if not answer and program and output:
        # Look for numbers that appear after the program line
        program_idx = output.find(program)
        if program_idx != -1:
            after_program = output[program_idx + len(program):]
            # Look for standalone numbers
            number_pattern = r'\b(-?\d+\.?\d*(?:e[+-]?\d+)?)\b'
            matches = re.findall(number_pattern, after_program)
            if matches:
                # Take the first number found
                answer = matches[0]
    
    # Clean up program: remove spaces between characters if present
    if program and ' ' in program and len(program.split()) > 10:
        # Likely has spaces between characters, try to fix common patterns
        program = program.replace(' s u b t r a c t ', ' subtract ').replace('s u b t r a c t', 'subtract')
        program = program.replace(' a d d ', ' add ').replace('a d d', 'add')
        program = program.replace(' m u l t i p l y ', ' multiply ').replace('m u l t i p l y', 'multiply')
        program = program.replace(' d i v i d e ', ' divide ').replace('d i v i d e', 'divide')
        program = program.replace(' g r e a t e r ', ' greater ').replace('g r e a t e r', 'greater')
        program = program.replace(' e x p ', ' exp ').replace('e x p', 'exp')
        # Clean up multiple spaces
        program = ' '.join(program.split())
    
    return program, answer


# Test cases from your output
test_cases = [
    ("divide(add(const_30_7, const_2_1), const_56_0)\nAnswer: 0.5714285714285714", "Example 1"),
    ("subtract(const_20692, const_4614)\nAnswer: 25088", "Example 2"),
    ("divide(const_121.4, const_4187.8)\n### Answer: 0.028989534883720615", "Example 3"),
    ("subtract(masco[2017], const_206.49), divide(#0, masco[2012])\n### Final Answer: 0.13232465756944\n\nThis means that the percentage cumulative total shareholder return on Masco common stock was 13.23% higher than the S&P 500 index for the five-year period ended 2017.", "Example 4"),
    ("subtract(const_86842000, const_92710000), divide(#0, const_86842000), multiply(#1, 100)\nAnswer: -5.626026993100051%", "Example 5"),
    ("divide(const_4612, add(const_4612, const_3152))\nAnswer: 0.13152542537626967\n\nTherefore, approximately 13.15% of the total recourse debt is current.", "Example 6"),
    ("divide(const_301, #0), multiply(#1, 100), divide(#2, #0)\n### Final Answer: 11.924524137931034%", "Example 7"),
    ("greater(286.61, 198.09)\nAnswer: True", "Example 8"),
    ("### Final\nsubtract(const_1171, #1)\nAnswer: -35.0\n\nHowever, since the question does not explicitly ask for the absolute value of the change, the previous answer (-36.0) is technically correct as well, although less informative.", "Example 9"),
]

print("Testing parse_model_output function:\n")
print("=" * 80)

for raw_output, label in test_cases:
    program, answer = parse_model_output(raw_output)
    print(f"\n{label}:")
    print(f"Raw output: {raw_output[:100]}...")
    print(f"Extracted program: {program}")
    print(f"Extracted answer: {answer}")
    print("-" * 80)
