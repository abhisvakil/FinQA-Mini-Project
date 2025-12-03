"""
Enhanced Program Executor with Validation and Feedback for FinQA.
Parses string-based programs and provides detailed error messages.
"""

import re
from typing import List, Dict, Any, Tuple, Optional
import numpy as np


class EnhancedExecutor:
    """Enhanced executor with validation, feedback, and retry support."""
    
    def __init__(self, table: List[List[str]] = None, context: str = ""):
        """
        Initialize executor with table data.
        
        Args:
            table: 2D list representing the table
            context: Additional context text
        """
        self.table = table or []
        self.context = context
        self.reset()
    
    def reset(self):
        """Reset executor state."""
        self.results = {}  # Store intermediate results (#0, #1, etc.)
        self.execution_trace = []
    
    def parse_program(self, program_str: str) -> List[Dict[str, Any]]:
        """
        Parse program string into structured operations.
        
        Args:
            program_str: Program string like "subtract(750, 500), divide(#0, 500)"
        
        Returns:
            List of operation dictionaries
        """
        if not program_str or program_str.strip() == "":
            return []
        
        operations = []
        
        # Use regex to find all operations with their arguments
        # Pattern: operation_name(args) where args can contain commas
        pattern = r'([a-z_]+)\s*\(([^)]+)\)'
        matches = re.findall(pattern, program_str)
        
        if not matches:
            raise ValueError(f"No valid operations found in: '{program_str}'")
        
        for op_name, args_str in matches:
            # Parse arguments - split by comma
            args = []
            if args_str.strip():
                # Split by comma and clean up
                arg_parts = [a.strip() for a in args_str.split(',')]
                args = [a for a in arg_parts if a]
            
            # Create raw representation
            raw = f"{op_name}({args_str})"
            
            operations.append({
                'operation': op_name,
                'args': args,
                'raw': raw
            })
        
        return operations
    
    def resolve_value(self, arg: str) -> float:
        """
        Resolve an argument to a numeric value.
        
        Args:
            arg: Argument string (number, #0, const_100, etc.)
        
        Returns:
            Numeric value
        """
        arg = arg.strip()
        
        # Check if it's a reference to previous result (#0, #1, etc.)
        if arg.startswith('#'):
            ref = arg
            if ref not in self.results:
                raise ValueError(f"Reference {ref} not found. Available: {list(self.results.keys())}")
            return self.results[ref]
        
        # Check if it's a constant (const_100, const_2, etc.)
        if arg.startswith('const_'):
            const_value = arg.replace('const_', '')
            return float(const_value)
        
        # Check for special table operations
        if arg.startswith('table_'):
            raise ValueError(f"Table operations not yet supported in this context: {arg}")
        
        # Try to parse as number
        try:
            # Remove common formatting
            cleaned = arg.replace(',', '').replace('$', '').replace('%', '').strip()
            # Handle spaces in numbers (e.g., "5 8 2 9" → "5829")
            cleaned = cleaned.replace(' ', '')
            return float(cleaned)
        except ValueError:
            raise ValueError(f"Cannot resolve argument '{arg}' to a number")
    
    def execute_operation(self, op_name: str, args: List[str]) -> float:
        """
        Execute a single operation.
        
        Args:
            op_name: Operation name
            args: List of arguments
        
        Returns:
            Result value
        """
        # Resolve all arguments to values
        try:
            values = [self.resolve_value(arg) for arg in args]
        except Exception as e:
            raise ValueError(f"Error resolving arguments for {op_name}: {str(e)}")
        
        # Execute operation
        if op_name == 'add':
            if len(values) != 2:
                raise ValueError(f"add requires 2 arguments, got {len(values)}")
            return values[0] + values[1]
        
        elif op_name == 'subtract':
            if len(values) != 2:
                raise ValueError(f"subtract requires 2 arguments, got {len(values)}")
            return values[0] - values[1]
        
        elif op_name == 'multiply':
            if len(values) != 2:
                raise ValueError(f"multiply requires 2 arguments, got {len(values)}")
            return values[0] * values[1]
        
        elif op_name == 'divide':
            if len(values) != 2:
                raise ValueError(f"divide requires 2 arguments, got {len(values)}")
            if abs(values[1]) < 1e-10:
                raise ValueError("Division by zero")
            return values[0] / values[1]
        
        elif op_name == 'greater':
            if len(values) != 2:
                raise ValueError(f"greater requires 2 arguments, got {len(values)}")
            return max(values[0], values[1])
        
        elif op_name == 'exp':
            if len(values) != 2:
                raise ValueError(f"exp requires 2 arguments, got {len(values)}")
            return values[0] ** values[1]
        
        elif op_name in ['table_max', 'table_min', 'table_sum', 'table_average']:
            # These would need table context
            raise NotImplementedError(f"Table operation '{op_name}' not yet implemented")
        
        else:
            raise ValueError(f"Unknown operation: '{op_name}'")
    
    def execute(self, program_str: str) -> Dict[str, Any]:
        """
        Execute a complete program and return detailed results.
        
        Args:
            program_str: Program string
        
        Returns:
            Dictionary with execution results, errors, and feedback
        """
        self.reset()
        
        result = {
            'success': False,
            'answer': None,
            'error': None,
            'error_type': None,
            'step_failed': None,
            'trace': [],
            'feedback': ""
        }
        
        try:
            # Parse program
            operations = self.parse_program(program_str)
            
            if not operations:
                result['error'] = "Empty program"
                result['error_type'] = 'empty_program'
                result['feedback'] = "Program is empty. Please provide a valid program."
                return result
            
            # Execute each operation
            for idx, op_dict in enumerate(operations):
                try:
                    value = self.execute_operation(op_dict['operation'], op_dict['args'])
                    ref = f'#{idx}'
                    self.results[ref] = value
                    
                    trace_entry = {
                        'step': idx,
                        'operation': op_dict['raw'],
                        'result': value,
                        'reference': ref
                    }
                    self.execution_trace.append(trace_entry)
                    result['trace'].append(trace_entry)
                    
                except Exception as e:
                    result['error'] = str(e)
                    result['error_type'] = 'execution_error'
                    result['step_failed'] = idx
                    result['feedback'] = self.generate_error_feedback(idx, op_dict, str(e))
                    result['trace'] = self.execution_trace
                    return result
            
            # Get final result (last operation's result)
            final_ref = f'#{len(operations) - 1}'
            result['answer'] = self.results[final_ref]
            result['success'] = True
            result['feedback'] = "✅ Program executed successfully"
            
        except ValueError as e:
            result['error'] = str(e)
            result['error_type'] = 'parse_error'
            result['feedback'] = self.generate_parse_feedback(program_str, str(e))
        except Exception as e:
            result['error'] = str(e)
            result['error_type'] = 'unknown_error'
            result['feedback'] = f"Unexpected error: {str(e)}"
        
        return result
    
    def generate_error_feedback(self, step: int, op_dict: Dict, error_msg: str) -> str:
        """Generate helpful feedback for execution errors."""
        feedback = f"❌ Error at step {step + 1}:\n"
        feedback += f"Operation: {op_dict['raw']}\n"
        feedback += f"Error: {error_msg}\n\n"
        
        # Provide specific suggestions based on error type
        if "not found" in error_msg.lower():
            feedback += "💡 Suggestion: Make sure you're referencing results from previous steps correctly.\n"
            feedback += f"Available references: {list(self.results.keys())}\n"
        elif "division by zero" in error_msg.lower():
            feedback += "💡 Suggestion: Check if you're dividing by zero. Review your calculation logic.\n"
        elif "requires 2 arguments" in error_msg:
            feedback += f"💡 Suggestion: The operation '{op_dict['operation']}' needs exactly 2 arguments.\n"
        elif "cannot resolve" in error_msg.lower():
            feedback += "💡 Suggestion: Check that all numbers are properly formatted (no typos).\n"
        
        feedback += "\nPlease revise your program and try again."
        return feedback
    
    def generate_parse_feedback(self, program_str: str, error_msg: str) -> str:
        """Generate helpful feedback for parsing errors."""
        feedback = "❌ Program Parsing Error:\n"
        feedback += f"Error: {error_msg}\n\n"
        feedback += "💡 Expected format: operation(arg1, arg2), operation(arg3, arg4), ...\n"
        feedback += "Examples:\n"
        feedback += "  - subtract(750, 500)\n"
        feedback += "  - divide(#0, 500), multiply(#1, 100)\n"
        feedback += "\nPlease check your program syntax and try again."
        return feedback
    
    def validate_result(self, answer: float, question: str) -> Tuple[bool, List[str]]:
        """
        Validate if the result makes sense for the question.
        
        Args:
            answer: Computed answer
            question: Question text
        
        Returns:
            (is_valid, list_of_warnings)
        """
        warnings = []
        
        # Check for extreme values
        if abs(answer) > 1e10:
            warnings.append(f"Result is extremely large ({answer}). Double-check your calculation.")
        
        # Check for percentage-related questions
        question_lower = question.lower()
        if any(word in question_lower for word in ['percentage', 'percent', '%', 'rate']):
            if answer > 1000:
                warnings.append(f"Result ({answer}) seems too large for a percentage. Did you forget to divide or multiply correctly?")
            elif answer < 0.01 and answer != 0:
                warnings.append(f"Result ({answer}) seems too small. Did you mean {answer * 100}%?")
        
        # Check for ratio/growth questions
        if any(word in question_lower for word in ['ratio', 'times', 'growth', 'change']):
            if answer < -10:
                warnings.append(f"Result ({answer}) seems unusual for a ratio/growth question.")
        
        is_valid = len(warnings) == 0
        return is_valid, warnings


def test_enhanced_executor():
    """Test the enhanced executor."""
    print("="*60)
    print("Testing Enhanced Executor")
    print("="*60)
    
    executor = EnhancedExecutor()
    
    # Test cases
    test_cases = [
        ("subtract(750, 500)", "Simple subtraction"),
        ("subtract(750, 500), divide(#0, 500)", "With reference"),
        ("subtract(750, 500), divide(#0, 500), multiply(#1, 100)", "Multi-step percentage"),
        ("add(100, 200), subtract(#0, 50)", "Multiple operations"),
        ("divide(100, 0)", "Division by zero (should fail)"),
        ("invalid_program", "Invalid format (should fail)"),
    ]
    
    for program, description in test_cases:
        print(f"\n{description}")
        print(f"Program: {program}")
        print("-" * 60)
        
        result = executor.execute(program)
        
        if result['success']:
            print(f"✅ Success!")
            print(f"Answer: {result['answer']}")
            print(f"Trace:")
            for trace in result['trace']:
                print(f"  Step {trace['step'] + 1}: {trace['operation']} → {trace['result']} (stored as {trace['reference']})")
        else:
            print(f"❌ Failed:")
            print(f"Error Type: {result['error_type']}")
            print(f"Error: {result['error']}")
            if result['step_failed'] is not None:
                print(f"Failed at step: {result['step_failed'] + 1}")
            print(f"\nFeedback:\n{result['feedback']}")


if __name__ == "__main__":
    test_enhanced_executor()

