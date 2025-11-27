"""
Program executor for FinQA reasoning programs.
Executes the generated programs to get final numerical answers.
"""

import re
from typing import List, Union, Dict, Any
import numpy as np


class ProgramExecutor:
    """Executes FinQA reasoning programs."""
    
    def __init__(self, table: List[List[str]], pre_text: List[str] = None, post_text: List[str] = None):
        """
        Initialize the executor with context.
        
        Args:
            table: 2D list representing the table (first row is headers)
            pre_text: List of text sentences before the table
            post_text: List of text sentences after the table
        """
        self.table = table
        self.pre_text = pre_text or []
        self.post_text = post_text or []
        self.variables = {}
    
    def execute(self, program: List[str]) -> Union[float, str]:
        """
        Execute a program and return the result.
        
        Args:
            program: List of program tokens (e.g., ['add(', '5', '3', ')'])
            
        Returns:
            Execution result (typically a float)
        """
        try:
            # Convert program tokens to executable expression
            expression = self._tokens_to_expression(program)
            
            # Evaluate the expression
            result = self._evaluate_expression(expression)
            
            return result
        except Exception as e:
            return f"ERROR: {str(e)}"
    
    def _tokens_to_expression(self, tokens: List[str]) -> str:
        """
        Convert program tokens to a Python-evaluable expression.
        
        Args:
            tokens: List of program tokens
            
        Returns:
            Expression string
        """
        # Join tokens and handle special cases
        program_str = ' '.join(tokens)
        
        # Replace FinQA operations with Python equivalents
        program_str = program_str.replace('add(', 'self._add(')
        program_str = program_str.replace('subtract(', 'self._subtract(')
        program_str = program_str.replace('multiply(', 'self._multiply(')
        program_str = program_str.replace('divide(', 'self._divide(')
        program_str = program_str.replace('greater(', 'self._greater(')
        program_str = program_str.replace('exp(', 'self._exp(')
        program_str = program_str.replace('max(', 'self._max(')
        program_str = program_str.replace('min(', 'self._min(')
        program_str = program_str.replace('table(', 'self._table(')
        program_str = program_str.replace('const(', 'self._const(')
        
        return program_str
    
    def _evaluate_expression(self, expression: str) -> float:
        """
        Safely evaluate the expression.
        
        Args:
            expression: Expression string
            
        Returns:
            Evaluated result
        """
        # This is a simplified version - in practice, you'd want a proper parser
        # For now, we'll use eval with a restricted namespace
        namespace = {
            'self': self,
            '__builtins__': {}
        }
        
        try:
            result = eval(expression, namespace)
            return float(result)
        except Exception as e:
            raise ValueError(f"Evaluation error: {str(e)}")
    
    def _add(self, a: Union[str, float], b: Union[str, float]) -> float:
        """Add two values."""
        return self._to_float(a) + self._to_float(b)
    
    def _subtract(self, a: Union[str, float], b: Union[str, float]) -> float:
        """Subtract two values."""
        return self._to_float(a) - self._to_float(b)
    
    def _multiply(self, a: Union[str, float], b: Union[str, float]) -> float:
        """Multiply two values."""
        return self._to_float(a) * self._to_float(b)
    
    def _divide(self, a: Union[str, float], b: Union[str, float]) -> float:
        """Divide two values."""
        b_val = self._to_float(b)
        if abs(b_val) < 1e-10:
            raise ValueError("Division by zero")
        return self._to_float(a) / b_val
    
    def _max(self, a: Union[str, float], b: Union[str, float]) -> float:
        """Return maximum of two values."""
        return max(self._to_float(a), self._to_float(b))
    
    def _min(self, a: Union[str, float], b: Union[str, float]) -> float:
        """Return minimum of two values."""
        return min(self._to_float(a), self._to_float(b))
    
    def _greater(self, a: Union[str, float], b: Union[str, float]) -> float:
        """Return the greater of two values."""
        return max(self._to_float(a), self._to_float(b))
    
    def _exp(self, a: Union[str, float], b: Union[str, float]) -> float:
        """Calculate a raised to the power of b."""
        return self._to_float(a) ** self._to_float(b)
    
    def _table(self, row: Union[str, int], col: Union[str, int]) -> float:
        """
        Access a table cell.
        
        Args:
            row: Row index (0-indexed)
            col: Column index (0-indexed)
            
        Returns:
            Cell value as float
        """
        row_idx = int(self._to_float(row))
        col_idx = int(self._to_float(col))
        
        if row_idx < 0 or row_idx >= len(self.table):
            raise ValueError(f"Row index {row_idx} out of bounds")
        if col_idx < 0 or col_idx >= len(self.table[row_idx]):
            raise ValueError(f"Column index {col_idx} out of bounds")
        
        cell_value = self.table[row_idx][col_idx]
        return self._to_float(cell_value)
    
    def _const(self, value: Union[str, float]) -> float:
        """Return a constant value."""
        return self._to_float(value)
    
    def _to_float(self, value: Union[str, float]) -> float:
        """
        Convert a value to float, handling various formats.
        
        Args:
            value: Value to convert
            
        Returns:
            Float value
        """
        if isinstance(value, (int, float)):
            return float(value)
        
        if isinstance(value, str):
            # Remove commas and other formatting
            cleaned = value.replace(',', '').replace('$', '').replace('%', '').strip()
            try:
                return float(cleaned)
            except ValueError:
                # Try to extract number from string
                numbers = re.findall(r'-?\d+\.?\d*', cleaned)
                if numbers:
                    return float(numbers[0])
                raise ValueError(f"Cannot convert '{value}' to float")
        
        raise ValueError(f"Unexpected value type: {type(value)}")


def test_executor():
    """Test the program executor with example programs."""
    # Example table
    table = [
        ["Year", "Revenue", "Expenses"],
        ["2020", "1000", "800"],
        ["2021", "1200", "900"]
    ]
    
    executor = ProgramExecutor(table)
    
    # Test cases
    test_cases = [
        (['add(', 'const(5)', 'const(3)', ')'], 8.0),
        (['subtract(', 'const(10)', 'const(4)', ')'], 6.0),
        (['multiply(', 'const(3)', 'const(4)', ')'], 12.0),
        (['divide(', 'const(15)', 'const(3)', ')'], 5.0),
        (['add(', 'table(1, 1)', 'table(2, 1)', ')'], 2200.0),  # Revenue sum
    ]
    
    print("Testing Program Executor:")
    print("-" * 50)
    
    for program, expected in test_cases:
        try:
            result = executor.execute(program)
            status = "✓" if abs(result - expected) < 1e-6 else "✗"
            print(f"{status} Program: {' '.join(program)}")
            print(f"  Expected: {expected}, Got: {result}")
        except Exception as e:
            print(f"✗ Program: {' '.join(program)}")
            print(f"  Error: {str(e)}")
        print()


if __name__ == "__main__":
    test_executor()

