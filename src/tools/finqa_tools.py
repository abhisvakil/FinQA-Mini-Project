"""
Custom FinQA tool implementations using EnhancedExecutor.
These tools will be wrapped in LangChain StructuredTool format.
"""

from typing import Union
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from enhanced_executor import EnhancedExecutor


class FinQATools:
    """Custom FinQA tool implementations using EnhancedExecutor."""
    
    def __init__(self, executor: EnhancedExecutor):
        """
        Initialize FinQA tools with an executor.
        
        Args:
            executor: EnhancedExecutor instance with table context
        """
        self.executor = executor
    
    def add(self, a: Union[float, str], b: Union[float, str]) -> float:
        """
        Add two numbers: add(a, b) returns a + b
        
        Args:
            a: First number
            b: Second number
            
        Returns:
            Sum of a and b
        """
        return self.executor.execute_operation('add', [str(a), str(b)])
    
    def subtract(self, a: Union[float, str], b: Union[float, str]) -> float:
        """
        Subtract two numbers: subtract(a, b) returns a - b
        
        Args:
            a: First number (minuend)
            b: Second number (subtrahend)
            
        Returns:
            Difference of a and b
        """
        return self.executor.execute_operation('subtract', [str(a), str(b)])
    
    def multiply(self, a: Union[float, str], b: Union[float, str]) -> float:
        """
        Multiply two numbers: multiply(a, b) returns a * b
        
        Args:
            a: First number
            b: Second number
            
        Returns:
            Product of a and b
        """
        return self.executor.execute_operation('multiply', [str(a), str(b)])
    
    def divide(self, a: Union[float, str], b: Union[float, str]) -> float:
        """
        Divide two numbers: divide(a, b) returns a / b
        
        Args:
            a: Numerator
            b: Denominator
            
        Returns:
            Quotient of a divided by b
        """
        return self.executor.execute_operation('divide', [str(a), str(b)])
    
    def greater(self, a: Union[float, str], b: Union[float, str]) -> float:
        """
        Return the greater of two numbers: greater(a, b) returns max(a, b)
        
        Args:
            a: First number
            b: Second number
            
        Returns:
            Maximum of a and b
        """
        return self.executor.execute_operation('greater', [str(a), str(b)])
    
    def exp(self, a: Union[float, str], b: Union[float, str]) -> float:
        """
        Exponentiation: exp(a, b) returns a raised to the power of b
        
        Args:
            a: Base
            b: Exponent
            
        Returns:
            a raised to the power of b
        """
        return self.executor.execute_operation('exp', [str(a), str(b)])
    
    def const(self, value: Union[float, str]) -> float:
        """
        Return a constant value: const(value) returns value
        
        Args:
            value: Constant value
            
        Returns:
            The constant value as float
        """
        return self.executor.execute_operation('const', [str(value)])
    
    def table_access(self, row: int, col: int) -> float:
        """
        Access a table cell: table_access(row, col) returns table[row][col]
        
        Args:
            row: Row index (0-indexed)
            col: Column index (0-indexed)
            
        Returns:
            Value at table[row][col] as float
        """
        if not self.executor.table:
            raise ValueError("No table available for table_access operation")
        
        if row < 0 or row >= len(self.executor.table):
            raise ValueError(f"Row index {row} out of bounds (table has {len(self.executor.table)} rows)")
        
        if col < 0 or col >= len(self.executor.table[row]):
            raise ValueError(f"Column index {col} out of bounds (row has {len(self.executor.table[row])} columns)")
        
        cell_value = self.executor.table[row][col]
        return self.executor._to_float(cell_value)

