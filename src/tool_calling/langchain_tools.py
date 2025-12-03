"""
LangChain tool wrappers for custom FinQA tools.
Wraps our custom tools as LangChain StructuredTool for use with agents.
"""

from typing import List, Optional
from langchain_core.tools import StructuredTool
# LangChain uses pydantic v1 internally
try:
    from pydantic.v1 import BaseModel, Field
except ImportError:
    from pydantic import BaseModel, Field
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.finqa_tools import FinQATools
from enhanced_executor import EnhancedExecutor


# Pydantic schemas for tool inputs
class AddInput(BaseModel):
    """Input schema for add tool."""
    a: float = Field(description="First number to add")
    b: float = Field(description="Second number to add")


class SubtractInput(BaseModel):
    """Input schema for subtract tool."""
    a: float = Field(description="First number (minuend)")
    b: float = Field(description="Second number (subtrahend)")


class MultiplyInput(BaseModel):
    """Input schema for multiply tool."""
    a: float = Field(description="First number to multiply")
    b: float = Field(description="Second number to multiply")


class DivideInput(BaseModel):
    """Input schema for divide tool."""
    a: float = Field(description="Numerator")
    b: float = Field(description="Denominator")


class GreaterInput(BaseModel):
    """Input schema for greater tool."""
    a: float = Field(description="First number")
    b: float = Field(description="Second number")


class ExpInput(BaseModel):
    """Input schema for exp tool."""
    a: float = Field(description="Base")
    b: float = Field(description="Exponent")


class ConstInput(BaseModel):
    """Input schema for const tool."""
    value: float = Field(description="Constant value")


class TableAccessInput(BaseModel):
    """Input schema for table_access tool."""
    row: int = Field(description="Row index (0-indexed)")
    col: int = Field(description="Column index (0-indexed)")


def create_finqa_langchain_tools(executor: EnhancedExecutor) -> List[StructuredTool]:
    """
    Create LangChain StructuredTool wrappers for custom FinQA tools.
    
    Args:
        executor: EnhancedExecutor instance with table context
        
    Returns:
        List of LangChain StructuredTool objects
    """
    finqa_tools = FinQATools(executor)
    
    tools = [
        StructuredTool.from_function(
            func=finqa_tools.add,
            name="add",
            description="Add two numbers. Use this when you need to add two values together. Example: add(5, 3) returns 8.",
            args_schema=AddInput
        ),
        StructuredTool.from_function(
            func=finqa_tools.subtract,
            name="subtract",
            description="Subtract two numbers. Use this when you need to find the difference between two values. Example: subtract(10, 3) returns 7.",
            args_schema=SubtractInput
        ),
        StructuredTool.from_function(
            func=finqa_tools.multiply,
            name="multiply",
            description="Multiply two numbers. Use this when you need to calculate the product of two values. Example: multiply(4, 5) returns 20.",
            args_schema=MultiplyInput
        ),
        StructuredTool.from_function(
            func=finqa_tools.divide,
            name="divide",
            description="Divide two numbers. Use this when you need to calculate a ratio or division. Example: divide(15, 3) returns 5. Be careful not to divide by zero.",
            args_schema=DivideInput
        ),
        StructuredTool.from_function(
            func=finqa_tools.greater,
            name="greater",
            description="Return the greater of two numbers. Use this when you need to find the maximum value. Example: greater(5, 3) returns 5.",
            args_schema=GreaterInput
        ),
        StructuredTool.from_function(
            func=finqa_tools.exp,
            name="exp",
            description="Exponentiation: raise first number to the power of second. Use this for power calculations. Example: exp(2, 3) returns 8.",
            args_schema=ExpInput
        ),
        StructuredTool.from_function(
            func=finqa_tools.const,
            name="const",
            description="Return a constant value. Use this when you need to use a specific number as-is. Example: const(100) returns 100.",
            args_schema=ConstInput
        ),
        StructuredTool.from_function(
            func=finqa_tools.table_access,
            name="table_access",
            description="Access a value from the table. Use this to get a specific cell value from the financial report table. Example: table_access(1, 2) returns the value at row 1, column 2. Remember: indices start at 0.",
            args_schema=TableAccessInput
        ),
    ]
    
    return tools

