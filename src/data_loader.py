"""
Data loading utilities for FinQA dataset.
"""

import json
import os
from typing import Dict, List, Tuple, Optional
import pandas as pd


class FinQADataLoader:
    """Loader for FinQA dataset files."""
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the data loader.
        
        Args:
            data_dir: Directory containing the dataset JSON files
        """
        self.data_dir = data_dir
    
    def load_json(self, filename: str) -> List[Dict]:
        """
        Load a JSON file from the data directory.
        
        Args:
            filename: Name of the JSON file (e.g., 'train.json')
            
        Returns:
            List of examples
        """
        filepath = os.path.join(self.data_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset file not found: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return data
    
    def load_split(self, split: str = "train") -> List[Dict]:
        """
        Load a specific data split.
        
        Args:
            split: One of 'train', 'dev', 'test'
            
        Returns:
            List of examples
        """
        filename = f"{split}.json"
        return self.load_json(filename)
    
    def parse_example(self, example: Dict) -> Dict:
        """
        Parse a single example into structured format.
        
        Args:
            example: Raw example from JSON file
            
        Returns:
            Parsed example with structured fields
        """
        parsed = {
            'id': example.get('id', ''),
            'pre_text': example.get('pre_text', []),
            'post_text': example.get('post_text', []),
            'table': example.get('table', []),
            'question': example.get('qa', {}).get('question', ''),
            'program': example.get('qa', {}).get('program', []),
            'gold_inds': example.get('qa', {}).get('gold_inds', []),
            'exe_ans': example.get('qa', {}).get('exe_ans', ''),
            'program_re': example.get('qa', {}).get('program_re', {})
        }
        return parsed
    
    def get_table_dataframe(self, example: Dict) -> pd.DataFrame:
        """
        Convert table from list format to pandas DataFrame.
        
        Args:
            example: Example dictionary with 'table' field
            
        Returns:
            DataFrame representation of the table
        """
        table = example.get('table', [])
        if not table:
            return pd.DataFrame()
        
        # First row is typically headers
        headers = table[0] if table else []
        rows = table[1:] if len(table) > 1 else []
        
        df = pd.DataFrame(rows, columns=headers)
        return df
    
    def get_all_text(self, example: Dict) -> str:
        """
        Combine pre_text and post_text into a single string.
        
        Args:
            example: Example dictionary
            
        Returns:
            Combined text string
        """
        pre_text = ' '.join(example.get('pre_text', []))
        post_text = ' '.join(example.get('post_text', []))
        return f"{pre_text} {post_text}".strip()
    
    def get_statistics(self, data: List[Dict]) -> Dict:
        """
        Compute statistics about the dataset.
        
        Args:
            data: List of examples
            
        Returns:
            Dictionary with statistics
        """
        stats = {
            'num_examples': len(data),
            'avg_question_length': 0,
            'avg_program_length': 0,
            'num_with_tables': 0,
            'num_with_text': 0,
            'program_operations': {}
        }
        
        question_lengths = []
        program_lengths = []
        
        for example in data:
            qa = example.get('qa', {})
            
            # Question length
            question = qa.get('question', '')
            question_lengths.append(len(question.split()))
            
            # Program length
            program = qa.get('program', [])
            program_lengths.append(len(program))
            
            # Count operations
            for token in program:
                if '(' in token:
                    op = token.split('(')[0]
                    stats['program_operations'][op] = stats['program_operations'].get(op, 0) + 1
            
            # Check for table and text
            if example.get('table'):
                stats['num_with_tables'] += 1
            if example.get('pre_text') or example.get('post_text'):
                stats['num_with_text'] += 1
        
        stats['avg_question_length'] = sum(question_lengths) / len(question_lengths) if question_lengths else 0
        stats['avg_program_length'] = sum(program_lengths) / len(program_lengths) if program_lengths else 0
        
        return stats


def main():
    """Example usage of the data loader."""
    loader = FinQADataLoader(data_dir="data")
    
    # Load training data
    try:
        train_data = loader.load_split("train")
        print(f"Loaded {len(train_data)} training examples")
        
        # Parse first example
        if train_data:
            example = loader.parse_example(train_data[0])
            print(f"\nExample ID: {example['id']}")
            print(f"Question: {example['question']}")
            print(f"Program: {example['program']}")
            
            # Get statistics
            stats = loader.get_statistics(train_data[:100])  # Sample
            print(f"\nStatistics (sample):")
            print(f"  Average question length: {stats['avg_question_length']:.2f} words")
            print(f"  Average program length: {stats['avg_program_length']:.2f} tokens")
            print(f"  Examples with tables: {stats['num_with_tables']}")
            print(f"  Examples with text: {stats['num_with_text']}")
    
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please download the FinQA dataset first.")


if __name__ == "__main__":
    main()

