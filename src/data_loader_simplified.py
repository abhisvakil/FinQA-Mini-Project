"""
Simplified data loader for FinQA - PEFT/ICL training.
Extracts only essential fields: question, context (pre_text, post_text, table), and targets (answer, program).
"""

import json
import os
from typing import Dict, List, Optional


class FinQASimplifiedLoader:
    """Simplified loader for FinQA dataset - extracts only essential fields."""
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the simplified data loader.
        
        Args:
            data_dir: Directory containing the dataset JSON files
        """
        self.data_dir = data_dir
    
    def load_split(self, split: str = "train") -> List[Dict]:
        """
        Load a specific data split with simplified format.
        
        Args:
            split: One of 'train', 'dev', 'test'
            
        Returns:
            List of simplified examples
        """
        filepath = os.path.join(self.data_dir, f"{split}.json")
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset file not found: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        # Convert to simplified format
        simplified_data = [self._simplify_example(ex) for ex in raw_data]
        
        return simplified_data
    
    def _simplify_example(self, example: Dict) -> Dict:
        """
        Extract only essential fields from raw example.
        
        Args:
            example: Raw example from JSON file
            
        Returns:
            Simplified example with only needed fields
        """
        qa = example.get('qa', {})
        
        simplified = {
            # Input fields
            'id': example.get('id', ''),
            'question': qa.get('question', ''),
            'pre_text': example.get('pre_text', []),
            'post_text': example.get('post_text', []),
            'table': example.get('table', []),
            
            # Output fields (targets)
            'answer': qa.get('exe_ans', ''),  # Final numerical answer
            'program': qa.get('program', [])   # Reasoning program
        }
        
        return simplified
    
    def format_for_training(self, example: Dict, include_answer: bool = True) -> Dict:
        """
        Format example for instruction-tuning.
        
        Args:
            example: Simplified example
            include_answer: Whether to include answer in output (False for inference)
            
        Returns:
            Formatted example with 'input' and 'output' fields
        """
        # Build context
        context_parts = []
        
        # Add text context
        if example['pre_text']:
            context_parts.append("Text Context:")
            context_parts.extend(example['pre_text'])
        
        if example['post_text']:
            if example['pre_text']:
                context_parts.append("")  # Blank line
            context_parts.extend(example['post_text'])
        
        # Add table
        if example['table']:
            context_parts.append("\nTable:")
            context_parts.append(self._format_table(example['table']))
        
        context = "\n".join(context_parts)
        
        # Build input
        input_text = f"Question: {example['question']}\n\nContext:\n{context}"
        
        # Build output
        program_str = " ".join(example['program']) if example['program'] else ""
        
        if include_answer:
            output_text = f"Program: {program_str}\nAnswer: {example['answer']}"
        else:
            output_text = f"Program: {program_str}"
        
        return {
            'id': example['id'],
            'input': input_text,
            'output': output_text,
            'question': example['question'],
            'answer': example['answer'],
            'program': example['program']
        }
    
    def _format_table(self, table: List[List[str]]) -> str:
        """
        Format table as markdown-style text.
        
        Args:
            table: 2D list representing table
            
        Returns:
            Formatted table string
        """
        if not table or len(table) == 0:
            return ""
        
        # Use markdown table format
        lines = []
        
        # Header row
        header = " | ".join(table[0])
        lines.append(f"| {header} |")
        
        # Separator
        separator = " | ".join(["---"] * len(table[0]))
        lines.append(f"| {separator} |")
        
        # Data rows
        for row in table[1:]:
            row_str = " | ".join(str(cell) for cell in row)
            lines.append(f"| {row_str} |")
        
        return "\n".join(lines)
    
    def get_statistics(self, data: List[Dict]) -> Dict:
        """
        Get statistics about the simplified dataset.
        
        Args:
            data: List of simplified examples
            
        Returns:
            Dictionary with statistics
        """
        stats = {
            'num_examples': len(data),
            'avg_pre_text_len': sum(len(ex['pre_text']) for ex in data) / len(data) if data else 0,
            'avg_post_text_len': sum(len(ex['post_text']) for ex in data) / len(data) if data else 0,
            'avg_table_rows': sum(len(ex['table']) for ex in data) / len(data) if data else 0,
            'avg_program_len': sum(len(ex['program']) for ex in data) / len(data) if data else 0,
            'examples_with_tables': sum(1 for ex in data if ex['table']),
            'examples_with_pre_text': sum(1 for ex in data if ex['pre_text']),
            'examples_with_post_text': sum(1 for ex in data if ex['post_text'])
        }
        
        return stats
    
    def save_simplified_split(self, split: str = "train", output_dir: str = "data/simplified"):
        """
        Load and save simplified version of a split.
        
        Args:
            split: Data split to process
            output_dir: Directory to save simplified data
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Load and simplify
        data = self.load_split(split)
        
        # Save
        output_path = os.path.join(output_dir, f"{split}_simplified.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(data)} simplified examples to {output_path}")


def main():
    """Example usage of the simplified data loader."""
    loader = FinQASimplifiedLoader(data_dir="data")
    
    print("=" * 80)
    print("FINQA SIMPLIFIED DATA LOADER")
    print("=" * 80)
    
    try:
        # Load training data
        print("\nLoading training data...")
        train_data = loader.load_split("train")
        print(f"✓ Loaded {len(train_data)} training examples")
        
        # Get statistics
        print("\n" + "-" * 80)
        print("DATASET STATISTICS")
        print("-" * 80)
        stats = loader.get_statistics(train_data)
        for key, value in stats.items():
            print(f"  {key}: {value:.2f}" if isinstance(value, float) else f"  {key}: {value}")
        
        # Show example
        print("\n" + "-" * 80)
        print("EXAMPLE (RAW SIMPLIFIED FORMAT)")
        print("-" * 80)
        example = train_data[0]
        print(f"ID: {example['id']}")
        print(f"Question: {example['question']}")
        print(f"Pre-text sentences: {len(example['pre_text'])}")
        print(f"Post-text sentences: {len(example['post_text'])}")
        print(f"Table rows: {len(example['table'])}")
        print(f"Program tokens: {len(example['program'])}")
        print(f"Answer: {example['answer']}")
        
        # Show formatted example
        print("\n" + "-" * 80)
        print("EXAMPLE (FORMATTED FOR TRAINING)")
        print("-" * 80)
        formatted = loader.format_for_training(example)
        print("INPUT:")
        print(formatted['input'][:500] + "..." if len(formatted['input']) > 500 else formatted['input'])
        print("\nOUTPUT:")
        print(formatted['output'])
        
        # Save simplified versions
        print("\n" + "-" * 80)
        print("SAVING SIMPLIFIED DATASETS")
        print("-" * 80)
        for split in ['train', 'dev', 'test']:
            try:
                loader.save_simplified_split(split)
            except FileNotFoundError as e:
                print(f"⚠ Skipping {split}: {e}")
        
        print("\n" + "=" * 80)
        print("✓ Data loading and simplification complete!")
        print("=" * 80)
        
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        print("\nPlease ensure the FinQA dataset files are in the 'data/' directory.")
        print("Expected files: data/train.json, data/dev.json, data/test.json")


if __name__ == "__main__":
    main()
