"""
LangChain tool calling inference for LoRA fine-tuned models.
Replaces feedback loop approach with proper multi-turn tool calling.
"""

import os
import sys
import json
import yaml
import argparse
from typing import List, Dict, Optional
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from enhanced_executor import EnhancedExecutor
from tool_calling.langchain_agent import create_finqa_agent_with_lora


def format_context(example: Dict) -> str:
    """
    Format context from example (pre_text, post_text, table).
    
    Args:
        example: Test example with pre_text, post_text, table
        
    Returns:
        Formatted context string
    """
    context_parts = []
    
    # Add pre_text
    if example.get('pre_text'):
        context_parts.append("Text before table:")
        context_parts.extend(example['pre_text'][:10])  # Limit length
    
    # Add table
    if example.get('table'):
        context_parts.append("\nTable:")
        table = example['table']
        if len(table) > 0:
            # Header
            header = " | ".join(str(cell) for cell in table[0])
            context_parts.append(f"| {header} |")
            context_parts.append("|" + "|".join(["---"] * len(table[0])) + "|")
            # Rows
            for row in table[1:]:
                row_str = " | ".join(str(cell) for cell in row)
                context_parts.append(f"| {row_str} |")
    
    # Add post_text
    if example.get('post_text'):
        context_parts.append("\nText after table:")
        context_parts.extend(example['post_text'][:10])  # Limit length
    
    return "\n".join(context_parts)


def extract_answer_from_agent_output(output: str) -> str:
    """
    Extract final answer from agent output.
    
    Args:
        output: Agent output string
        
    Returns:
        Extracted answer
    """
    import re
    
    # Clean up spacing issues first
    # Fix patterns like "a d d" -> "add", "s u b t r a c t" -> "subtract"
    output = re.sub(r'\ba\s+d\s+d\b', 'add', output)
    output = re.sub(r'\bs\s+u\s+b\s+t\s+r\s+a\s+c\s+t\b', 'subtract', output)
    output = re.sub(r'\bm\s+u\s+l\s+t\s+i\s+p\s+l\s+y\b', 'multiply', output)
    output = re.sub(r'\bd\s+i\s+v\s+i\s+d\s+e\b', 'divide', output)
    output = re.sub(r'\bg\s+r\s+e\s+a\s+t\s+e\s+r\b', 'greater', output)
    output = re.sub(r'\be\s+x\s+p\b', 'exp', output)
    
    # Look for "Answer:" marker
    answer_match = re.search(r'Answer:\s*([^\n]+)', output, re.IGNORECASE)
    if answer_match:
        answer = answer_match.group(1).strip()
        # Remove trailing explanation text
        answer = answer.split('This')[0].split('Therefore')[0].split('So')[0].strip()
        # Remove trailing punctuation
        answer = answer.rstrip('.,;:!?%')
        return answer
    
    # Look for "Final Answer:" marker
    final_answer_match = re.search(r'Final\s+Answer:\s*([^\n]+)', output, re.IGNORECASE)
    if final_answer_match:
        answer = final_answer_match.group(1).strip()
        answer = answer.split('This')[0].split('Therefore')[0].split('So')[0].strip()
        answer = answer.rstrip('.,;:!?%')
        return answer
    
    # Look for numbers after "Observation:" (tool results)
    observation_pattern = r'Observation:\s*([^\n]+)'
    observations = re.findall(observation_pattern, output, re.IGNORECASE)
    if observations:
        # Take the last observation (usually the final result)
        last_obs = observations[-1].strip()
        # Extract number from observation
        number_pattern = r'\b(-?\d+\.?\d*(?:e[+-]?\d+)?)\b'
        numbers = re.findall(number_pattern, last_obs)
        if numbers:
            return numbers[-1]
    
    # Look for standalone numbers at the end
    number_pattern = r'\b(-?\d+\.?\d*(?:e[+-]?\d+)?)\b'
    matches = re.findall(number_pattern, output)
    if matches:
        return matches[-1]  # Take the last number found
    
    return output.strip()  # Return as-is if no pattern matches


def process_sample_with_tool_calling(
    sample: Dict,
    agent,
    executor: EnhancedExecutor
) -> Dict:
    """
    Process a single sample using LangChain tool calling.
    
    Args:
        sample: Test sample
        agent: LangChain agent
        executor: EnhancedExecutor instance
        
    Returns:
        Result dictionary
    """
    question = sample.get('qa', {}).get('question', '')
    gold_program = sample.get('qa', {}).get('program', '')
    gold_answer = str(sample.get('qa', {}).get('answer', ''))
    
    # Format context
    context_str = format_context(sample)
    
    # Create prompt for agent
    prompt = f"""You are a financial analyst. Answer the question by using the available tools step by step.

Context:
{context_str}

Question: {question}

Use the tools to calculate the answer step by step. When you have the final answer, provide it clearly with "Final Answer: <value>".

Available tools:
- add(a, b): Add two numbers
- subtract(a, b): Subtract two numbers  
- multiply(a, b): Multiply two numbers
- divide(a, b): Divide two numbers
- greater(a, b): Return the greater value
- exp(a, b): Raise a to power b
- const(value): Use a constant value
- table_access(row, col): Access table cell

Format your tool calls as: Tool: tool_name, Arguments: a=value1, b=value2
"""
    
    # Update executor with this sample's table
    executor.table = sample.get('table', [])
    executor.reset()
    
    try:
        # Run agent with intermediate steps
        from langchain.agents import AgentExecutor
        
        if isinstance(agent, AgentExecutor):
            # Use invoke to get intermediate steps
            response = agent.invoke({"input": prompt})
            result = response.get("output", "")
            intermediate_steps = response.get("intermediate_steps", [])
        else:
            # Fallback to run - try to get intermediate steps from agent
            result = agent.run(prompt)
            # Try to extract intermediate steps from agent if available
            intermediate_steps = []
            if hasattr(agent, 'agent_executor'):
                if hasattr(agent.agent_executor, 'intermediate_steps'):
                    intermediate_steps = agent.agent_executor.intermediate_steps
            elif hasattr(agent, 'intermediate_steps'):
                intermediate_steps = agent.intermediate_steps
        
        # Extract answer - clean up the result first
        result_str = str(result)
        # Remove spaces between characters in tool calls (fix formatting issue)
        import re
        # Fix patterns like "a d d" -> "add", "s u b t r a c t" -> "subtract"
        result_str = re.sub(r'\ba\s+d\s+d\b', 'add', result_str)
        result_str = re.sub(r'\bs\s+u\s+b\s+t\s+r\s+a\s+c\s+t\b', 'subtract', result_str)
        result_str = re.sub(r'\bm\s+u\s+l\s+t\s+i\s+p\s+l\s+y\b', 'multiply', result_str)
        result_str = re.sub(r'\bd\s+i\s+v\s+i\s+d\s+e\b', 'divide', result_str)
        result_str = re.sub(r'\bg\s+r\s+e\s+a\s+t\s+e\s+r\b', 'greater', result_str)
        result_str = re.sub(r'\be\s+x\s+p\b', 'exp', result_str)
        
        predicted_answer = extract_answer_from_agent_output(result_str)
        
        return {
            'id': sample.get('id', ''),
            'question': question,
            'predicted_answer': predicted_answer,
            'gold_program': ' '.join(gold_program) if isinstance(gold_program, list) else gold_program,
            'gold_answer': gold_answer,
            'tool_calls': len(intermediate_steps),
            'raw_output': result_str,
            'intermediate_steps': str(intermediate_steps) if intermediate_steps else ''
        }
    except Exception as e:
        return {
            'id': sample.get('id', ''),
            'question': question,
            'predicted_answer': '',
            'gold_program': ' '.join(gold_program) if isinstance(gold_program, list) else gold_program,
            'gold_answer': gold_answer,
            'tool_calls': 0,
            'raw_output': f"ERROR: {str(e)}",
            'intermediate_steps': ''
        }


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="LangChain Tool Calling for LoRA Models")
    parser.add_argument("--config", type=str, default="configs/lora_tool_calling_config.yaml",
                        help="Path to config YAML file")
    parser.add_argument("--model_name", type=str, default=None,
                        help="Override model name from config")
    parser.add_argument("--adapter_path", type=str, default=None,
                        help="Override adapter path from config")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max test samples (for quick testing)")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("LANGCHAIN TOOL CALLING FOR LORA MODELS")
    print("=" * 80)
    
    # Load config
    config = load_config(args.config)
    
    # Override config with command line args
    if args.model_name:
        config['model']['model_name_or_path'] = args.model_name
    if args.adapter_path:
        config['model']['adapter_path'] = args.adapter_path
    
    model_name = config['model']['model_name_or_path']
    adapter_path = config['model'].get('adapter_path')
    
    print(f"Model: {model_name}")
    print(f"Adapters: {adapter_path if adapter_path else 'None (base model)'}")
    print(f"Config: {args.config}")
    print("=" * 80)
    
    # Initialize executor
    print("\n[1/5] Initializing executor...")
    executor = EnhancedExecutor()
    
    # Create agent
    print("\n[2/5] Creating LangChain agent with tools...")
    agent, model, tokenizer = create_finqa_agent_with_lora(
        executor=executor,
        model_name=model_name,
        adapter_path=adapter_path,
        load_in_8bit=config['model'].get('load_in_8bit', False),
        max_iterations=config['agent'].get('max_iterations', 10),
        verbose=config['agent'].get('verbose', True)
    )
    
    # Load test data
    print("\n[3/5] Loading test data...")
    data_dir = config['data']['data_dir']
    test_file = config['data']['test_file']
    test_path = os.path.join(data_dir, test_file)
    
    with open(test_path, 'r') as f:
        test_data = json.load(f)
    
    if args.max_samples:
        test_data = test_data[:args.max_samples]
        print(f"  Processing first {args.max_samples} samples")
    
    print(f"  Test examples: {len(test_data)}")
    
    # Process samples
    print("\n[4/5] Running tool calling inference...")
    results = []
    
    for sample in tqdm(test_data, desc="Processing"):
        result = process_sample_with_tool_calling(sample, agent, executor)
        results.append(result)
    
    # Save results
    print("\n[5/5] Saving predictions...")
    output_dir = config['data']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    model_name_short = model_name.split("/")[-1]
    adapter_suffix = "lora" if adapter_path else "base"
    output_filename = f"{model_name_short}_{adapter_suffix}_tool_calling_predictions.json"
    output_path = os.path.join(output_dir, output_filename)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"  Saved to: {output_path}")
    
    # Print statistics
    print("\n" + "=" * 80)
    print("STATISTICS")
    print("=" * 80)
    total = len(results)
    avg_tool_calls = sum(r['tool_calls'] for r in results) / total if total > 0 else 0
    errors = sum(1 for r in results if 'ERROR' in r['raw_output'])
    
    print(f"Total samples: {total}")
    print(f"Average tool calls per question: {avg_tool_calls:.2f}")
    print(f"Errors: {errors} ({errors/total*100:.1f}%)")
    
    # Preview
    print("\n" + "=" * 80)
    print("PREVIEW OF PREDICTIONS")
    print("=" * 80)
    if results:
        print(f"\nFirst prediction:")
        print(f"  Question: {results[0]['question'][:80]}...")
        print(f"  Predicted answer: {results[0]['predicted_answer']}")
        print(f"  Gold answer: {results[0]['gold_answer']}")
        print(f"  Tool calls: {results[0]['tool_calls']}")
    
    print("=" * 80)
    print(f"\n✓ Tool calling inference complete! Results saved to {output_path}")


if __name__ == "__main__":
    main()

