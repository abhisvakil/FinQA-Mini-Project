"""
LoRA Fine-tuning script for FinQA.
Trains Llama-3-8B or Mistral-7B using LoRA on the simplified FinQA dataset.
"""

import os
import sys
import json
import torch
import argparse
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from datasets import Dataset
import yaml

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_loader_simplified import FinQASimplifiedLoader


@dataclass
class ModelArguments:
    """Arguments for model configuration."""
    model_name_or_path: str = field(
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    use_flash_attention: bool = field(
        default=False,
        metadata={"help": "Whether to use flash attention 2"}
    )


@dataclass
class DataArguments:
    """Arguments for data loading."""
    data_dir: str = field(
        default="data/simplified",
        metadata={"help": "Directory containing simplified data"}
    )
    max_seq_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length"}
    )
    max_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum number of samples for quick testing"}
    )


@dataclass
class LoraArguments:
    """Arguments for LoRA configuration."""
    lora_r: int = field(default=8, metadata={"help": "LoRA rank"})
    lora_alpha: int = field(default=16, metadata={"help": "LoRA alpha"})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout"})
    lora_target_modules: Optional[List[str]] = field(
        default=None,
        metadata={"help": "Target modules for LoRA"}
    )


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def format_instruction(example: Dict) -> str:
    """Format example as instruction for training."""
    instruction = """Answer the following financial question by generating a reasoning program and the final answer.

Available Operations:
- add(a, b): Add two numbers
- subtract(a, b): Subtract b from a
- multiply(a, b): Multiply two numbers
- divide(a, b): Divide a by b
- greater(a, b): Return the greater of two numbers
- exp(a, b): Calculate a raised to the power of b

"""
    
    # Add context
    context_parts = []
    
    if example['pre_text']:
        context_parts.append("Text:")
        context_parts.extend(example['pre_text'][:10])  # Limit to avoid too long
    
    if example['table']:
        context_parts.append("\nTable:")
        # Format table
        table = example['table']
        if len(table) > 0:
            header = " | ".join(table[0])
            context_parts.append(f"| {header} |")
            for row in table[1:]:
                row_str = " | ".join(str(cell) for cell in row)
                context_parts.append(f"| {row_str} |")
    
    context = "\n".join(context_parts)
    
    # Format input
    input_text = f"{instruction}Question: {example['question']}\n\nContext:\n{context}\n\nGenerate the reasoning program and final answer:"
    
    # Format output
    program_str = " ".join(example['program']) if example['program'] else ""
    output_text = f"\nProgram: {program_str}\nAnswer: {example['answer']}"
    
    return input_text, output_text


def prepare_dataset(data_path: str, tokenizer, max_length: int = 2048, max_samples: Optional[int] = None):
    """Prepare dataset for training."""
    # Load simplified data
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    if max_samples:
        data = data[:max_samples]
    
    # Format examples
    formatted_examples = []
    for example in data:
        input_text, output_text = format_instruction(example)
        full_text = input_text + output_text + tokenizer.eos_token
        
        formatted_examples.append({
            'text': full_text,
            'input': input_text,
            'output': output_text
        })
    
    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            max_length=max_length,
            padding='max_length',
            return_tensors='pt'
        )
    
    dataset = Dataset.from_list(formatted_examples)
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    return tokenized_dataset


def main():
    parser = argparse.ArgumentParser(description="LoRA Fine-tuning for FinQA")
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct",
                        help="Model name or path")
    parser.add_argument("--config", type=str, default="../configs/lora_config.yaml",
                        help="Path to config file")
    parser.add_argument("--output_dir", type=str, default="../results/lora",
                        help="Output directory")
    parser.add_argument("--data_dir", type=str, default="../data/simplified",
                        help="Data directory")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max samples for testing")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Training batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                        help="Learning rate")
    parser.add_argument("--lora_r", type=int, default=8,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16,
                        help="LoRA alpha")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("LORA FINE-TUNING FOR FINQA")
    print("=" * 80)
    print(f"Model: {args.model_name}")
    print(f"Output: {args.output_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"LoRA r: {args.lora_r}, alpha: {args.lora_alpha}")
    print("=" * 80)
    
    # Load tokenizer
    print("\n[1/6] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Load model
    print("\n[2/6] Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Configure LoRA
    print("\n[3/6] Configuring LoRA...")
    
    # Determine target modules based on model
    if "llama" in args.model_name.lower():
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
    elif "mistral" in args.model_name.lower():
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
    else:
        target_modules = ["q_proj", "v_proj"]
    
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Prepare datasets
    print("\n[4/6] Preparing datasets...")
    train_path = os.path.join(args.data_dir, "train_simplified.json")
    dev_path = os.path.join(args.data_dir, "dev_simplified.json")
    
    train_dataset = prepare_dataset(train_path, tokenizer, max_samples=args.max_samples)
    eval_dataset = prepare_dataset(dev_path, tokenizer, max_samples=100 if args.max_samples else None)
    
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Eval samples: {len(eval_dataset)}")
    
    # Training arguments
    print("\n[5/6] Setting up training...")
    model_name_short = args.model_name.split("/")[-1]
    output_dir = os.path.join(args.output_dir, model_name_short)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_steps=100,
        logging_steps=50,
        eval_steps=250,
        save_steps=500,
        save_total_limit=3,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        bf16=True,
        report_to="tensorboard",
        logging_dir=os.path.join(output_dir, "logs"),
        gradient_checkpointing=True,
        optim="adamw_torch"
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    # Train
    print("\n[6/6] Training...")
    print("-" * 80)
    trainer.train()
    
    # Save final model
    print("\n" + "=" * 80)
    print("SAVING MODEL")
    print("=" * 80)
    model.save_pretrained(os.path.join(output_dir, "final_model"))
    tokenizer.save_pretrained(os.path.join(output_dir, "final_model"))
    
    print(f"\n✓ Training complete! Model saved to {output_dir}/final_model")
    print("=" * 80)


if __name__ == "__main__":
    main()
