"""
QLoRA Fine-tuning script for FinQA.
Trains Llama-3-8B or Mistral-7B using QLoRA (4-bit quantized LoRA).
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
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
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


# Import functions from train_lora.py to avoid duplication
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train_lora import format_instruction, prepare_dataset


def main():
    parser = argparse.ArgumentParser(description="QLoRA Fine-tuning for FinQA")
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct",
                        help="Model name or path")
    parser.add_argument("--config", type=str, default="../configs/qlora_config.yaml",
                        help="Path to config file")
    parser.add_argument("--output_dir", type=str, default="../results/qlora",
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
    print("QLORA (4-BIT QUANTIZED) FINE-TUNING FOR FINQA")
    print("=" * 80)
    print(f"Model: {args.model_name}")
    print(f"Output: {args.output_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"LoRA r: {args.lora_r}, alpha: {args.lora_alpha}")
    print(f"Quantization: 4-bit (QLoRA)")
    print("=" * 80)
    
    # Load tokenizer
    print("\n[1/6] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Configure 4-bit quantization
    print("\n[2/6] Configuring 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    # Load model with quantization
    print("  Loading model in 4-bit...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
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
        optim="paged_adamw_32bit",  # Special optimizer for QLoRA
        max_grad_norm=0.3  # Gradient clipping for stability
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
