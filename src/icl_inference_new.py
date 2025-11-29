# icl_inference.py

"""
In-Context Learning (ICL) inference script for FinQA.
Uses few-shot prompting with Llama-3-8B or Mistral-7B without fine-tuning.
"""

import os
import sys
import json
import yaml
import torch
import random
from typing import List, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import argparse
# --------- Utility Functions ---------

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def select_few_shot_examples(train_data: List[Dict], num_examples: int = 5, selection_method: str = "diverse") -> List[Dict]:
    # You can adjust this logic to your needs!
    if selection_method == "random":
        return random.sample(train_data, min(num_examples, len(train_data)))
    elif selection_method == "diverse":
        # Select diverse examples based on program length and table size
        examples = []
        sorted_by_program = sorted(train_data, key=lambda x: len(x.get('program', [])))
        n = len(sorted_by_program)
        indices = [
            n // 4,
            n // 2,
            3 * n // 4,
        ]
        for idx in indices:
            if idx < len(sorted_by_program):
                examples.append(sorted_by_program[idx])
        remaining = [ex for ex in train_data if ex not in examples]
        if remaining and len(examples) < num_examples:
            examples.extend(random.sample(remaining, min(num_examples - len(examples), len(remaining))))
        return examples[:num_examples]
    return train_data[:num_examples]

def finqa_to_yaml_examples(selected: List[Dict]) -> List[Dict]:
    out = []
    for ex in selected:
        # Build table as markdown string, adjust fields as needed!
        table_str = ""
        if ex.get("table"):
            table_str = "Table:\n"
            header = " | ".join(ex["table"][0])
            table_str += f"| {header} |\n"
            for row in ex["table"][1:]:
                row_str = " | ".join(str(cell) for cell in row)
                table_str += f"| {row_str} |\n"
        context_str = table_str # + other context as needed
        out.append({
            "question": ex["question"],
            "context": context_str.strip(),
            "program": " ".join(ex['program']) if isinstance(ex['program'], list) else ex['program']
        })
    return out

def format_few_shot_examples(few_shots: List[Dict]) -> str:
    formatted = []
    for ex in few_shots:
        formatted.append(
            f"Question: {ex['question']}\n{ex['context']}\nProgram: {ex['program']}"
        )
    return "\n\n".join(formatted)

def create_prompt_from_config(config: dict, example: dict) -> str:
    template = config["prompt_template"]
    system_prompt = config["system_prompt"]
    few_shot_block = format_few_shot_examples(config.get("few_shot_examples", []))

    # Compose context from example fields (FinQA specific)
    context = ""
    if "pre_text" in example:
        context += " ".join(example.get("pre_text", [])) + "\n"
    if "post_text" in example:
        context += " ".join(example.get("post_text", [])) + "\n"
    if "table" in example and example["table"]:
        context += "Table:\n"
        header = " | ".join(example["table"][0])
        context += f"| {header} |\n"
        for row in example["table"][1:]:
            row_str = " | ".join(str(cell) for cell in row)
            context += f"| {row_str} |\n"
    context = context.strip()

    return template.format(
        system_prompt=system_prompt,
        few_shot_examples=few_shot_block,
        question=example["question"],
        context=context,
    )

def parse_model_output(output: str) -> tuple:
    program = ""
    answer = ""
    if "Program:" in output:
        program_part = output.split("Program:")[1]
        if "Answer:" in program_part:
            program = program_part.split("Answer:")[0].strip()
            answer = program_part.split("Answer:")[1].strip()
        else:
            program = program_part.strip()
    else:
        lines = output.strip().split('\n')
        if lines:
            program = lines[0].strip()
    return program, answer

# --------- MAIN FUNCTION ---------

def main():
    parser = argparse.ArgumentParser(description="ICL Inference Script with Configurable YAML Path")
    parser.add_argument('--config', type=str, default="configs/icl_config.yaml",
                        help='Path to the configuration YAML file')
    args = parser.parse_args()

    config_path = args.config
    config = load_config(config_path)
    model_conf = config["model"]
    gen_conf = config["generation"]
    data_conf = config["data"]
    icl_conf = config["icl"]

    # --------- FEW-SHOT EXAMPLE SELECTION FROM TRAINING DATA ---------
    with open('data/train.json', 'r') as f:
        train_data = json.load(f)
    num_shots = icl_conf['num_shots']
    selection_method = icl_conf.get('example_selection', 'diverse')
    few_shot_examples = select_few_shot_examples(train_data, num_shots, selection_method)
    config['few_shot_examples'] = finqa_to_yaml_examples(few_shot_examples)

    # --------- MODEL AND TOKENIZER ---------
    tokenizer = AutoTokenizer.from_pretrained(model_conf["model_name_or_path"])
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_conf["model_name_or_path"],
        torch_dtype=getattr(torch, model_conf.get("torch_dtype", "bfloat16")),
        device_map=model_conf.get("device_map", "auto"),
        load_in_8bit=model_conf.get("load_in_8bit", False),
        trust_remote_code=True,
        offload_folder=model_conf.get("offload_folder", "./offload"),
    )
    model.eval()

    # --------- DATA LOADING ---------
    with open(data_conf["test_file"], "r") as f:
        test_data = json.load(f)
    if data_conf.get("max_samples"):
        test_data = test_data[:data_conf["max_samples"]]

    # --------- INFERENCE LOOP ---------
    predictions = []
    for example in tqdm(test_data, desc="Running inference"):
        prompt = create_prompt_from_config(config, example)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=gen_conf["max_new_tokens"],
                temperature=gen_conf["temperature"],
                do_sample=gen_conf.get("do_sample", True),
                top_p=gen_conf.get("top_p", 0.95),
                num_beams=gen_conf.get("num_beams", 1),
                repetition_penalty=gen_conf.get("repetition_penalty", 1.1),
                pad_token_id=tokenizer.eos_token_id,
            )
        generated_text = tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True
        )

        program, answer = parse_model_output(generated_text)
        predictions.append({
            "id": example.get("id", None),
            "question": example["question"],
            "predicted_program": program,
            "predicted_answer": answer,
            "gold_program": " ".join(example["program"]) if isinstance(example.get("program"), list) else example.get("program"),
            "gold_answer": example.get("answer"),
            "raw_output": generated_text
        })

    # --------- SAVE RESULTS ---------
    os.makedirs(os.path.dirname(data_conf["output_file"]), exist_ok=True)
    with open(data_conf["output_file"], 'w') as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Inference complete! Results saved to {data_conf['output_file']}")

if __name__ == "__main__":
    main()