#!/bin/bash

# FinQA Complete Experiment Pipeline
# Handles training AND inference for all methods

set -e  # Exit on error

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Paths
DATA_DIR="data/simplified"
RESULTS_DIR="results"
PRED_DIR="$RESULTS_DIR/predictions"

# Models
LLAMA_MODEL="meta-llama/Meta-Llama-3-8B-Instruct"
MISTRAL_MODEL="mistralai/Mistral-7B-Instruct-v0.2"

echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}       FinQA PEFT vs ICL - Complete Pipeline${NC}"
echo -e "${BLUE}============================================================${NC}\n"

function show_usage() {
    echo "Usage: ./run_all_experiments.sh [command]"
    echo ""
    echo "Training Commands:"
    echo "  train-lora-llama      - Train LoRA on Llama"
    echo "  train-lora-mistral    - Train LoRA on Mistral"
    echo "  train-qlora-llama     - Train QLoRA on Llama"
    echo "  train-qlora-mistral   - Train QLoRA on Mistral"
    echo "  train-all             - Train all (LoRA + QLoRA, both models)"
    echo ""
    echo "Inference Commands:"
    echo "  infer-lora-llama      - Inference with LoRA Llama"
    echo "  infer-lora-mistral    - Inference with LoRA Mistral"
    echo "  infer-qlora-llama     - Inference with QLoRA Llama"
    echo "  infer-qlora-mistral   - Inference with QLoRA Mistral"
    echo "  infer-icl-llama       - ICL inference with Llama"
    echo "  infer-icl-mistral     - ICL inference with Mistral"
    echo "  infer-all             - All inference (6 total)"
    echo ""
    echo "Full Pipeline:"
    echo "  full                  - Train all + Infer all"
    echo ""
    echo "Quick Test:"
    echo "  test                  - Quick test with 100 samples"
}

# Training functions
function train_lora_llama() {
    echo -e "${GREEN}[TRAIN] LoRA on Llama-3-8B...${NC}"
    cd src
    python train_lora.py \
        --model_name "$LLAMA_MODEL" \
        --data_dir "../$DATA_DIR" \
        --output_dir "../$RESULTS_DIR/lora" \
        --epochs 3 \
        --batch_size 4 \
        --gradient_accumulation_steps 4 \
        --learning_rate 2e-4
    cd ..
    echo -e "${GREEN}✓ LoRA Llama training complete${NC}\n"
}

function train_lora_mistral() {
    echo -e "${GREEN}[TRAIN] LoRA on Mistral-7B...${NC}"
    cd src
    python train_lora.py \
        --model_name "$MISTRAL_MODEL" \
        --data_dir "../$DATA_DIR" \
        --output_dir "../$RESULTS_DIR/lora" \
        --epochs 3 \
        --batch_size 4 \
        --gradient_accumulation_steps 4 \
        --learning_rate 2e-4
    cd ..
    echo -e "${GREEN}✓ LoRA Mistral training complete${NC}\n"
}

function train_qlora_llama() {
    echo -e "${GREEN}[TRAIN] QLoRA on Llama-3-8B...${NC}"
    cd src
    python train_qlora.py \
        --model_name "$LLAMA_MODEL" \
        --data_dir "../$DATA_DIR" \
        --output_dir "../$RESULTS_DIR/qlora" \
        --epochs 3 \
        --batch_size 4 \
        --gradient_accumulation_steps 4 \
        --learning_rate 2e-4
    cd ..
    echo -e "${GREEN}✓ QLoRA Llama training complete${NC}\n"
}

function train_qlora_mistral() {
    echo -e "${GREEN}[TRAIN] QLoRA on Mistral-7B...${NC}"
    cd src
    python train_qlora.py \
        --model_name "$MISTRAL_MODEL" \
        --data_dir "../$DATA_DIR" \
        --output_dir "../$RESULTS_DIR/qlora" \
        --epochs 3 \
        --batch_size 4 \
        --gradient_accumulation_steps 4 \
        --learning_rate 2e-4
    cd ..
    echo -e "${GREEN}✓ QLoRA Mistral training complete${NC}\n"
}

# Inference functions
function infer_lora_llama() {
    echo -e "${YELLOW}[INFER] LoRA Llama...${NC}"
    cd src
    python inference.py \
        --model_name "$LLAMA_MODEL" \
        --adapter_path "../$RESULTS_DIR/lora/Meta-Llama-3-8B-Instruct/final_model" \
        --method "lora" \
        --data_dir "../$DATA_DIR" \
        --output_dir "../$PRED_DIR" \
        --temperature 0.1
    cd ..
    echo -e "${YELLOW}✓ LoRA Llama inference complete${NC}\n"
}

function infer_lora_mistral() {
    echo -e "${YELLOW}[INFER] LoRA Mistral...${NC}"
    cd src
    python inference.py \
        --model_name "$MISTRAL_MODEL" \
        --adapter_path "../$RESULTS_DIR/lora/Mistral-7B-Instruct-v0.2/final_model" \
        --method "lora" \
        --data_dir "../$DATA_DIR" \
        --output_dir "../$PRED_DIR" \
        --temperature 0.1
    cd ..
    echo -e "${YELLOW}✓ LoRA Mistral inference complete${NC}\n"
}

function infer_qlora_llama() {
    echo -e "${YELLOW}[INFER] QLoRA Llama...${NC}"
    cd src
    python inference.py \
        --model_name "$LLAMA_MODEL" \
        --adapter_path "../$RESULTS_DIR/qlora/Meta-Llama-3-8B-Instruct/final_model" \
        --method "qlora" \
        --data_dir "../$DATA_DIR" \
        --output_dir "../$PRED_DIR" \
        --temperature 0.1 \
        --load_in_8bit
    cd ..
    echo -e "${YELLOW}✓ QLoRA Llama inference complete${NC}\n"
}

function infer_qlora_mistral() {
    echo -e "${YELLOW}[INFER] QLoRA Mistral...${NC}"
    cd src
    python inference.py \
        --model_name "$MISTRAL_MODEL" \
        --adapter_path "../$RESULTS_DIR/qlora/Mistral-7B-Instruct-v0.2/final_model" \
        --method "qlora" \
        --data_dir "../$DATA_DIR" \
        --output_dir "../$PRED_DIR" \
        --temperature 0.1 \
        --load_in_8bit
    cd ..
    echo -e "${YELLOW}✓ QLoRA Mistral inference complete${NC}\n"
}

function infer_icl_llama() {
    echo -e "${YELLOW}[INFER] ICL Llama...${NC}"
    cd src
    python icl_inference.py \
        --model_name "$LLAMA_MODEL" \
        --data_dir "../$DATA_DIR" \
        --output_dir "../$PRED_DIR" \
        --num_shots 5 \
        --temperature 0.1
    cd ..
    echo -e "${YELLOW}✓ ICL Llama inference complete${NC}\n"
}

function infer_icl_mistral() {
    echo -e "${YELLOW}[INFER] ICL Mistral...${NC}"
    cd src
    python icl_inference.py \
        --model_name "$MISTRAL_MODEL" \
        --data_dir "../$DATA_DIR" \
        --output_dir "../$PRED_DIR" \
        --num_shots 5 \
        --temperature 0.1
    cd ..
    echo -e "${YELLOW}✓ ICL Mistral inference complete${NC}\n"
}

# Main command handling
case "$1" in
    # Training
    train-lora-llama) train_lora_llama ;;
    train-lora-mistral) train_lora_mistral ;;
    train-qlora-llama) train_qlora_llama ;;
    train-qlora-mistral) train_qlora_mistral ;;
    
    train-all)
        echo -e "${BLUE}Training all methods (will take 10-14 hours on g5.2xlarge)${NC}\n"
        train_lora_llama
        train_lora_mistral
        train_qlora_llama
        train_qlora_mistral
        echo -e "${GREEN}✓✓✓ All training complete!${NC}"
        ;;
    
    # Inference
    infer-lora-llama) infer_lora_llama ;;
    infer-lora-mistral) infer_lora_mistral ;;
    infer-qlora-llama) infer_qlora_llama ;;
    infer-qlora-mistral) infer_qlora_mistral ;;
    infer-icl-llama) infer_icl_llama ;;
    infer-icl-mistral) infer_icl_mistral ;;
    
    infer-all)
        echo -e "${BLUE}Running all inference (will take 2-3 hours)${NC}\n"
        infer_lora_llama
        infer_lora_mistral
        infer_qlora_llama
        infer_qlora_mistral
        infer_icl_llama
        infer_icl_mistral
        echo -e "${GREEN}✓✓✓ All inference complete!${NC}"
        echo -e "${BLUE}Predictions saved to: $PRED_DIR/${NC}"
        ;;
    
    # Full pipeline
    full)
        echo -e "${BLUE}Running FULL pipeline (train + infer)${NC}"
        echo -e "${BLUE}This will take 12-17 hours on g5.2xlarge${NC}\n"
        
        # Training
        train_lora_llama
        train_lora_mistral
        train_qlora_llama
        train_qlora_mistral
        
        # Inference
        infer_lora_llama
        infer_lora_mistral
        infer_qlora_llama
        infer_qlora_mistral
        infer_icl_llama
        infer_icl_mistral
        
        echo -e "${GREEN}✓✓✓ FULL PIPELINE COMPLETE!${NC}"
        echo -e "${BLUE}Next: Run evaluation with src/evaluate.py${NC}"
        ;;
    
    # Quick test
    test)
        echo -e "${YELLOW}Running quick test (100 samples)${NC}"
        cd src
        python train_lora.py --model_name "$LLAMA_MODEL" --data_dir "../$DATA_DIR" --output_dir "../$RESULTS_DIR/test" --epochs 1 --max_samples 100
        cd ..
        echo -e "${GREEN}✓ Test complete${NC}"
        ;;
    
    *)
        show_usage
        exit 1
        ;;
esac
