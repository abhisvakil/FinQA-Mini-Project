#!/bin/bash

# Quick test script for ICL inference on 10 examples
# Usage: ./test_icl_inference.sh <model_name>
# Example: ./test_icl_inference.sh mistral  (or llama)

set -e

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Parse argument
MODEL_TYPE=${1:-mistral}

if [ "$MODEL_TYPE" = "mistral" ]; then
    MODEL_NAME="mistralai/Mistral-7B-Instruct-v0.2"
elif [ "$MODEL_TYPE" = "llama" ]; then
    MODEL_NAME="meta-llama/Meta-Llama-3-8B-Instruct"
else
    echo "Usage: ./test_icl_inference.sh [mistral|llama]"
    exit 1
fi

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Testing ICL Inference on 10 Examples${NC}"
echo -e "${GREEN}Model: $MODEL_NAME${NC}"
echo -e "${GREEN}========================================${NC}\n"

# Run inference with max_samples=10
cd src
python icl_inference.py \
    --config ../configs/icl_config_1.yaml \
    --model_name "$MODEL_NAME" \
    --data_dir data/simplified \
    --output_dir ../results_test \
    --max_samples 10

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}Test Complete!${NC}"
echo -e "${GREEN}========================================${NC}"

# Extract model short name
MODEL_SHORT=$(echo $MODEL_NAME | cut -d'/' -f2)

# Show where results are saved
RESULTS_DIR="../results_test/icl/${MODEL_SHORT}/5shot_diverse"
echo -e "\n${YELLOW}Results saved to:${NC}"
echo "  $RESULTS_DIR/predictions.json"
echo "  $RESULTS_DIR/config.yaml"
echo "  $RESULTS_DIR/metadata.json"

# Show first prediction
echo -e "\n${YELLOW}First prediction:${NC}"
python3 << EOF
import json
with open('$RESULTS_DIR/predictions.json', 'r') as f:
    preds = json.load(f)
    if preds:
        p = preds[0]
        print(f"Question: {p['question'][:80]}...")
        print(f"Predicted program: {p['predicted_program']}")
        print(f"Gold program: {p['gold_program']}")
        print(f"Match: {p['predicted_program'] == p['gold_program']}")
EOF

echo -e "\n${GREEN}To view all predictions:${NC}"
echo "  cat $RESULTS_DIR/predictions.json | jq '.[] | {question, predicted_program, gold_program}'"
