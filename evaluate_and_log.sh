#!/bin/bash
# Wrapper script to run check_accuracy_simple.py and log results to CSV

# Usage: ./evaluate_and_log.sh <predictions_file> <model_name> <config_name> <num_shots>
# Example: ./evaluate_and_log.sh results/icl_config_1/Mistral-7B-Instruct-v0.2_icl_predictions_latest.json Mistral-7B-Instruct-v0.2 config_1 5

if [ "$#" -lt 4 ]; then
    echo "Usage: $0 <predictions_file> <model_name> <config_name> <num_shots>"
    echo "Example: $0 results_test/Mistral-7B-Instruct-v0.2_icl_predictions_latest.json Mistral-7B-Instruct-v0.2 config_1 5"
    exit 1
fi

PREDICTIONS_FILE="$1"
MODEL_NAME="$2"
CONFIG_NAME="$3"
NUM_SHOTS="$4"

# CSV file to store results
CSV_FILE="evaluation_results.csv"

# Create CSV header if file doesn't exist
if [ ! -f "$CSV_FILE" ]; then
    echo "timestamp,model_name,config_name,num_shots,program_accuracy,answer_accuracy,total_samples" > "$CSV_FILE"
fi

# Run check_accuracy_simple.py and capture output
echo "Running evaluation on: $PREDICTIONS_FILE"
OUTPUT=$(python3 check_accuracy_simple.py "$PREDICTIONS_FILE" 2>&1)

# Display output to user
echo "$OUTPUT"

# Extract accuracy numbers from output
# Looking for lines like: "Results: 3/50 correct programs (6.0%)"
PROGRAM_LINE=$(echo "$OUTPUT" | grep "correct programs")
ANSWER_LINE=$(echo "$OUTPUT" | grep "correct answers")

# Extract numbers using regex
PROGRAM_CORRECT=$(echo "$PROGRAM_LINE" | grep -oE '[0-9]+/[0-9]+' | head -1 | cut -d'/' -f1)
PROGRAM_TOTAL=$(echo "$PROGRAM_LINE" | grep -oE '[0-9]+/[0-9]+' | head -1 | cut -d'/' -f2)
PROGRAM_PCT=$(echo "$PROGRAM_LINE" | grep -oE '\([0-9]+\.[0-9]+%\)' | tr -d '()%')

ANSWER_CORRECT=$(echo "$ANSWER_LINE" | grep -oE '[0-9]+/[0-9]+' | head -1 | cut -d'/' -f1)
ANSWER_TOTAL=$(echo "$ANSWER_LINE" | grep -oE '[0-9]+/[0-9]+' | head -1 | cut -d'/' -f2)
ANSWER_PCT=$(echo "$ANSWER_LINE" | grep -oE '\([0-9]+\.[0-9]+%\)' | tr -d '()%')

# Get timestamp
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

# Append to CSV
if [ -n "$PROGRAM_PCT" ] && [ -n "$ANSWER_PCT" ]; then
    echo "$TIMESTAMP,$MODEL_NAME,$CONFIG_NAME,$NUM_SHOTS,$PROGRAM_PCT,$ANSWER_PCT,$PROGRAM_TOTAL" >> "$CSV_FILE"
    echo ""
    echo "✓ Results logged to $CSV_FILE"
    echo "  Model: $MODEL_NAME"
    echo "  Config: $CONFIG_NAME"
    echo "  Shots: $NUM_SHOTS"
    echo "  Program Accuracy: $PROGRAM_PCT%"
    echo "  Answer Accuracy: $ANSWER_PCT%"
else
    echo "⚠ Warning: Could not extract accuracy metrics from output"
    exit 1
fi
