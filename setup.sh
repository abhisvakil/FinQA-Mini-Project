#!/bin/bash

# Setup script for FinQA Mini Project - PEFT vs ICL

echo "Setting up FinQA Mini Project (PEFT vs ICL)..."

# Create necessary directories
echo "Creating project directories..."
mkdir -p data
mkdir -p results/lora results/qlora results/icl
mkdir -p src notebooks configs logs

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Check if FinQA repository exists
if [ ! -d "FinQA" ]; then
    echo "Cloning FinQA repository..."
    git clone https://github.com/czyssrs/FinQA.git
    echo "Note: You'll need to copy dataset files from FinQA/dataset/ to data/ directory"
fi

echo ""
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. Copy dataset files: cp FinQA/dataset/*.json data/"
echo "2. Activate virtual environment: source venv/bin/activate"
echo "3. Start exploring: jupyter notebook notebooks/data_exploration.ipynb"
echo ""

