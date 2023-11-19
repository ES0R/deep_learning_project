#!/bin/bash

# Bash script to run a training script with a specified model

# Check if a model type argument is provided
if [ -z "$1" ]; then
    echo "Error: No model type specified. Usage: $0 [MODEL_TYPE]"
    exit 1
fi

MODEL_TYPE=$1
MODEL_FILE=""

# Set the model file based on the model type
if [ "$MODEL_TYPE" == "YOLO" ]; then
    MODEL_FILE="yolo-model-file.pt"  # Replace with actual YOLO model file
elif [ "$MODEL_TYPE" == "RTDETR" ]; then
    MODEL_FILE="rtdetr-l.pt"  # Replace with actual RTDETR model file
else
    echo "Error: Invalid model type. Use 'YOLO' or 'RTDETR'."
    exit 1
fi

# Set the project directory to the current working directory
PROJECT_DIR=$(pwd)

# Update the model in config.json
CONFIG_PATH="$PROJECT_DIR/config.json"
if [ -f "$CONFIG_PATH" ]; then
    # Use jq to update the model in config.json
    jq --arg model "$MODEL_FILE" '.model = $model' "$CONFIG_PATH" > temp.json && mv temp.json "$CONFIG_PATH"
else
    echo "Error: config.json not found."
    exit 1
fi

# Load the necessary modules
module load python3/3.10.12

# Activate your Python virtual environment
source "$PROJECT_DIR/deeper/bin/activate"

# Navigate to the directory containing your script
cd "$PROJECT_DIR/$MODEL_TYPE"

# Run the Python script
python train.py

# Deactivate the virtual environment
deactivate
