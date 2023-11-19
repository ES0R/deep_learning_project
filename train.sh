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
    MODEL_FILE="yolov8n.pt" 
elif [ "$MODEL_TYPE" == "RTDETR" ]; then
    MODEL_FILE="rtdetr-l.pt"  
else
    echo "Error: Invalid model type. Use 'YOLO' or 'RTDETR'."
    exit 1
fi

# Set the project directory to the current working directory
PROJECT_DIR=$(pwd)

# Load the necessary modules
module load python3/3.10.12

# Activate your Python virtual environment
source "$PROJECT_DIR/deeper/bin/activate"

# Navigate to the directory containing your script
cd "$PROJECT_DIR/main"

# Run the Python script with the model file as an argument
python train.py "$MODEL_FILE"

# Deactivate the virtual environment
deactivate
