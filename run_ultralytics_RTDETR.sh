#!/bin/bash

# Bash script to run ultralytics_RTDETR.py

# Load the necessary modules
# Replace with actual module names and versions as per your HPC environment
module load python3/3.10.12
# Add any other modules you might need, like CUDA, etc.

# Activate your Python virtual environment
source /zhome/95/b/147257/Desktop/deep_learning_project/deeper/bin/activate

# Navigate to the directory containing your script
cd /zhome/95/b/147257/Desktop/deep_learning_project/ViT

# Run the Python script
python ultralytics_RTDETR.py

# Deactivate the virtual environment
deactivate
