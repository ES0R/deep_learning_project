import argparse
import json
import os
from ultralytics import RTDETR, YOLO  
from utils import generate_dynamic_name

# Parse the model type argument
parser = argparse.ArgumentParser()
parser.add_argument("model", help="Specify the model type (RTDETR or YOLO)")
args = parser.parse_args()

# Load configuration from JSON file
with open('../config.json', 'r') as config_file:
    config = json.load(config_file)


# Initialize model
if args.model.lower().startswith("rtdetr"):
    model = RTDETR(args.model)
elif args.model.lower().startswith("yolo"):
    model = YOLO(args.model)   
else:
    raise ValueError("Unsupported model type. Please use RTDETR or YOLO.")

# Display model 
#model.info()

# Generate a dynamic name
dynamic_name = generate_dynamic_name(args.model.lower())

# Train the model using parameters from the config file
results = model.train(
    task=config['task'],
    mode=config['mode'],
    data=config['data'],
    epochs=config['epochs'],
    batch=config['batch'],
    imgsz=config['imgsz'],
    iou=config['iou'],
    name=dynamic_name,
    classes=config['classes']
)