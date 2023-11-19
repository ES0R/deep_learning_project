import json
import os
from ultralytics import RTDETR
from utils import generate_dynamic_name 

# Load configuration from JSON file
with open('../config.json', 'r') as config_file:
    config = json.load(config_file)

# Load a COCO-pretrained RT-DETR-l model
model = RTDETR(config['model'])

# Display model information (optional)
model.info()

# Generate a dynamic name based on the model
dynamic_name = generate_dynamic_name(config['model'])

# Train the model using parameters from the config file
results = model.train(
    task=config['task'],
    mode=config['mode'],
    data=config['data'],
    epochs=config['epochs'],
    batch=config['batch'],
    imgsz=config['imgsz'],
    iou=config['iou'],
    name=dynamic_name
)
