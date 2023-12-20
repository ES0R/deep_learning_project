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
results = model.train(classes=[0, 1, 2, 3, 9],
    task=config['task'],
    mode=config['mode'],
    data=config['data'],
    epochs=config['epochs'],
    batch=config['batch'],
    imgsz=config['imgsz'],
    iou=config['iou'],
    name=dynamic_name,
    dropout=config['dropout'],
    weight_decay=config['weight_decay'],
    optimizer=config['optimizer'],
    lr0=config['lr0'],
    fliplr=config['fliplr'],
    translate=config['translate'],
    scale=config['scale'],
    hsv_h=0, 
    hsv_s=0, 
    hsv_v=0,
    mosaic=0
)

metrics = model.val()
print("map50: " + str(metrics.box.map50))
print("map75: " + str(metrics.box.map75))