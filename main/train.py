import argparse
import json
import os
from ultralytics import RTDETR, YOLO  
from utils import generate_dynamic_name
import torch
import pandas as pd

# Parse the model type argument
parser = argparse.ArgumentParser()
parser.add_argument("model", help="Specify the model type (RTDETR or YOLO)")
args = parser.parse_args()

# Load configuration from JSON file
with open('../config.json', 'r') as config_file:
    config = json.load(config_file)

# Disable enforcing deterministic algorithms
torch.use_deterministic_algorithms(False)

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


csv_file = f'deep_learning_project/main/runs/detect/{dynamic_name}/results.csv'

# Read the CSV file
df = pd.read_csv(csv_file)

# Plotting each metric
metrics = ['train/giou_loss', 'train/cls_loss', 'train/l1_loss', 'metrics/precision(B)', 'metrics/recall(B)', 
           'metrics/mAP50(B)', 'metrics/mAP50-95(B)', 'val/giou_loss', 'val/cls_loss', 'val/l1_loss', 
           'lr/pg0', 'lr/pg1', 'lr/pg2']

for metric in metrics:
    plt.figure()
    plt.plot(df['epoch'], df[metric], label=metric)
    plt.xlabel('Epoch')
    plt.ylabel(metric)
    plt.title(f'Epoch vs {metric}')
    plt.legend()
    plt.savefig(f'{dynamic_name}_{metric}.png')

print("Plots saved successfully.")

metrics = model.val()
print(metrics)
print("map50: " + str(metrics.box.map50))
print("map75: " + str(metrics.box.map75))
