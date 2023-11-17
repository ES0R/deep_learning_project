
from ultralytics import RTDETR

# Load a COCO-pretrained RT-DETR-l model
model = RTDETR('rtdetr-l.pt')

# Display model information (optional)
model.info()

# Train the model on the COCO8 example dataset for 100 epochs
results = model.train(data='coco8.yaml', epochs = 1, imgsz = 640, iou=0.9)


# Access the confusion matrix
# Access the confusion matrix object
conf_matrix_obj = results.confusion_matrix

# Check available methods or attributes
print(dir(conf_matrix_obj))

## Data
# Get coco to work on HPC
# Classes should be changed to more appropiate 

## Important parameters for presentation
# Inference time
# mAP



