import os
import json
import pandas as pd
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pycocotools.coco import COCO
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
import cv2
import matplotlib.pyplot as plt
import random
from torchvision import transforms
import torch.nn as nn
import torch
from torch.optim import Adam
from torch.nn import MSELoss, SmoothL1Loss, CrossEntropyLoss
import logging
from tqdm import tqdm
from torchvision import datasets, models
# import the necessary packages
from torch.nn import Dropout
from torch.nn import Identity
from torch.nn import Linear
from torch.nn import Module
from torch.nn import ReLU
from torch.nn import Sequential
from torch.nn import Sigmoid
from torch.optim import Adam

def preprocess_labels(labels):
    # Replace all 'Unknown' (non-background) labels with the index for 'background' (0)
    labels[labels != 0] = 0  # Assuming 0 is the background index
    return labels

def coco_to_voc(coco_boxes):
    voc_boxes = []
    for box in coco_boxes:
        x_min, y_min, width, height = box
        x_max = x_min + width
        y_max = y_min + height
        # Ensure x_max > x_min and y_max > y_min
        x_max = max(x_max, x_min + 1)
        y_max = max(y_max, y_min + 1)
        voc_boxes.append([x_min, y_min, x_max, y_max])
    return voc_boxes

def pad_to_fixed_size(boxes, labels, pad_size=93, image_size=256):
    # Create empty arrays with the fixed size
    padded_boxes = np.full((pad_size, 4), [0, 0, image_size, image_size], dtype=np.float32)
    padded_labels = np.full((pad_size,), 0, dtype=np.int64)  # Use 0 for background

    # Copy original boxes and labels
    num_boxes = min(len(boxes), pad_size)
    if num_boxes > 0:
        padded_boxes[:num_boxes] = boxes[:num_boxes]
        padded_labels[:num_boxes] = labels[:num_boxes]

    return padded_boxes, padded_labels



# Function to get a random training sample
def get_random_sample(dataset):
    random_index = 80 #random.randint(0, len(dataset) - 1)
    image, boxes, labels = dataset[random_index]
    image = image.permute(1, 2, 0).numpy()  # Convert from torch format to numpy format for plotting
    return image, boxes, labels

# Function to Save Plots
def save_plot(image, voc_boxes, image_name):
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    for box in voc_boxes:
        # Create a rectangle patch for the bounding box
        x_min, y_min, x_max, y_max = box
        rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)
    plt.axis('on')
    plt.savefig(f'plots/{image_name}.png', bbox_inches='tight')
    plt.close()
# Define paths and parameters
dataset_path = "/dtu/blackhole/19/147257/deep/coco/"
anns_file_train = dataset_path + "annotations/instances_train2017.json"
anns_file_val = dataset_path + "annotations/instances_val2017.json"
image_size = 256
batch_size = 128
num_classes = 81

# Custom Dataset Class
class CocoDataset(Dataset):
    def __init__(self, root, annotation, transform=None):
        self.root = root
        self.coco = COCO(annotation)
        self.image_ids = list(self.coco.imgs.keys())
        self.transform = transform

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        image_info = self.coco.loadImgs(image_id)[0]
        image_path = os.path.join(self.root, image_info['file_name'])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        annotations = self.coco.loadAnns(self.coco.getAnnIds(imgIds=image_id))
        boxes = np.array([ann['bbox'] for ann in annotations])
        labels = np.array([ann['category_id'] for ann in annotations])
        
        # Shift labels by +1
        labels = labels + 1

        # Convert COCO format boxes to Pascal VOC format before transformation
        boxes = coco_to_voc(boxes)

        if self.transform:
            # Apply transformations to the image and bounding boxes
            transformed = self.transform(image=image, bboxes=boxes, labels=labels)
            image = transformed['image']
            boxes = transformed['bboxes']
            labels = transformed['labels']

        boxes, labels = pad_to_fixed_size(boxes, labels)

        # Convert boxes and labels to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        return image, boxes, labels

    def __len__(self):
        return len(self.image_ids)

# Transformations using Albumentations
transform = A.Compose([
    A.Resize(width=image_size, height=image_size),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'], min_area=10, min_visibility=0.5))

# Dataset instances
dataset_path_x = "/dtu/blackhole/19/147257/coco_data/images/"

train_dataset = CocoDataset(root=dataset_path_x + "train2017/", annotation=anns_file_train, transform=transform)
val_dataset = CocoDataset(root=dataset_path_x + "val2017/", annotation=anns_file_val, transform=transform)

# Dataloaders
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)


# Load and process categories
# Load class names and IDs from the COCO dataset
# Load and process categories
with open(anns_file_train, 'r') as f:
    dataset = json.loads(f.read())
categories_df = pd.DataFrame(dataset['categories'])
categories_df['label'] = categories_df['id'].astype(int) + 1
categories_dict = dict(zip(categories_df['label'], categories_df['name']))
categories_dict[0] = "background"

# Validate the dictionary
categories_dict = {0: "background"}  # Background class
idx = 1  # Start indexing from 1 for actual classes

categories = [
    {"id": 1, "name": "person"},
    {"id": 2, "name": "bicycle"},
    {"id": 3, "name": "car"},
    {"id": 4, "name": "motorcycle"},
    {"id": 5, "name": "airplane"},
    {"id": 6, "name": "bus"},
    {"id": 7, "name": "train"},
    {"id": 8, "name": "truck"},
    {"id": 9, "name": "boat"},
    {"id": 10, "name": "traffic light"},
    {"id": 11, "name": "fire hydrant"},
    {"id": 12, "name": "stop sign"},
    {"id": 13, "name": "parking meter"},
    {"id": 14, "name": "bench"},
    {"id": 15, "name": "bird"},
    {"id": 16, "name": "cat"},
    {"id": 17, "name": "dog"},
    {"id": 18, "name": "horse"},
    {"id": 19, "name": "sheep"},
    {"id": 20, "name": "cow"},
    {"id": 21, "name": "elephant"},
    {"id": 22, "name": "bear"},
    {"id": 23, "name": "zebra"},
    {"id": 24, "name": "giraffe"},
    {"id": 25, "name": "backpack"},
    {"id": 26, "name": "umbrella"},
    {"id": 27, "name": "handbag"},
    {"id": 28, "name": "tie"},
    {"id": 29, "name": "suitcase"},
    {"id": 30, "name": "frisbee"},
    {"id": 31, "name": "skis"},
    {"id": 32, "name": "snowboard"},
    {"id": 33, "name": "sports ball"},
    {"id": 34, "name": "kite"},
    {"id": 35, "name": "baseball bat"},
    {"id": 36, "name": "baseball glove"},
    {"id": 37, "name": "skateboard"},
    {"id": 38, "name": "surfboard"},
    {"id": 39, "name": "tennis racket"},
    {"id": 40, "name": "bottle"},
    {"id": 41, "name": "wine glass"},
    {"id": 42, "name": "cup"},
    {"id": 43, "name": "fork"},
    {"id": 44, "name": "knife"},
    {"id": 45, "name": "spoon"},
    {"id": 46, "name": "bowl"},
    {"id": 47, "name": "banana"},
    {"id": 48, "name": "apple"},
    {"id": 49, "name": "sandwich"},
    {"id": 50, "name": "orange"},
    {"id": 51, "name": "broccoli"},
    {"id": 52, "name": "carrot"},
    {"id": 53, "name": "hot dog"},
    {"id": 54, "name": "pizza"},
    {"id": 55, "name": "donut"},
    {"id": 56, "name": "cake"},
    {"id": 57, "name": "chair"},
    {"id": 58, "name": "couch"},
    {"id": 59, "name": "potted plant"},
    {"id": 60, "name": "bed"},
    {"id": 61, "name": "dining table"},
    {"id": 62, "name": "toilet"},
    {"id": 63, "name": "tv"},
    {"id": 64, "name": "laptop"},
    {"id": 65, "name": "mouse"},
    {"id": 66, "name": "remote"},
    {"id": 67, "name": "keyboard"},
    {"id": 68, "name": "cell phone"},
    {"id": 69, "name": "microwave"},
    {"id": 70, "name": "oven"},
    {"id": 71, "name": "toaster"},
    {"id": 72, "name": "sink"},
    {"id": 73, "name": "refrigerator"},
    {"id": 74, "name": "book"},
    {"id": 75, "name": "clock"},
    {"id": 76, "name": "vase"},
    {"id": 77, "name": "scissors"},
    {"id": 78, "name": "teddy bear"},
    {"id": 79, "name": "hair drier"},
    {"id": 80, "name": "toothbrush"}
]

for category in categories:
    if category['name'].lower() != "unknown":
        categories_dict[category['id']] = idx
        idx += 1

# Validate the dictionary
for label in range(len(categories_dict)):
    print(f"{label}: {categories_dict.get(label, 'Unknown')}")

# Now categories_dict should correctly map all COCO class IDs to names, with no 'Unknown' entries


# Load a random sample
image, boxes, labels = get_random_sample(train_dataset)

# Plot the image and bounding boxes
image_name = f"sample_{random.randint(1, 10)}"
save_plot(image, boxes, image_name)

print(image.shape)

# Print labels
print("Labels and boxes:")
for label, box in zip(labels, boxes):  # Use zip to iterate over labels and boxes together
    if label.item() > 0:  # Check if the label is greater than 0 to ensure it's not a padded label
        label_name = categories_dict.get(label.item(), "Unknown")
        print(f" - {label_name}: {box.tolist()}")  # Convert the box to a list for readability
    #else:
    #    print(" - background or padding")


# Assuming categories_df is already loaded from the COCO dataset
categories_df = pd.DataFrame(dataset['categories'])
categories_df['label'] = categories_df['id'].astype(int) + 1  # Shift labels by +1
categories_dict = dict(zip(categories_df['label'], categories_df['name']))

# Add background class
categories_dict[0] = "background"

all_labels = []
for label in range(92):  # 80 object categories + 1 background class
    print(f"{label}: {categories_dict.get(label, 'Unknown')}")
    all_labels.append(categories_dict.get(label, 'Unknown'))
#[203.9, 87.33, 107.09, 259.59, 1]

print("init model...")

# Model
class ObjectDetector(Module):
    def __init__(self, baseModel, numClasses):
        super(ObjectDetector, self).__init__()
        # Initialize the base model and the number of classes
        self.baseModel = baseModel
        self.numClasses = numClasses
        self.num_ftrs = baseModel.fc.in_features

        # Build the regressor head for outputting the bounding box coordinates
        # Adjust the output layer to predict 93 boxes each with 4 coordinates
        self.regressor = Sequential(
            Linear(self.num_ftrs, 128),
            ReLU(),
            Linear(128, 64),
            ReLU(),
            Linear(64, 32),
            ReLU(),
            Linear(32, 93 * 4),  # Output 93 * 4 values for bounding boxes
            Sigmoid())

        # Build the classifier head to predict the class labels
        # Here, we predict class labels for each of the 93 boxes
        self.classifier = Sequential(
            Linear(self.num_ftrs, 93 * self.numClasses),  # Output 93 * numClasses values for class labels
        )

        # Set the classifier of our base model to produce outputs from the last convolution block
        self.baseModel.fc = Identity()

    def forward(self, x):
        # Pass the inputs through the base model and then obtain predictions from two different branches of the network
        features = self.baseModel(x)

        # Reshape the output to [batch_size, 93, 4] for bounding boxes
        bboxes = self.regressor(features).view(-1, 93, 4)

        # Reshape the output to [batch_size, 93, numClasses] for class labels
        classLogits = self.classifier(features).view(-1, 93, self.numClasses)

        # Return the outputs as a tuple
        return (bboxes, classLogits)

    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# load the ResNet50 network
resnet = models.resnet152(weights=models.ResNet152_Weights.DEFAULT).to(device)#resnet152(weights=ResNet152_Weights.IMAGENET1K_V1)
# freeze all ResNet50 layers so they will *not* be updated during the
# training process
for param in resnet.parameters():
    param.requires_grad = False
# create our custom object detector model and flash it to the current
# device
model = ObjectDetector(resnet, num_classes)

model = model.to(device)

# Define loss functions
regression_loss_fn = SmoothL1Loss()
classification_loss_fn = CrossEntropyLoss()

# Optimizer
optimizer = Adam(model.parameters(), lr=1e-3)

# Number of epochs
num_epochs = 1

print("starting train...")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Training loop
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    total_loss = 0.0
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs}")

    for i, (images, bboxes, labels) in progress_bar:
        images, bboxes, labels = images.to(device), bboxes.to(device), labels.to(device)
        labels = preprocess_labels(labels)

        # Forward pass
        predicted_bboxes, predicted_labels = model(images)
        predicted_labels_flat = predicted_labels.view(-1, num_classes)
        labels_flat = labels.view(-1)

        # Compute losses
        bbox_loss = regression_loss_fn(predicted_bboxes, bboxes)
        class_loss = classification_loss_fn(predicted_labels_flat, labels_flat)
        loss = bbox_loss + class_loss

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Update tqdm progress bar
        progress_bar.set_postfix(loss=total_loss / (i + 1))

    # Log epoch summary
    avg_loss = total_loss / len(train_loader)
    logging.info(f'Epoch [{epoch + 1}/{num_epochs}], Average Loss: {avg_loss:.4f}')
