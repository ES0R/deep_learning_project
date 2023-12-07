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

def coco_to_voc(coco_boxes):
    voc_boxes = []
    for box in coco_boxes:
        x_min, y_min, width, height = box
        x_max = x_min + width
        y_max = y_min + height
        voc_boxes.append([x_min, y_min, x_max, y_max])
    return voc_boxes

def pad_to_fixed_size(boxes, labels, pad_size=93, pad_value=0):
    # Create empty arrays with the fixed size, filled with pad_value for boxes or -1 for labels
    padded_boxes = np.full((pad_size, 4), pad_value, dtype=np.float32)
    padded_labels = np.full((pad_size,), 0, dtype=np.int64)  # Now 0 is used for background/padding

    # Calculate how many items to copy from the original boxes and labels
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
batch_size = 8
num_classes = 80

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
    ToTensorV2()
], bbox_params=A.BboxParams(format='pascal_voc', min_area=500, min_visibility=0.5, label_fields=['labels']))

# Dataset instances
dataset_path_x = "/dtu/blackhole/19/147257/coco_data/images/"

train_dataset = CocoDataset(root=dataset_path_x + "train2017/", annotation=anns_file_train, transform=transform)
val_dataset = CocoDataset(root=dataset_path_x + "val2017/", annotation=anns_file_val, transform=transform)

# Dataloaders
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)


# Load and process categories
with open(anns_file_train, 'r') as f:
    dataset = json.loads(f.read())
categories_df = pd.DataFrame(dataset['categories'])
categories_df['label'] = categories_df['id'].astype(int) + 1
categories_dict = dict(zip(categories_df['label'], categories_df['name']))
categories_dict[0] = "background"



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


#[203.9, 87.33, 107.09, 259.59, 1]

print("init model...")

# Model
class ObjectDetector(Module):
    def __init__(self, baseModel, numClasses):
        super(ObjectDetector, self).__init__()
        # initialize the base model and the number of classes
        self.baseModel = baseModel
        self.numClasses = numClasses
        self.num_ftrs = baseModel.fc.in_features
        # build the regressor head for outputting the bounding boxcoordinates
        self.regressor = Sequential(
            Linear(self.num_ftrs, 128),
            ReLU(),
            Linear(128, 64),
            ReLU(),
            Linear(64, 32),
            ReLU(),
            Linear(32, 4),
            Sigmoid())
        # build the classifier head to predict the class labels
        self.classifier = Sequential(
            Linear(self.num_ftrs, self.numClasses),
        )
        # set the classifier of our base model to produce outputs
        # from the last convolution block
        self.baseModel.fc = Identity()
    def forward(self, x):
        # pass the inputs through the base model and then obtain
        # predictions from two different branches of the network
        features = self.baseModel(x)
        bboxes = self.regressor(features)
        classLogits = self.classifier(features)
        # return the outputs as a tuple
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

# Training loop
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    for i, (images, bboxes, labels) in enumerate(train_loader):
        images, bboxes, labels = images.to(device), bboxes.to(device), labels.to(device)

        # Forward pass
        predicted_bboxes, predicted_labels = model(images)

        # Compute losses
        bbox_loss = regression_loss_fn(predicted_bboxes, bboxes)
        class_loss = classification_loss_fn(predicted_labels, labels)

        # Combine losses
        total_loss = bbox_loss + class_loss

        # Backward pass and optimize
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if i % 10 == 0:  # Print loss every 10 batches
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Total Loss: {total_loss.item()}")
