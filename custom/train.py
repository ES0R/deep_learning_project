import torch
from torchvision import transforms
from data_loader import COCODataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import ObjectDetectionModel
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import torch
from torch.utils.data import Dataset
import os
import cv2
from torchvision import transforms
import json
from PIL import Image, ImageOps
import warnings

warnings.filterwarnings('ignore', category=UserWarning, module='torchvision')
########

def collate_fn(batch):
    # Filter out the placeholders
    batch = [data for data in batch if data[0].nelement() != 0 and data[1].nelement() != 0 and data[2].nelement() != 0]

    if len(batch) == 0:
        return torch.zeros(0, 3, 224, 224), torch.zeros(0, 4), torch.zeros(0, dtype=torch.int64)

    images, bboxes, labels = zip(*batch)
    images = torch.stack(images, dim=0)
    bboxes = tuple(bboxes)
    labels = tuple(labels)

    return images, bboxes, labels



# Define image transformations
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor()
])


# Paths to image folders and annotation files
print("Starting data load")
images_train_folder = '/dtu/blackhole/19/147257/deep/coco/images/train2017/' 
images_val_folder = '/dtu/blackhole/19/147257/deep/coco/images/val2017/' 
anno_train_file = '/dtu/blackhole/19/147257/deep/coco/annotations/instances_train2017.json'
anno_val_file = '/dtu/blackhole/19/147257/deep/coco/annotations/instances_val2017.json'

# Create dataset instances
train_dataset = COCODataset(root=images_train_folder, annotation_file=anno_train_file, transform=transform)
val_dataset = COCODataset(root=images_val_folder, annotation_file=anno_val_file, transform=transform)

# DataLoader parameters
batch_size = 64
shuffle = True
num_workers = 4 # Adjust as per your machine's capability

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)#,collate_fn=collate_fn)
print("Finished data load")


# Check if GPU is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the model
num_classes = 80  # Number of classes in COCO dataset
model = ObjectDetectionModel(num_classes=num_classes).to(device)

#Optimizer
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)

sample = train_dataset[28095]  
image_tensor, bboxes, labels = sample
print(f"1: {sample[0].shape}, 2: {sample[1].shape}, 3: {sample[2].shape}, {len(sample)}")

print(type(train_dataset))

print(train_dataset[1][2].shape)

### Train
num_epochs = 1

# Loss functions
regression_loss_fn = torch.nn.SmoothL1Loss()
classification_loss_fn = torch.nn.CrossEntropyLoss()

# Training loop
print("Starting training")
# for epoch in range(num_epochs):
#     train_loop = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch')

#     for images, bbox_tuples, labels in train_loop:
#         images = images.to(device)
        
#         bboxes = tuple(bbox.to(device) for bbox in bbox_tuples)
#         labels = tuple(label.to(device) for label in labels)

#         pred_bboxes, pred_labels = model(images)

#         regression_loss = regression_loss_fn(pred_bboxes, bboxes)
#         classification_loss = classification_loss_fn(pred_labels, labels)
#         total_loss = regression_loss + classification_loss

#         optimizer.zero_grad()
#         total_loss.backward()
#         optimizer.step()

#         train_loop.set_postfix(loss=total_loss.item())