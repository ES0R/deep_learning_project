import torch
from torch.utils.data import Dataset
import os
import cv2
from torchvision import transforms
import json
from torchvision.transforms import functional as F
from PIL import Image


class COCODataset(Dataset):
    def __init__(self, root, annotation_file, dataset_name="Dataset", transform=None):
        self.dataset_name = dataset_name
        self.root = root
        self.transform = transform

        with open(annotation_file) as f:
            data = json.load(f)

            self.images = {image['id']: image for image in data['images']}
            self.image_ids = list(self.images.keys())  # List of image IDs
            self.categories = {cat['id']: cat['name'] for cat in data['categories']}

            # Group annotations by image_id
            self.annotations = {}
            for ann in data['annotations']:
                image_id = ann['image_id']
                if image_id not in self.annotations:
                    self.annotations[image_id] = []
                self.annotations[image_id].append(ann)

    def resize_image_and_bboxes(self, image, bboxes, target_size):
        # Calculate scale factors
        orig_height, orig_width = image.shape[1], image.shape[2]
        target_width, target_height = target_size
        scale_x = target_width / orig_width
        scale_y = target_height / orig_height

        # Resize image using PyTorch
        resized_image = F.resize(image, target_size)

        # Adjust bounding boxes
        resized_bboxes = []
        for bbox in bboxes:
            x, y, width, height = bbox
            x = x * scale_x
            y = y * scale_y
            width = width * scale_x
            height = height * scale_y
            resized_bboxes.append([x, y, width, height])

        return resized_image, resized_bboxes

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]  # Fetch the actual image ID
        image_info = self.images[image_id]

        # Check if image_id is in annotations
        if image_id not in self.annotations:
            print(f"Missing annotations for image_id: {image_id}")
            # Return a placeholder
            return torch.zeros(3, 224, 224), torch.zeros(0, 4), torch.zeros(0, dtype=torch.int64)


        image_anns = self.annotations[image_id]
        image_path = os.path.join(self.root, image_info['file_name'])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert image to tensor
        image = F.to_tensor(image)

        # Load annotations
        bboxes = [ann['bbox'] for ann in image_anns]
        labels = [ann['category_id'] for ann in image_anns]

        # Resize image and bounding boxes
        target_size = (224, 224)  # Example target size, modify as needed
        image, bboxes = self.resize_image_and_bboxes(image, bboxes, target_size)

        # Convert bounding boxes and labels to tensor
        bboxes = torch.tensor(bboxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        # Apply additional transformations if needed
        if self.transform:
            image = self.transform(image)

        return image, bboxes, labels

