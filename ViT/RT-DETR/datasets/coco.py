import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import json
from torchvision.transforms import functional as F

class CocoDetection(Dataset):
    def __init__(self, root, annotation, transforms=None):
        self.root = root
        self.transforms = transforms
        with open(annotation) as f:
            self.coco = json.load(f)

        self.ids = list(sorted(self.coco['images'], key=lambda x: x['id']))

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id['id'])
        target = coco.loadAnns(ann_ids)

        path = os.path.join(self.root, img_id['file_name'])
        img = Image.open(path).convert('RGB')
        
        boxes = []
        labels = []
        for ann in target:
            boxes.append(ann['bbox'])
            labels.append(ann['category_id'])
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.ids)

    def get_height_and_width(self, idx):
        img_info = self.coco['images'][idx]
        return img_info['height'], img_info['width']
