import json
import torch
import os
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from torchvision import transforms
from models.detr import DETR
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from models.matcher import HungarianMatcher
from models.loss import SetCriterion

class Trainer:
    def __init__(self, model, train_loader, val_loader, config):
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.optimizer = AdamW(model.parameters(), lr=self.config["lr"])
        self.scheduler = StepLR(self.optimizer, step_size=self.config["lr_drop"])
        self.matcher = HungarianMatcher()
        self.criterion = SetCriterion(num_classes=91, matcher=self.matcher,
                                      weight_dict=self.config["loss_weights"],
                                      eos_coef=self.config["eos_coef"],
                                      losses=self.config["losses"])
        self.criterion.to(self.device)

    def train_one_epoch(self):
        self.model.train()
        epoch_loss = 0
        for images, targets in self.train_loader:
            images = images.to(self.device)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            
            # Forward pass
            outputs = self.model(images)

            # Compute loss
            loss_dict = self.criterion(outputs, targets)
            weight_dict = self.criterion.weight_dict
            loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()

        self.scheduler.step()
        return epoch_loss / len(self.train_loader)

    def evaluate(self):
        self.model.eval()
        epoch_loss = 0
        with torch.no_grad():
            for images, targets in self.val_loader:
                images = images.to(self.device)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                outputs = self.model(images)
                loss_dict = self.criterion(outputs, targets)
                weight_dict = self.criterion.weight_dict
                loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

                epoch_loss += loss.item()

        return epoch_loss / len(self.val_loader)

config = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config.json')))

#with open('config.json') as config_file:
#    config = json.load(config_file)

# Define the root directory of your project dynamically
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Construct the paths to the data directories and annotation files
data_dir = os.path.join(project_root, "data")
train_data_dir = os.path.join(data_dir, "train")
val_data_dir = os.path.join(data_dir, "val")
train_ann_file = os.path.join(data_dir, "annotations", "instances_train.json")
val_ann_file = os.path.join(data_dir, "annotations", "instances_val.json")

# Data loaders
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = CocoDetection(root=train_data_dir, annFile=train_ann_file, transform=transform)
val_dataset = CocoDetection(root=val_data_dir, annFile=val_ann_file, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4)

model = DETR(num_classes=91, num_queries=config["num_queries"])
trainer = Trainer(model, train_loader, val_loader, config)

for epoch in range(config["epochs"]):
    train_loss = trainer.train_one_epoch()
    val_loss = trainer.evaluate()
    print(f"Epoch {epoch}, Train loss: {train_loss}, Val loss: {val_loss}")
    torch.save(model.state_dict(), os.path.join(project_root, f'checkpoint_{epoch}.pth'))