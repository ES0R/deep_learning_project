import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from models.detr import DETR
from models.backbone import Backbone
from models.transformer import Transformer
from models.matcher import HungarianMatcher
from models.loss import SetCriterion
from datasets.coco import CocoDetection
from models.utils import collate_fn
from training.trainer import Trainer
import json

# Load configurations
with open('config.json') as config_file:
    config = json.load(config_file)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model configuration
backbone = Backbone()
transformer = Transformer(d_model=config['hidden_dim'], nhead=config['nheads'], num_encoder_layers=config['enc_layers'],
                          num_decoder_layers=config['dec_layers'])
model = DETR(backbone, transformer, num_classes=config['num_classes'], num_queries=config['num_queries'])
model = model.to(device)

# Datasets and Dataloader
train_dataset = CocoDetection(root=config['train_images'], annotation=config['train_annotations'], transforms=...)
valid_dataset = CocoDetection(root=config['val_images'], annotation=config['val_annotations'], transforms=...)

train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=collate_fn)
valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=collate_fn)

# Optimizer and Scheduler
optimizer = optim.AdamW(model.parameters(), lr=config['lr'])
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config['lr_drop'])

# Loss and Matcher
matcher = HungarianMatcher(cost_class=config['cost_class'], cost_bbox=config['cost_bbox'], cost_giou=config['cost_giou'])
weight_dict = {'loss_ce': config['ce_loss_weight'], 'loss_bbox': config['bbox_loss_weight']}
losses = SetCriterion(num_classes=config['num_classes'], matcher=matcher, weight_dict=weight_dict, eos_coef=config['eos_coef'], losses=['labels', 'boxes', 'cardinality'])

# Training Loop
trainer = Trainer(model, optimizer, loss_function=losses, device=device, lr_scheduler=lr_scheduler, train_loader=train_loader, val_loader=valid_loader)

for epoch in range(config['epochs']):
    trainer.train_one_epoch()
    trainer.evaluate()
    lr_scheduler.step()

# Save the model
torch.save(model.state_dict(), 'detr_model.pth')
