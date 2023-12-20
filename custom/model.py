import torch
import torchvision.models as models
from torchvision.models.resnet import ResNet50_Weights
import torch.nn as nn


class ObjectDetectionModel(nn.Module):
    def __init__(self, num_classes):
        super(ObjectDetectionModel, self).__init__()
        # Load a pre-trained ResNet
        backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

        self.base_layers = nn.Sequential(*list(backbone.children())[:-2])

        # Self-attention layer (can be a Transformer encoder layer)
        self.attention = nn.TransformerEncoderLayer(d_model=2048, nhead=8)

        # Calculate feature map size
        input_size = 224  # Assuming square input images
        downscaling_factor = 32  # Downscaling in ResNet50
        feature_map_size = input_size // downscaling_factor

        # Additional layer to reduce the dimensionality
        self.intermediate_layer = nn.Linear(2048 * feature_map_size * feature_map_size, 2048)

        # Final detection layers
        self.fc_bbox = nn.Linear(2048, 4)  # 4 for bbox coordinates
        self.fc_class = nn.Linear(2048, num_classes)

    def forward(self, x):
        # Feature extraction
        features = self.base_layers(x)

        # Reshape features for the self-attention layer
        batch_size, channels, height, width = features.size()
        features = features.view(batch_size, -1, channels)  # Reshape to [batch_size, seq_len, features]

        attn_output = self.attention(features)

        # Flatten the output for the fully connected layers
        flattened = attn_output.view(batch_size, -1)

        # Reduce dimensionality
        intermediate_output = self.intermediate_layer(flattened)

        # Predict bounding boxes and class labels
        bboxes = self.fc_bbox(intermediate_output)
        classes = self.fc_class(intermediate_output)

        return bboxes, classes
