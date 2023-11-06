import torch.nn as nn
from torchvision.models.resnet import resnet50

class Backbone(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        # Load a pre-trained ResNet-50 model
        self.resnet = resnet50(pretrained=pretrained)
        del self.resnet.fc  # Remove the fully connected layer

        # Use the outputs from the last three stages
        self.out_channels = [self.resnet.layer2[-1].bn3.num_features,
                             self.resnet.layer3[-1].bn3.num_features,
                             self.resnet.layer4[-1].bn3.num_features]

    def forward(self, x):
        # Forward through the ResNet layers
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x1 = self.resnet.layer1(x)
        x2 = self.resnet.layer2(x1)
        x3 = self.resnet.layer3(x2)
        x4 = self.resnet.layer4(x3)

        # Return the outputs from the last three stages
        return [x2, x3, x4]

# You can initialize the backbone like this:
# backbone = Backbone(pretrained=True)
