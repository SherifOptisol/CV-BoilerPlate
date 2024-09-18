import torch
import torch.nn as nn
from torchvision import models

class ResNetModel(nn.Module):
    def __init__(self, num_classes=10, pretrained=True):
        super(ResNetModel, self).__init__()
        self.resnet = models.resnet18(pretrained=pretrained)  # Load pre-trained ResNet-18
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)  # Modify the final layer for our task

    def forward(self, x):
        return self.resnet(x)

# Example Usage:
if __name__ == "__main__":
    model = ResNetModel(num_classes=10, pretrained=True)
    print(model)
