import torch.nn as nn
import torchvision.models as models

class ResNetSteeringModel(nn.Module):
    def __init__(self):
        super(ResNetSteeringModel, self).__init__()
        # Load a pretrained ResNet and modify the final layer
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)  # or use 'DEFAULT' for latest weights
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 1)  # Assuming output is a single scalar (steering angle)

    def forward(self, x):
        # Ensure input tensor is on the same device as the model
        return self.resnet(x.to(self.resnet.fc.weight.device))
