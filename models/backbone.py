import torch
import torch.nn as nn
import torchvision.models as models

class Backbone(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.body = nn.Sequential(*list(resnet.children())[:-2])
        self.conv = nn.Conv2d(2048, hidden_dim, kernel_size=1)

    def forward(self, x):
        x = self.body(x)
        x = self.conv(x)
        return x
