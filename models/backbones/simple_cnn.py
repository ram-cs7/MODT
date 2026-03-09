"""
Simple CNN Backbone
Lightweight backbone for Re-Identification (ReID) feature extraction.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    """
    Basic CNN for generating appearance embeddings.
    Input: [B, 3, 128, 64] (Typical ReID size)
    Output: [B, num_features] (Normalized)
    """
    def __init__(self, num_features=128):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1, stride=2)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1, stride=2)
        self.bn4 = nn.BatchNorm2d(256)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_features)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        # Normalize for cosine similarity
        x = F.normalize(x, p=2, dim=1)
        return x

def create_reid_backbone(name="simple_cnn", pretrained=False):
    """Factory function for ReID backbones"""
    if name == "simple_cnn":
        return SimpleCNN()
    else:
        raise NotImplementedError(f"Backbone {name} not implemented")
