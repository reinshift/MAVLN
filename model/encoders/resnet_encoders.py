import torch.nn as nn
import torchvision.models as models
"""
to extract features from images
use pretrained resnet50
"""

class ResnetEncoder(nn.Module):
    def __init__(self, output_size=512):
        super(ResnetEncoder, self).__init__()
        self.feature_dim = 2048
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        # remove the last layer of resnet
        self.resnet_features = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(self.feature_dim, output_size)

    def forward(self, x):
        """
        x: (B, C, H, W)
        return: (B, out_size)
        """
        batch_size = x.size(0)
        features = self.resnet_features(x) # (B, feature_dim, 1, 1)
        features = features.view(batch_size, -1) # (B, feature_dim)
        output = self.fc(features) # (B, out_size)
        return output