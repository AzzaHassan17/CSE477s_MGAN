import torch
import torch.nn as nn
from torchvision.models import vgg16
 
class VGGBackbone(nn.Module):
    def __init__(self, pretrained=True):
        super(VGGBackbone, self).__init__()
        vgg = vgg16(pretrained=pretrained)
        # Extract features up to conv5_3 (final conv block)
        self.features = nn.Sequential(*list(vgg.features.children())[:30])  # conv5_3
        # Optional: freeze earlier layers
        for layer in self.features[:10]:  # freeze first two blocks (optional)
            for param in layer.parameters():
                param.requires_grad = False
 
    def forward(self, x):
        return self.features(x)  # [B, 512, H/16, W/16]