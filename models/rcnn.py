import torch
import torch.nn as nn
import torch.nn.functional as F

class RCNNHead(nn.Module):
    def __init__(self, in_channels=512, num_classes=2):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))  # Ensure [N, C, 7, 7]
        self.fc1 = nn.Linear(in_channels * 7 * 7, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.cls_score = nn.Linear(1024, num_classes)  # Class scores
        self.bbox_pred = nn.Linear(1024, 8)            # Bounding box regression
 
    def forward(self, x):
        x = self.avgpool(x)  # [N, C, 7, 7] → always safe
        x = torch.flatten(x, 1)  # [N, C×7×7]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
 
        scores = self.cls_score(x)     # [N, 2]
        bbox_deltas = self.bbox_pred(x)  # [N, 4]
        bbox_full = bbox_deltas[:, :4]
        bbox_vis = bbox_deltas[:, 4:]

        return scores, bbox_full, bbox_vis