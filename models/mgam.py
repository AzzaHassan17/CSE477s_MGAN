import torch
import torch.nn as nn
import torch.nn.functional as F
 


class MGAM(nn.Module):
    def __init__(self, in_channels):
        super(MGAM, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )
 
    def forward(self, x_full, x_vis):
        # x_full and x_vis: [N, C, H, W]
        """
        feat_full: [N, C, H, W]  - from full-body RoIs
        feat_vis:  [N, C, H, W]  - from visible RoIs
        """
        x = torch.cat([x_full, x_vis], dim=1)  # [N, 2C, H, W]
        attention_mask = self.conv(x)          # [N, 1, H, W]
        attended = x_full * attention_mask     # [N, C, H, W] * [N, 1, H, W] -> broadcast
        return attended, attention_mask

 