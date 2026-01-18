import torch
import torch.nn as nn
from .attention import PolarizedSelfAttention

class MLDA(nn.Module):
    def __init__(self, channels):
        super().__init__()

        # Local projection
        self.pre_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias = False),
            nn.BatchNorm2d(channels),
            nn.GELU()
        )

        # PSA (as defined earlier)
        self.psa = PolarizedSelfAttention(channels)

        # Multi-level depthwise branches (Fig. 4)
        self.branch0 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding = 1, groups = channels, bias = False),
            nn.BatchNorm2d(channels),
            nn.GELU()
        )

        self.branch1 = nn.Sequential(
            nn.Conv2d(channels, channels, (7, 1), padding = (3, 0), groups = channels, bias = False),
            nn.Conv2d(channels, channels, (1, 7), padding = (0, 3), groups = channels, bias = False),
            nn.BatchNorm2d(channels),
            nn.GELU()
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(channels, channels, (11, 1), padding = (5, 0), groups = channels, bias = False),
            nn.Conv2d(channels, channels, (1, 11), padding = (0, 5), groups = channels, bias = False),
            nn.BatchNorm2d(channels),
            nn.GELU()
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(channels, channels, (21, 1), padding = (10, 0), groups = channels, bias = False),
            nn.Conv2d(channels, channels, (1, 21), padding = (0, 10), groups = channels, bias = False),
            nn.BatchNorm2d(channels),
            nn.GELU()
        )

        # Fusion (Fig. 4)
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 4, channels, 1, bias = False),
            nn.BatchNorm2d(channels),
            nn.GELU()
        )

    def forward(self, x):
        identity = x

        x = self.pre_conv(x)
        x = self.psa(x)

        b0 = self.branch0(x)
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)

        x = torch.cat([b0, b1, b2, b3], dim = 1)
        x = self.fuse(x)

        return x + identity