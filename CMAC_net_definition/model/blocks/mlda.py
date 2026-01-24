import torch
import torch.nn as nn
from .attention import ChannelPSA, DropPath

class MultiScaleSpatialAttention(nn.Module):
    """
    Implements Figure 4(b): Multi-scale Spatial Attention
    Logic: Input -> 5x5 DW -> Branches -> Sum -> 1x1 -> Sigmoid -> Mask
    """
    def __init__(self, channels):
        super().__init__()
        
        # 1. Local Spatial Context (5x5 Depthwise) - MISSED in your original code
        # Paper: "we first use 5x5 depthwise separable convolution"
        self.local_dw = nn.Sequential(
            nn.Conv2d(channels, channels, 5, padding=2, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
        )

        # 2. Multi-scale Strip Convolutions (The "Branches")
        # Note: We use pairs of (k,1) and (1,k) to simulate large kernels efficiently
        
        # Branch 1: 7x7 context
        self.branch1 = nn.Sequential(
            nn.Conv2d(channels, channels, (7, 1), padding=(3, 0), groups=channels, bias=False),
            nn.Conv2d(channels, channels, (1, 7), padding=(0, 3), groups=channels, bias=False),
            nn.BatchNorm2d(channels)
        )

        # Branch 2: 11x11 context
        self.branch2 = nn.Sequential(
            nn.Conv2d(channels, channels, (11, 1), padding=(5, 0), groups=channels, bias=False),
            nn.Conv2d(channels, channels, (1, 11), padding=(0, 5), groups=channels, bias=False),
            nn.BatchNorm2d(channels)
        )

        # Branch 3: 21x21 context
        self.branch3 = nn.Sequential(
            nn.Conv2d(channels, channels, (21, 1), padding=(10, 0), groups=channels, bias=False),
            nn.Conv2d(channels, channels, (1, 21), padding=(0, 10), groups=channels, bias=False),
            nn.BatchNorm2d(channels)
        )

        # 3. Fusion & Mask Generation
        # Eq 2: Conv1x1(Sum(branches))
        self.fusion_conv = nn.Conv2d(channels, channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 1. 5x5 Local Context
        local_feat = self.local_dw(x)

        # 2. Branches (Input is the local feature)
        # Branch 0 is just the local feature itself (Identity)
        b0 = local_feat 
        b1 = self.branch1(local_feat)
        b2 = self.branch2(local_feat)
        b3 = self.branch3(local_feat)

        # 3. Summation (Paper uses Sum, not Concat)
        summed = b0 + b1 + b2 + b3
        
        # 4. Generate Attention Mask
        mask = self.fusion_conv(summed)
        mask = self.sigmoid(mask)

        # 5. Apply Attention
        return x * mask


class MLDA(nn.Module):
    """
    Implements Figure 4(a): Multi-scale Large-kernel Dual Attention
    Structure: 1x1 -> ChannelPSA -> MultiScaleSpatialAttention -> GELU -> 1x1
    """
    def __init__(self, channels):
        super().__init__()
        
        # 1. Expansion / Pre-projection
        self.proj_1 = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        
        # 2. Channel Attention (using your corrected class)
        self.channel_att = ChannelPSA(channels)

        # 3. Multi-scale Spatial Attention (The class above)
        self.spatial_att = MultiScaleSpatialAttention(channels)
        
        # 4. Activation
        self.gelu = nn.GELU()

        # 5. Reduction / Post-projection
        self.proj_2 = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        # The MLDA block processes the feature map sequentially
        out = self.proj_1(x)
        out = self.channel_att(out)  # Refine channels first
        out = self.spatial_att(out)  # Refine spatial context
        out = self.gelu(out)
        out = self.proj_2(out)
        return out