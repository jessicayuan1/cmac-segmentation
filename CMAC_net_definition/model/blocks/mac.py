import torch
import torch.nn as nn
from .attention import PolarizedSelfAttention, DropPath

class MAC(nn.Module):
    """
    Mobile Attention Convolution
    
    Based on MobileNet's inverted residual block with PSA attention.
    Paper Fig. 3: 1×1 expand -> 3×3 DW -> PSA -> 1×1 project
    
    Args:
        n_channels: Input channels
        out_ch: Output channels
        expansion: Channel expansion ratio (use 1 for lightweight, matching MobileNetV2-style)
        drop_path: DropPath probability
    """
    def __init__(self, n_channels, out_ch, expansion=4, drop_path=0.0):
        super().__init__()
        
        hidden_dim = n_channels * expansion

        # 1×1 expand
        self.expand = nn.Conv2d(n_channels, hidden_dim, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_dim)

        # 3×3 depthwise
        self.dw_conv = nn.Conv2d(
            hidden_dim, hidden_dim, 3,
            padding=1, groups=hidden_dim, bias=False
        )
        self.bn2 = nn.BatchNorm2d(hidden_dim)

        # Polarized Self-Attention
        self.psa = PolarizedSelfAttention(hidden_dim)

        # 1×1 project
        self.project = nn.Conv2d(hidden_dim, out_ch, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch)

        self.act = nn.GELU()
        self.drop_path = DropPath(drop_path)
        self.use_residual = (n_channels == out_ch)

    def forward(self, x):
        identity = x

        # Expand
        x = self.act(self.bn1(self.expand(x)))
        
        # Depthwise + PSA
        x = self.act(self.bn2(self.dw_conv(x)))
        x = self.psa(x)
        
        # Project
        x = self.bn3(self.project(x))

        # DropPath + Residual
        if self.use_residual:
            # ONLY drop path if we have an identity shortcut to fall back on
            x = identity + self.drop_path(x)
            x = self.act(x)

        return x