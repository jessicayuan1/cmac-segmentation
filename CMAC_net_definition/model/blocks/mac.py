import torch
import torch.nn as nn
from .attention import PolarizedSelfAttention


class DropPath(nn.Module):
    """Stochastic Depth (Drop Path) for regularization"""
    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, device=x.device)
        binary_mask = random_tensor.floor()
        return x / keep_prob * binary_mask


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
    def __init__(self, n_channels, out_ch, expansion=1, drop_path=0.0):
        super().__init__()
        
        # CRITICAL FIX: Use expansion=1 to match lightweight design
        # MobileNet typically uses 1-2, NOT 4!
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
        x = self.drop_path(x)
        if self.use_residual:
            x = x + identity

        return x