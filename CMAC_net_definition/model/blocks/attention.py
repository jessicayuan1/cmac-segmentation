import torch
import torch.nn as nn

class ChannelPSA(nn.Module):
    def __init__(self, channels):
        super().__init__()
        mid = channels // 2
        self.wq = nn.Conv2d(channels, mid, 1)
        self.wv = nn.Conv2d(channels, mid, 1)
        self.wz = nn.Conv2d(mid, channels, 1)
        self.softmax = nn.Softmax(dim = 2)      # over HW
        self.ln = nn.LayerNorm(channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.shape
        hw = h * w

        # Value: (B, C/2, HW)
        v = self.wv(x).view(b, -1, hw)

        # Query: (B, C/2, HW) -> spatial weights
        q = self.wq(x).view(b, -1, hw)
        q = self.softmax(q)                     # FSM
        q = q.mean(dim=1, keepdim=True)         # collapse channels -> (B, 1, HW)

        # Polarization: (B, C/2, HW) × (B, HW, 1)
        z = torch.matmul(v, q.transpose(1, 2))  # (B, C/2, 1)
        z = z.view(b, -1, 1, 1)

        # Projection C/2 -> C
        z = self.wz(z)
        
        # FIX 1: Apply LayerNorm
        # LayerNorm expects (B, ..., C), so we permute
        z = z.permute(0, 2, 3, 1) # (B, 1, 1, C)
        z = self.ln(z)
        z = z.permute(0, 3, 1, 2) # (B, C, 1, 1)

        # Restore channels
        z = self.sigmoid(z)            # (B, C, 1, 1)
        return x * z
    
class SpatialPSA(nn.Module):
    def __init__(self, channels):
        super().__init__()
        mid = channels // 2
        self.wq = nn.Conv2d(channels, mid, 1)
        self.wv = nn.Conv2d(channels, mid, 1)
        self.softmax = nn.Softmax(dim = 1)      # channel softmax
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.shape

        # Query branch
        q = self.wq(x)                              # (B, C/2, H, W)
        q = q.mean(dim = (2, 3), keepdim = True)    # FGP -> (B, C/2, 1, 1)
        q = q.view(b, 1, -1)                        # σ1 -> (B, 1, C/2)
        q = self.softmax(q)                         # channel softmax

        # Value branch
        v = self.wv(x).view(b, -1, h * w)           # σ2 → (B, C/2, HW)

        # Polarization
        attn = torch.matmul(q, v)                   # (B, 1, HW)
        attn = attn.view(b, 1, h, w)                # σ3
        attn = self.sigmoid(attn)                   # FSG
        return x * attn
    
class PolarizedSelfAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channel_psa = ChannelPSA(channels)
        self.spatial_psa = SpatialPSA(channels)

    def forward(self, x):
        # Apply channel attention, then refine with spatial attention
        z_ch = self.channel_psa(x)
        z_sp = self.spatial_psa(z_ch)
        return z_sp
    
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
