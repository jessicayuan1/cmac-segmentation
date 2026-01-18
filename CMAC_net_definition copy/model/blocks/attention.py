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

        # Restore channels
        z = self.sigmoid(self.wz(z))            # (B, C, 1, 1)
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
    