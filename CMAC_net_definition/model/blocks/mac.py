import torch
import torch.nn as nn
from .attention import PolarizedSelfAttention

class DropPath(nn.Module):
    def __init__(self, drop_prob = 0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, device = x.device)
        binary_mask = random_tensor.floor()
        return x / keep_prob * binary_mask
    
class MAC(nn.Module):
    def __init__(self, n_channels, out_ch, expansion = 4, drop_path = 0.0):
        super().__init__()
        hidden_dim = n_channels * expansion

        self.expand = nn.Conv2d(n_channels, hidden_dim, 1, bias = False)
        self.bn1 = nn.BatchNorm2d(hidden_dim)

        self.dw_conv = nn.Conv2d(
            hidden_dim, hidden_dim, 3,
            padding = 1, groups = hidden_dim, bias = False
        )
        self.bn2 = nn.BatchNorm2d(hidden_dim)

        self.psa = PolarizedSelfAttention(hidden_dim)

        self.project = nn.Conv2d(hidden_dim, out_ch, 1, bias = False)
        self.bn3 = nn.BatchNorm2d(out_ch)

        self.act = nn.GELU()
        self.drop_path = DropPath(drop_path)
        self.use_residual = (n_channels == out_ch)

    def forward(self, x):
        identity = x

        x = self.act(self.bn1(self.expand(x)))
        x = self.act(self.bn2(self.dw_conv(x)))
        x = self.psa(x)
        x = self.bn3(self.project(x))

        x = self.drop_path(x)
        if self.use_residual:
            x = x + identity

        return x

