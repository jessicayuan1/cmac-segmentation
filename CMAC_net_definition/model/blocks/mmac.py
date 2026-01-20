import torch
import torch.nn as nn
from .mac import MAC
from .mlda import MLDA

class MMAC(nn.Module):
    def __init__(self, n_channels, out_ch, drop_path = 0.0):
        super().__init__()

        dp1 = drop_path * 0.5
        dp2 = drop_path

        self.mac1 = MAC(n_channels, out_ch, drop_path = dp1)
        self.mlda = MLDA(out_ch)
        self.mac2 = MAC(out_ch, out_ch, drop_path = dp2)

        # Only needed if in != out
        self.proj = nn.Identity()
        if n_channels != out_ch:
            self.proj = nn.Conv2d(n_channels, out_ch, kernel_size = 1, bias = False)

    def forward(self, x):
        shortcut = self.proj(x)
        x = self.mac1(x)
        x = self.mlda(x)
        x = x + shortcut
        x = self.mac2(x)
        return x