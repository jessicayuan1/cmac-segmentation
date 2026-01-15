import torch
import torch.nn as nn
from .mac import MAC
from .mlda import MLDA

class MMAC(nn.Module):
    def __init__(self, n_channels, out_ch, drop_path=0.0):
        super().__init__()

        # Split drop path across internal MAC blocks
        dp1 = drop_path * 0.5
        dp2 = drop_path

        self.mac1 = MAC(n_channels, out_ch, drop_path=dp1)
        self.mac2 = MAC(out_ch, out_ch, drop_path=dp2)
        self.mlda = MLDA(out_ch)

    def forward(self, x):
        x = self.mac1(x)
        x = self.mac2(x)
        x = self.mlda(x)
        return x