import torch
import torch.nn as nn
from .mac import MAC
from .mlda import MLDA

class MMAC(nn.Module):
    def __init__(self, n_channels, out_ch):
        super().__init__()
        self.mac1 = MAC(n_channels, out_ch)
        self.mac2 = MAC(out_ch, out_ch)
        self.mlda = MLDA(out_ch)
        
    def forward(self, x):
        x = self.mac1(x)
        x = self.mac2(x)
        x = self.mlda(x)
        return x
