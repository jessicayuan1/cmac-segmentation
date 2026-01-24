import torch
import torch.nn as nn
from .mac import MAC
from .mlda import MLDA
from .attention import DropPath

class MMAC(nn.Module):
    """
    Mobile Multi-scale Attention Convolution (MMAC)
    Paper Fig 4(a): MAC -> BN -> (MLDA + Residual) -> MAC
    """
    def __init__(self, n_channels, out_ch, drop_path=0.0):
        super().__init__()

        # In CMAC-Net (Fig 2), MMAC blocks usually keep channels constant.
        # But if channel change happens, MAC1 handles it.
        
        # 1. First MAC
        self.mac1 = MAC(n_channels, out_ch, drop_path=drop_path * 0.5)
        
        # 2. BatchNorm (Explicitly shown in Fig 4a)
        self.bn = nn.BatchNorm2d(out_ch)

        # 3. MLDA Block
        self.mlda = MLDA(out_ch)
        
        # 4. Second MAC
        self.mac2 = MAC(out_ch, out_ch, drop_path=drop_path)

        # DropPath for the local MLDA residual
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        # 1. MAC 1
        x = self.mac1(x)
        
        # 2. BatchNorm
        x = self.bn(x)
        
        # 3. MLDA with Local Residual (Fig 4a)
        # The skip connection wraps ONLY the MLDA block
        mlda_out = self.mlda(x)
        x = x + self.drop_path(mlda_out)
        
        # 4. MAC 2
        x = self.mac2(x)
        
        return x