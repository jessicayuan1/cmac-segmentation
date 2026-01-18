import torch.nn.functional as F
import torch
import torch.nn as nn
from .mlda import MLDA

class ResizeOp(nn.Module):
    """
    Learnable resize operation with projection to target channels
    """
    def __init__(self, n_channels, out_ch, mode = 'up'):
        super().__init__()
        self.mode = mode
        self.conv = nn.Conv2d(n_channels, out_ch, 1, bias = False)
        
    def forward(self, x):
        x = self.conv(x)
        if self.mode == 'up':
            return F.interpolate(x, scale_factor = 2, 
                                 mode = 'bilinear', 
                                 align_corners = False)
        else:
            return F.max_pool2d(x, kernel_size = 2, stride = 2)

class CPCFModule(nn.Module):
    """
    Fixed CPCF module with proper channel projection
    """
    def __init__(self, target_channels, stage, f1_ch, f2_ch, f3_ch, f4_ch):
        super().__init__()
        self.stage = stage

        # Resize operators (each includes 1×1 projection)
        self.rup1 = ResizeOp(f4_ch, target_channels, mode = 'up')
        self.rup2 = ResizeOp(target_channels, target_channels, mode = 'up')
        self.rup3 = ResizeOp(target_channels, target_channels, mode = 'up')

        self.rdown1 = ResizeOp(f1_ch, target_channels, mode = 'down')
        self.rdown2 = ResizeOp(target_channels, target_channels, mode = 'down')
        self.rdown3 = ResizeOp(target_channels, target_channels, mode = 'down')

        # Direct projections (no resize)
        self.proj1 = nn.Conv2d(f1_ch, target_channels, 1, bias = False)
        self.proj2 = nn.Conv2d(f2_ch, target_channels, 1, bias = False)
        self.proj3 = nn.Conv2d(f3_ch, target_channels, 1, bias = False)
        self.proj4 = nn.Conv2d(f4_ch, target_channels, 1, bias = False)

        # Attention
        self.att = MLDA(target_channels)

    def forward(self, f1, f2, f3, f4):
        if self.stage == 1:
            # S1 = Rup(Rup(Rup(f4) + f3) + f2) + f1
            s = self.rup3(
                    self.rup2(
                        self.rup1(f4) + self.proj3(f3)
                    ) + self.proj2(f2)
                ) + self.proj1(f1)
            ref = self.proj1(f1)

        elif self.stage == 2:
            # S2 = Rup(Rup(f4) + f3) + Rdown(f1) + f2
            s = (
                self.rup2(self.rup1(f4) + self.proj3(f3))
                + self.rdown1(f1)
                + self.proj2(f2)
            )
            ref = self.proj2(f2)

        elif self.stage == 3:
            # S3 = Rdown(Rdown(f1) + f2) + Rup(f4) + f3
            s = (
                self.rdown2(self.rdown1(f1) + self.proj2(f2))
                + self.rup1(f4)
                + self.proj3(f3)
            )
            ref = self.proj3(f3)

        else:  # stage == 4
            # S4 = Rdown(Rdown(Rdown(f1) + f2) + f3) + f4
            s = (
                self.rdown3(
                    self.rdown2(self.rdown1(f1) + self.proj2(f2))
                    + self.proj3(f3)
                ) + self.proj4(f4)
            )
            ref = self.proj4(f4)

        return self.att(s) + ref
