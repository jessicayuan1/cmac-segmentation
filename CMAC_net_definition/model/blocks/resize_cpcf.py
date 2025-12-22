import torch.nn.functional as F
import torch
import torch.nn as nn
from .mlda import MLDA

class ResizeOp(nn.Module):
    """Learnable resize operation with projection to target channels"""
    def __init__(self, n_channels, out_ch, mode='up'):
        super().__init__()
        self.mode = mode
        self.conv = nn.Conv2d(n_channels, out_ch, 1, bias=False)
        
    def forward(self, x):
        x = self.conv(x)
        if self.mode == 'up':
            return F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        else:
            return F.avg_pool2d(x, kernel_size=2, stride=2)

class CPCFModule(nn.Module):
    """
    Fixed CPCF module with proper channel projection
    """
    def __init__(self, target_channels, stage, f1_ch, f2_ch, f3_ch, f4_ch):
        super().__init__()
        self.stage = stage
        self.target_channels = target_channels
        
        # Projection layers for each input feature to target_channels
        self.proj1 = nn.Conv2d(f1_ch, target_channels, 1, bias=False)
        self.proj2 = nn.Conv2d(f2_ch, target_channels, 1, bias=False)
        self.proj3 = nn.Conv2d(f3_ch, target_channels, 1, bias=False)
        self.proj4 = nn.Conv2d(f4_ch, target_channels, 1, bias=False)
        
        # Resize operations
        self.rup = ResizeOp(target_channels, target_channels, mode='up')
        self.rdown = ResizeOp(target_channels, target_channels, mode='down')
        
        # Attention block
        self.att = MLDA(target_channels)
        
    def forward(self, f1, f2, f3, f4):
        # Project all features to target channel dimension
        f1_p = self.proj1(f1)
        f2_p = self.proj2(f2)
        f3_p = self.proj3(f3)
        f4_p = self.proj4(f4)
        
        # Implement the exact fusion strategy from paper Eq.(3)
        if self.stage == 1:
            # S1 = Rup(Rup(Rup(f4) + f3) + f2) + f1
            s = self.rup(self.rup(self.rup(f4_p) + f3_p) + f2_p) + f1_p
            ref_feature = f1_p
        elif self.stage == 2:
            # S2 = Rup(Rup(f4) + f3) + Rdown(f1) + f2
            s = self.rup(self.rup(f4_p) + f3_p) + self.rdown(f1_p) + f2_p
            ref_feature = f2_p
        elif self.stage == 3:
            # S3 = Rdown(Rdown(f1) + f2) + Rup(f4) + f3
            s = self.rdown(self.rdown(f1_p) + f2_p) + self.rup(f4_p) + f3_p
            ref_feature = f3_p
        elif self.stage == 4:
            # S4 = Rdown(Rdown(Rdown(f1) + f2) + f3) + f4
            s = self.rdown(self.rdown(self.rdown(f1_p) + f2_p) + f3_p) + f4_p
            ref_feature = f4_p
        
        # Apply attention and skip connection
        m = self.att(s) + ref_feature
        return m
