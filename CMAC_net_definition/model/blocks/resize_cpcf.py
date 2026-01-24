import torch
import torch.nn as nn
import torch.nn.functional as F
from .mlda import MLDA


class ResizeOp(nn.Module):
    """
    Resize operation with channel projection
    Per paper Eq. 4-5:
        x' = Rup(x) = Up(Conv1×1(x))
        y' = Rdown(y) = Down(Conv1×1(y))
    """
    def __init__(self, in_channels, out_channels, mode='up'):
        super().__init__()
        self.mode = mode
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=True)
        
    def forward(self, x):
        x = self.conv(x)
        if self.mode == 'up':
            return F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        else:
            return F.max_pool2d(x, kernel_size=2, stride=2)


class CPCFModule(nn.Module):
    """
    Cascade Progressive Context Fusion Module
    
    Per paper Equation 3:
        S1 = Rup(Rup(Rup(f4) + f3) + f2) + f1
        S2 = Rup(Rup(f4) + f3) + Rdown(f1) + f2
        S3 = Rdown(Rdown(f1) + f2) + Rup(f4) + f3
        S4 = Rdown(Rdown(Rdown(f1) + f2) + f3) + f4
    
    Per paper Equation 6:
        m_i = Att(S_i) + f_i
    
    Args:
        target_channels: Output channels (should equal f{stage}_ch)
        stage: Which stage (1, 2, 3, or 4)
        f1_ch, f2_ch, f3_ch, f4_ch: Input feature channels
    """
    def __init__(self, target_channels, stage, f1_ch, f2_ch, f3_ch, f4_ch):
        super().__init__()
        self.stage = stage
        
        # Create only the resize ops we actually need for this stage
        # Each ResizeOp includes the Conv1x1 projection
        
        if stage == 1:
            # Need: f4->f3, f3->f2, f2->f1
            self.rup1 = ResizeOp(f4_ch, f3_ch, mode='up')  # f4 -> f3 scale
            self.rup2 = ResizeOp(f3_ch, f2_ch, mode='up')  # f3 -> f2 scale  
            self.rup3 = ResizeOp(f2_ch, f1_ch, mode='up')  # f2 -> f1 scale
            
        elif stage == 2:
            # Need: f4->f3, f3->f2, f1->f2
            self.rup1 = ResizeOp(f4_ch, f3_ch, mode='up')      # f4 -> f3
            self.rup2 = ResizeOp(f3_ch, f2_ch, mode='up')      # f3 -> f2
            self.rdown1 = ResizeOp(f1_ch, f2_ch, mode='down')  # f1 -> f2
            
        elif stage == 3:
            # Need: f1->f2, f2->f3, f4->f3
            self.rdown1 = ResizeOp(f1_ch, f2_ch, mode='down')  # f1 -> f2
            self.rdown2 = ResizeOp(f2_ch, f3_ch, mode='down')  # f2 -> f3
            self.rup1 = ResizeOp(f4_ch, f3_ch, mode='up')      # f4 -> f3
            
        else:  # stage == 4
            # Need: f1->f2, f2->f3, f3->f4
            self.rdown1 = ResizeOp(f1_ch, f2_ch, mode='down')  # f1 -> f2
            self.rdown2 = ResizeOp(f2_ch, f3_ch, mode='down')  # f2 -> f3
            self.rdown3 = ResizeOp(f3_ch, f4_ch, mode='down')  # f3 -> f4
        
        # Attention block operates at target resolution
        self.att = MLDA(target_channels)
        
    def forward(self, f1, f2, f3, f4):
        """
        Fuse multi-scale features according to paper Eq. 3
        """
        if self.stage == 1:
            # S1 = Rup(Rup(Rup(f4) + f3) + f2) + f1
            s = self.rup1(f4) + f3           # f3 scale
            s = self.rup2(s) + f2            # f2 scale
            s = self.rup3(s) + f1            # f1 scale
            ref = f1
            
        elif self.stage == 2:
            # S2 = Rup(Rup(f4) + f3) + Rdown(f1) + f2
            s = self.rup1(f4) + f3           # f3 scale
            s = self.rup2(s)                 # f2 scale
            s = s + self.rdown1(f1) + f2     # f2 scale
            ref = f2
            
        elif self.stage == 3:
            # S3 = Rdown(Rdown(f1) + f2) + Rup(f4) + f3
            s = self.rdown1(f1) + f2         # f2 scale
            s = self.rdown2(s)               # f3 scale
            s = s + self.rup1(f4) + f3       # f3 scale
            ref = f3
            
        else:  # stage == 4
            # S4 = Rdown(Rdown(Rdown(f1) + f2) + f3) + f4
            s = self.rdown1(f1) + f2         # f2 scale
            s = self.rdown2(s) + f3          # f3 scale
            s = self.rdown3(s) + f4          # f4 scale
            ref = f4
        
        # m_i = Att(S_i) + f_i (Equation 6)
        return self.att(s) + ref