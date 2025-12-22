import torch
import torch.nn as nn
from .convs import DepthwiseSeparable
from .attention import PolarizedSelfAttention

class MLDA(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        
        self.local_conv = nn.Conv2d(channels, channels, 1, bias=False)
        self.local_bn = nn.BatchNorm2d(channels)
        
        self.channel_att = PolarizedSelfAttention(channels)
        
        self.branch0 = DepthwiseSeparable(channels, channels, k=5)
        
        self.branch1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(7, 1), padding=(3, 0), groups=channels, bias=False),
            nn.Conv2d(channels, channels, kernel_size=(1, 7), padding=(0, 3), groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU()
        )
        
        self.branch2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(11, 1), padding=(5, 0), groups=channels, bias=False),
            nn.Conv2d(channels, channels, kernel_size=(1, 11), padding=(0, 5), groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU()
        )
        
        self.branch3 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(21, 1), padding=(10, 0), groups=channels, bias=False),
            nn.Conv2d(channels, channels, kernel_size=(1, 21), padding=(0, 10), groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU()
        )
        
        self.fuse = nn.Conv2d(channels * 4, channels, 1, bias=False)
        self.fuse_bn = nn.BatchNorm2d(channels)
        self.fuse_act = nn.GELU()
        
        self.final_conv = nn.Conv2d(channels, channels, 1, bias=False)
        
    def forward(self, x):
        local = self.local_conv(x)
        local = self.local_bn(local)
        
        x_att = self.channel_att(local)
        
        b0 = self.branch0(x_att)
        b1 = self.branch1(x_att)
        b2 = self.branch2(x_att)
        b3 = self.branch3(x_att)
        
        multi_scale = torch.cat([b0, b1, b2, b3], dim=1)
        fused = self.fuse(multi_scale)
        fused = self.fuse_bn(fused)
        fused = self.fuse_act(fused)
        
        out = self.final_conv(fused)
        return out
