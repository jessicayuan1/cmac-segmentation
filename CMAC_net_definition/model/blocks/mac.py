import torch
import torch.nn as nn
from .attention import PolarizedSelfAttention

class MAC(nn.Module):
    def __init__(self, n_channels, out_ch, expansion=4):
        super().__init__()
        hidden_dim = n_channels * expansion
        
        self.expand = nn.Conv2d(n_channels, hidden_dim, 1, bias=False) if n_channels != hidden_dim else nn.Identity()
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        
        self.dw_conv = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        
        self.psa = PolarizedSelfAttention(hidden_dim)
        
        self.project = nn.Conv2d(hidden_dim, out_ch, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch)
        
        self.act = nn.GELU()
        self.use_residual = (n_channels == out_ch)
    
    def forward(self, x):
        identity = x
        
        x = self.expand(x)
        x = self.bn1(x)
        x = self.act(x)
        
        x = self.dw_conv(x)
        x = self.bn2(x)
        x = self.act(x)
        
        x = self.psa(x)
        
        x = self.project(x)
        x = self.bn3(x)
        
        if self.use_residual:
            x = x + identity
            
        return self.act(x)
