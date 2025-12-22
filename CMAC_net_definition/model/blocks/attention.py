import torch
import torch.nn as nn

class PolarizedSelfAttention(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.channels = channels
        
        self.channel_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.GELU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
        
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.spatial_sigmoid = nn.Sigmoid()
    
    def channel_attention(self, x):
        b, c, _, _ = x.size()
        y = self.channel_pool(x).view(b, c)
        y = self.channel_fc(y).view(b, c, 1, 1)
        return x * y
    
    def spatial_attention(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.spatial_conv(y)
        y = self.spatial_sigmoid(y)
        return x * y
    
    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x
