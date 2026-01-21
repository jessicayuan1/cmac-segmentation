import torch
import torch.nn as nn

class ConvBNReLU(nn.Module):
    def __init__(self, n_channels, out_ch, k = 3, stride = 1, padding = None, groups = 1):
        super().__init__()
        if padding is None:
            padding = (k - 1) // 2
        self.conv = nn.Conv2d(n_channels, out_ch, k, stride = stride, padding = padding, groups = groups, bias = False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class DepthwiseSeparable(nn.Module):
    def __init__(self, n_channels, out_ch, k = 3, stride = 1):
        super().__init__()
        pad = (k-1)//2
        self.dw = nn.Conv2d(n_channels, n_channels, kernel_size = k, stride = stride, padding = pad, groups = n_channels, bias = False)
        self.pw = nn.Conv2d(n_channels, out_ch, kernel_size = 1, bias = False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()
    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        return self.act(x)
