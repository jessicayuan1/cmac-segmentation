import torch
from torch import nn
import torch.nn.functional as F

class HaarWaveletDownsampling(nn.Module):
    """
    Haar Wavelet Downsampling (HWD) for lossless 2x downsampling.
    
    Decomposes image into 4 subbands:
    - LL (low-low): approximation
    - LH (low-high): horizontal details
    - HL (high-low): vertical details  
    - HH (high-high): diagonal details
    
    For RGB input (3 channels), output is 12 channels (3*4).
    """
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        
        # Haar wavelet basis (not learnable)
        haar_ll = torch.tensor([[1, 1], [1, 1]], dtype = torch.float32) / 2.0
        haar_lh = torch.tensor([[-1, -1], [1, 1]], dtype = torch.float32) / 2.0
        haar_hl = torch.tensor([[-1, 1], [-1, 1]], dtype = torch.float32) / 2.0
        haar_hh = torch.tensor([[1, -1], [-1, 1]], dtype = torch.float32) / 2.0
        
        # Stack and create conv weights [out_ch, in_ch, h, w]
        haar_filters = torch.stack([haar_ll, haar_lh, haar_hl, haar_hh], dim = 0)
        haar_filters = haar_filters.unsqueeze(1)  # [4, 1, 2, 2]
        
        # Repeat for each input channel
        self.register_buffer(
            'haar_weights',
            haar_filters.repeat(in_channels, 1, 1, 1)  # [in_ch * 4, 1, 2, 2]
        )
        
    def forward(self, x):
        # Apply depthwise convolution with Haar filters
        # groups = in_channels ensures each input channel is processed separately
        out = F.conv2d(x, self.haar_weights, stride = 2, groups = self.in_channels)
        return out