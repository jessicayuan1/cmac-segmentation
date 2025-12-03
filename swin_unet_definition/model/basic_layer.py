import torch
from torch import nn

from model.blocks.swin_block import SwinTransformerBlock
from model.expand_merge.patch_merging import PatchMerging
from model.expand_merge.patch_expanding import PatchExpanding

class BasicLayer(nn.Module):
    """
    A stage of the Swin-UNet encoder or decoder
    Structure:
        - Upsampling (decoder) OR Downsampling (encoder)
        - A sequence of Swin Transformer Blocks (depth times)
    Input Shape:
        (B, C, H, W)
    Parameters:
        dim:                Embedding dimension (C)
        input_resolution:   (H, W)
        depth:              Number of SwinTransformerBlocks in this stage
        num_heads:          Number of attention heads
        window_size:        Window size for attention
        mlp_ratio:          Expansion ratio for the MLP inside each block
        downsample:         If True -> apply PatchMerging at end (encoder)
        upsample:           If True -> apply PatchExpanding at start (decoder)
    Output Shape:
        Encoder stage with downsampling:
            (B, 2C, H/2, W/2)
        Decoder stage with upsampling:
            (B, C/2, 2H, 2W)
        No sampling:
            Same shape as input
    """
    def __init__(self, dim, input_resolution, depth, num_heads,
        window_size, mlp_ratio = 4.0, downsample = False, upsample = False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.window_size = window_size

        # Upsampling (Decoder)
        if upsample:
            self.upsample = PatchExpanding(dim)
        else:
            self.upsample = None
        # Swin Transformer Blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim = dim,
                input_resolution = input_resolution,
                num_heads = num_heads,
                window_size = window_size,
                shift_size = 0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio = mlp_ratio
            )
            for i in range(depth)
        ])
        # Downsampling (Encoder)
        if downsample:
            self.downsample = PatchMerging(dim)
        else:
            self.downsample = None
    def forward(self, x):
        # Pass through Swin Transformer Blocks
        for blk in self.blocks:
            x = blk(x)
        # Upsample if Decoder
        if self.upsample:
            x = self.upsample(x)
        # Downsample if Encoder
        if self.downsample:
            x = self.downsample(x)
        return x