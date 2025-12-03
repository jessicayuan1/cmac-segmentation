import torch
from torch import nn

from model.encoder import Encoder
from model.decoder import Decoder


class SwinUNet(nn.Module):
    """
    Full Swin-UNet Model
    Default input size: 1024 *1024
    Default configuration (Swin-Small):
        patch_size = 4
        embed_dim  = 96
        depths     = [2, 2, 6, 2]
        num_heads  = [3, 6, 12, 24]
        window_size = 8
    """
    def __init__(self, img_size = 1024, patch_size = 4, embed_dim = 96, depths = [2, 2, 6, 2],
        num_heads = [3, 6, 12, 24], window_size = 8, mlp_ratio = 4.0, out_channels = 1, in_channels = 3):
        super().__init__()
        # Encoder
        self.encoder = Encoder(
            img_size = img_size,
            patch_size = patch_size,
            embed_dim = embed_dim,
            depths = depths,
            num_heads = num_heads,
            window_size = window_size,
            mlp_ratio = mlp_ratio,
            in_channels = in_channels
        )
        # Decoder
        self.decoder = Decoder(
            img_size = img_size,
            embed_dim = embed_dim,
            depths = depths,
            num_heads = num_heads,
            window_size = window_size,
            mlp_ratio = mlp_ratio
        )
        # Final Segmentation Head
        self.output = nn.Conv2d(embed_dim // 2, out_channels, kernel_size = 1)
    def forward(self, x):
        bottleneck, skip16, skip8, skip4 = self.encoder(x)
        x = self.decoder(bottleneck, [skip16, skip8, skip4])
        return self.output(x)
