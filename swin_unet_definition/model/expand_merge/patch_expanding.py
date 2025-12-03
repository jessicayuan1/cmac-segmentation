import torch
from torch import nn

class PatchExpanding(nn.Module):
    """
    Patch Expanding Layer used in the Swin-UNet Decoder
    Input Shape:
        (B, C, H, W)
    Operation:
        - Applies a linear layer to expand channel dimension: C → 2C
            (prepares tensor for spatial rearrangement)
        - Rearranges channels into spatial dimension to upsample:
            (B, 2C, H, W) -> (B, C/2, 2H, 2W)
        - Doubles the height and width while reducing channels.
    Output Shape:
        (B, C/2, 2H, 2W)
    This is the decoder upsampling step (2x Resolution Increase)
    """
    def __init__(self, dim):
        super().__init__()
        self.expand = nn.Linear(dim, 2 * dim, bias = False)
    def forward(self, x):
        B, C, H, W = x.shape
        # Flatten Spacial Dim
        x = x.permute(0, 2, 3, 1)   # (B, H, W, C)
        # Linear expansion: C → 2C
        x = self.expand(x)          # (B, H, W, 2C)
        # Reshape into spatial upsample
        x = x.reshape(B, H, W, 2, 2, C // 2)  # (B, H, W, 2, 2, C//2)
        # Move upsample blocks into spatial dims
        x = x.permute(0, 3, 4, 1, 2, 5)       # (B, 2, 2, H, W, C//2)
        x = x.reshape(B, C // 2, H * 2, W * 2)
        return x