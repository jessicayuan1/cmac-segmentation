import torch
from torch import nn

class PatchMerging(nn.Module):
    """
    Patch Merging Layer used in the Swin-UNet Encoder
    Input Shape:
        (B, C, H, W)
    Operation:
        - Divides the feature map into 2x2 spatial blocks
        - Each block contains 4 patches: 
              top-left, top-right, bottom-left, bottom-right
        - Concatenate the 4 patches along the channel dimension
              (C, H/2, W/2) * 4 -> (4C, H/2, W/2)
        - Applies a linear projection: 4C -> 2C
        - This performs downsampling by a factor of 2 in H and W,
          and increases feature dimension for deeper transformer layers.
    Output Shape:
        (B, 2C, H/2, W/2)
    """
    def __init__(self, dim):
        super().__init__()
        self.reduction = nn.Linear(4 * dim, 2 * dim)
    def forward(self, x):
        # Split into 4 quadrants
        x0 = x[:, :, 0::2, 0::2]  # top-left
        x1 = x[:, :, 0::2, 1::2]  # top-right
        x2 = x[:, :, 1::2, 0::2]  # bottom-left
        x3 = x[:, :, 1::2, 1::2]  # bottom-right
        # Concatenate along channel axis
        x = torch.cat([x0, x1, x2, x3], dim = 1)  # (B, 4C, H/2, W/2)
        # Prepare for linear projection
        x = x.permute(0, 2, 3, 1)  # (B, H/2, W/2, 4C)
        x = self.reduction(x)      # (B, H/2, W/2, 2C)
        x = x.permute(0, 3, 1, 2)  # (B, 2C, H/2, W/2)
        return x