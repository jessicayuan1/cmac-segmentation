import torch
from torch import nn

class WindowPartition(nn.Module):
    """
    Split feature map into non-overlapping windows of size (window_size x window_size)
    Input Shape:
        (B, C, H, W)
    Operation:
        - Divides the H and W dimensions into a grid of windows
        - Windows are flattened into shape (window_size * window_size, C).
        This is the token sequence fed into W-MSA / SW-MSA attention.
    Output Shape:
        (num_windows * B, window_size * window_size, C)
        where: num_windows = (H / window_size) * (W / window_size)
    Required for computing local self-attention inside each window.
    """
    def __init__(self, window_size):
        super().__init__()
        self.window_size = window_size
    def forward(self, x):
        B, C, H, W = x.shape
        ws = self.window_size
        # Reshape into grid of windows
        # (B, C, H/ws, ws, W/ws, ws)
        x = x.reshape(B, C, H // ws, ws, W // ws, ws)
        # Move windows to batch dimension
        # (B, H/ws, W/ws, ws, ws, C)
        x = x.permute(0, 2, 4, 3, 5, 1)
        # Flatten each window
        # (B * num_windows, ws * ws, C)
        x = x.reshape(-1, ws * ws, C)
        return x

class WindowReverse(nn.Module):
    """
    Reconstructs the feature map from windowed tokens. Inverse operation of WindowPartition.
    Input Shape:
        x_windows: (num_windows * B, ws * ws, C)
        H: original height
        W: original width
    Operation:
        - Each window of size (ws x ws) is reshaped back to (ws, ws, C)
        - Windows are placed back into their correct locations in the H x W grid
        - Final output is the full feature map
    Output shape:
        (B, C, H, W)
    """
    def __init__(self, window_size):
        super().__init__()
        self.window_size = window_size

    def forward(self, x_windows, H, W):
        # x_windows: (B * num_windows, ws * ws, C)
        ws = self.window_size
        B = x_windows.shape[0] // ((H // ws) * (W // ws))
        C = x_windows.shape[-1]
        # Reshape every window back to (ws, ws, C)
        x = x_windows.reshape(B, H // ws, W // ws, ws, ws, C)
        # Move axes to get (B, C, H, W)
        x = x.permute(0, 5, 1, 3, 2, 4)
        x = x.reshape(B, C, H, W)
        return x