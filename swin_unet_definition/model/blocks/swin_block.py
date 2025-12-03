import torch
from torch import nn
from model.ops.attention import ShiftedWindowAttention, WindowAttention
from model.ops.window_ops import WindowReverse, WindowPartition

class SwinTransformerBlock(nn.Module):
    """
    Core Swin Transformer Block
    Input Shape:
        (B, C, H, W)
    Operations in Order:
        - LayerNorm
        - Window-based self-attention (W-MSA) OR Shifted version (SW-MSA)
        - Residual connection
        - LayerNorm
        - MLP (2-layer feedforward network with GELU activation)
        - Residual connection
    Alternation:
        - If shift_size = 0 → W-MSA (no shift)
        - If shift_size > 0 → SW-MSA (uses cyclic shift + masking)
    Output Shape:
        (B, C, H, W)
    Note:
        This block does NOT perform patch merging or expanding.
        It only applies transformer-style processing to the current resolution.
    """
    def __init__(self, dim, input_resolution, num_heads, window_size = 7, shift_size = 0, mlp_ratio = 4.0):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.shift_size = shift_size
        self.num_heads = num_heads
        # First LayerNorm
        self.norm1 = nn.LayerNorm(dim)
        # Attention: Shifted or Not
        if shift_size > 0:
            self.attn = ShiftedWindowAttention(
                dim, window_size, num_heads, shift_size
            )
        else:
            # Wrap WindowAttention into a simple class that just calls it
            self.attn = self._build_nonshift_attn(dim, window_size, num_heads)
        # Second LayerNorm
        self.norm2 = nn.LayerNorm(dim)
        # MLP (feedforward network)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    def _build_nonshift_attn(self, dim, window_size, num_heads):
        """
        Wrap WindowAttention in a class that expects (B, C, H, W)
        and handles WindowPartition + WindowReverse internally.
        """
        class NonShiftAttention(nn.Module):
            def __init__(self):
                super().__init__()
                self.window_attn = WindowAttention(dim, window_size, num_heads)
                self.window_size = window_size
            def forward(self, x):
                B, C, H, W = x.shape
                ws = self.window_size
                # Partition windows
                x_windows = WindowPartition(ws)(x)
                # Attention
                attn_windows = self.window_attn(x_windows)
                # Reverse windows
                out = WindowReverse(ws)(attn_windows, H, W)
                return out
        return NonShiftAttention()
    def forward(self, x):
        B, C, H, W = x.shape
        # Flatten channels for LayerNorm: (B, H * W, C)
        shortcut = x
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        # Norm + Reshape back to (B, C, H, W)
        x = self.norm1(x).reshape(B, H, W, C).permute(0, 3, 1, 2)
        # Attention (shifted or unshifted)
        x = self.attn(x)
        # First Residual
        x = shortcut + x
        # Second LayerNorm + MLP
        shortcut2 = x
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        x = self.norm2(x)
        x = self.mlp(x)
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
        # Second residual
        x = shortcut2 + x
        return x