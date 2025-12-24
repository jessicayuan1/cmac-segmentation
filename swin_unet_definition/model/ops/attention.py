import torch
from torch import nn
from .window_ops import WindowPartition, WindowReverse
    
class WindowAttention(nn.Module):
    """
    Multi-head self-attention computed inside each window (W-MSA)
    Note!!!:
        This module assumes input is already partitioned into windows.
        It does NOT call WindowPartition or WindowReverse.
        The SW-MSA Version does not expect pre-partitioned windows.
    Input Shape:
        x: (num_windows * B, ws*ws, C)
    Operation:
        - Computes Q, K, V for each token inside a window (Query, Key, Value)
        - Performs multi-head self-attention within each window only
        - Uses relative position bias as described in Swin Transformer
        - Applies softmax(QK^T / sqrt(d)) and combines with V
        - Projects result back to dimension C
    Output Shape:
        (num_windows * B, ws*ws, C)
    Core attention mechanism for Swin Transformer blocks without shifting. 
    SW-MSA wraps this with cyclic shifts.
    """
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # QKV linear layers
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        # Relative position bias table
        relative_size = (2 * window_size - 1)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(relative_size * relative_size, num_heads)
        )
        # Compute Pairwise Relative Position Indices
        coords = torch.stack(
            torch.meshgrid(
                torch.arange(window_size),
                torch.arange(window_size),
                indexing = "ij"
            )
        )  # (2, ws, ws)
        coords_flat = coords.reshape(2, -1)  # (2, ws * ws)
        rel = coords_flat[:, :, None] - coords_flat[:, None, :]  # (2, N, N)
        rel = rel.permute(1, 2, 0)  # (N, N, 2)
        rel[:, :, 0] += window_size - 1
        rel[:, :, 1] += window_size - 1
        rel_idx = rel[:, :, 0] * (2 * window_size - 1) + rel[:, :, 1]
        self.register_buffer("relative_position_index", rel_idx)
    def forward(self, x, mask = None):
        # x: (B * nW, N, C), N = ws * ws
        B_, N, C = x.shape
        # Compute Q, K, V
        qkv = self.qkv(x)               # (B * nW, N, 3C)
        q, k, v = qkv.chunk(3, dim = -1)  # each (B * n W, N, C)
        # Reshape for multi-head attention
        q = q.reshape(B_, N, self.num_heads, C // self.num_heads).transpose(1, 2)
        k = k.reshape(B_, N, self.num_heads, C // self.num_heads).transpose(1, 2)
        v = v.reshape(B_, N, self.num_heads, C // self.num_heads).transpose(1, 2)
        # Attention Score
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B * nW, heads, N, N)
        # Add relative position bias
        bias = self.relative_position_bias_table[
            self.relative_position_index.reshape(-1)
        ].reshape(N, N, self.num_heads)  # (N, N, heads)
        bias = bias.permute(2, 0, 1)  # (heads, N, N)
        attn = attn + bias.unsqueeze(0)
        if mask is not None:
            attn = attn + mask.unsqueeze(1)  # (1, heads, N, N) safe broadcast
        # Softmax
        attn = attn.softmax(dim = -1)
        # Attention Output
        out = attn @ v  # (B * nW, heads, N, head_dim)
        out = out.transpose(1, 2).reshape(B_, N, C)
        # Final Projection
        out = self.proj(out)
        return out

class ShiftedWindowAttention(nn.Module):
    """
    Shifted multi-head self-attention (SW-MSA) used in Swin Transformer blocks.
    Note!!!:
        SW-MSA handles window partitioning and reversing internally.
        It uses WindowPartition to extract windows and WindowReverse to
        reconstruct the feature map after attention.
    Input Shape:
        x: (B, C, H, W)
    Operation:
        - Cyclically shifts the feature map by (-shift, -shift)
        - Partitions the shifted map into windows
        - Applies standard WindowAttention inside each window
        - Applies an attention mask so windows do not attend across boundaries
        - Reverses the windows back to (B, C, H, W)
        - Cyclically shifts back to the original spatial alignment
    Output Shape:
        (B, C, H, W)
    SW-MSA enables cross-window information propagation between layers
    """
    def __init__(self, dim, window_size, num_heads, shift_size):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size
        self.num_heads = num_heads
        # Window Attention
        self.attn = WindowAttention(dim, window_size, num_heads)
        self.register_buffer("attn_mask", None)
    def create_mask(self, H, W, device):
        ws = self.window_size
        ss = self.shift_size
        img_mask = torch.zeros((1, 1, H, W), device = device)  # (1, 1, H, W)
        h_slices = (
            slice(0, -ws),
            slice(-ws, -ss),
            slice(-ss, None)
        )
        w_slices = (
            slice(0, -ws),
            slice(-ws, -ss),
            slice(-ss, None)
        )
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, :, h, w] = cnt
                cnt += 1
        # Partition mask exactly the same way as the image
        mask_windows = WindowPartition(ws)(img_mask)    # (nW, ws * ws, 1)
        mask_windows = mask_windows.reshape(-1, ws*ws)
        # Compute Attention Mask
        mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        mask = mask.masked_fill(mask != 0, float("-inf"))
        mask = mask.masked_fill(mask == 0, 0.0)
        # Return only first mask - all windows have the same pattern
        return mask[0:1]  # (1, ws * ws, ws * ws)

    def forward(self, x):
        B, C, H, W = x.shape
        ws = self.window_size
        ss = self.shift_size
        
        # Cyclic Shift
        if ss > 0:
            shifted_x = torch.roll(x, shifts = (-ss, -ss), dims = (2, 3))
        else:
            shifted_x = x
        
        # Build mask if needed
        num_windows = (H // ws) * (W // ws)
        if (self.attn_mask is None) or (self.attn_mask.shape[1] != ws * ws):
            self.attn_mask = self.create_mask(H, W, x.device)
        
        # Partition into Windows
        x_windows = WindowPartition(ws)(shifted_x)  # (B * nW, ws * ws, C)
        # Apply window attention with mask
        attn_windows = self.attn(x_windows, mask = self.attn_mask)
        # Reverse Windows -> Full Feature Map
        shifted_out = WindowReverse(ws)(attn_windows, H, W)
        # Undo Cyclic Shift
        if ss > 0:
            out = torch.roll(shifted_out, shifts = (ss, ss), dims = (2, 3))
        else:
            out = shifted_out
        return out