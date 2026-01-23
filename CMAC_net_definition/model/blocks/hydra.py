import torch
import torch.nn as nn

class HydraSegHead(nn.Module):
    def __init__(self, in_ch, num_classes, hidden_ratio = 0.25):
        super().__init__()
        hidden = max(1, int(in_ch * hidden_ratio))

        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, hidden, kernel_size = 1, bias = True),
                nn.GELU(),
                nn.Conv2d(hidden, 1, kernel_size = 1, bias = True),
            )
            for _ in range(num_classes)
        ])

    def forward(self, x):
        logits = [head(x) for head in self.heads]
        return torch.cat(logits, dim = 1)