import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import (
    ConvBNReLU, DepthwiseSeparable,
    PolarizedSelfAttention, MAC, MLDA, MMAC,
    ResizeOp, CPCFModule
)

class CMACNet(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=5,
        embed_dim=96,
        depths=None,
        img_size=1024,
    ):
        super().__init__()

        # Default depths if not provided
        if depths is None:
            depths = [2, 2, 6, 2]

        # Derive base_ch from embed_dim for scaling
        base_ch = embed_dim // 3  # e.g., 96 -> 32

        # Store configuration
        self.img_size = img_size
        self.embed_dim = embed_dim
        self.depths = depths

        # -------------------------
        # Encoder (MDAC Backbone)
        # -------------------------
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_ch, 3, stride=2, padding=1),
            nn.Conv2d(base_ch, base_ch, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_ch),
            nn.GELU(),
        )

        # Stage 1: depths[0] MAC blocks, all base_ch -> base_ch
        self.stage1 = nn.Sequential(*[MAC(base_ch, base_ch) for _ in range(depths[0])])
        self.pool1 = nn.MaxPool2d(2)

        # Stage 2: first MAC base_ch -> base_ch*2, then depths[1]-1 MAC base_ch*2 -> base_ch*2
        stage2_blocks = [MAC(base_ch, base_ch * 2)]
        if depths[1] > 1:
            stage2_blocks.extend([MAC(base_ch * 2, base_ch * 2) for _ in range(depths[1] - 1)])
        self.stage2 = nn.Sequential(*stage2_blocks)
        self.pool2 = nn.MaxPool2d(2)

        # Stage 3: first MMAC base_ch*2 -> base_ch*4, then depths[2]-1 MMAC base_ch*4 -> base_ch*4
        stage3_blocks = [MMAC(base_ch * 2, base_ch * 4)]
        if depths[2] > 1:
            stage3_blocks.extend([MMAC(base_ch * 4, base_ch * 4) for _ in range(depths[2] - 1)])
        self.stage3 = nn.Sequential(*stage3_blocks)
        self.pool3 = nn.MaxPool2d(2)

        # Stage 4: first MMAC base_ch*4 -> base_ch*8, then depths[3]-1 MMAC base_ch*8 -> base_ch*8
        stage4_blocks = [MMAC(base_ch * 4, base_ch * 8)]
        if depths[3] > 1:
            stage4_blocks.extend([MMAC(base_ch * 8, base_ch * 8) for _ in range(depths[3] - 1)])
        self.stage4 = nn.Sequential(*stage4_blocks)
        self.pool4 = nn.MaxPool2d(2)

        # Stage 5 (bottleneck): fixed 3 MMAC blocks for simplicity, base_ch*8 -> base_ch*16, then base_ch*16 -> base_ch*16
        self.stage5 = nn.Sequential(
            MMAC(base_ch * 8, base_ch * 16),
            MMAC(base_ch * 16, base_ch * 16),
            MMAC(base_ch * 16, base_ch * 16),
        )

        # -------------------------
        # CPCF Modules with proper channel handling
        # -------------------------
        # Define the actual channel dimensions for each stage
        f1_ch = base_ch  # Stage 1 output channels
        f2_ch = base_ch * 2  # Stage 2 output channels
        f3_ch = base_ch * 4  # Stage 3 output channels
        f4_ch = base_ch * 8  # Stage 4 output channels

        self.cpcf1 = CPCFModule(f1_ch, stage=1, f1_ch=f1_ch, f2_ch=f2_ch, f3_ch=f3_ch, f4_ch=f4_ch)
        self.cpcf2 = CPCFModule(f2_ch, stage=2, f1_ch=f1_ch, f2_ch=f2_ch, f3_ch=f3_ch, f4_ch=f4_ch)
        self.cpcf3 = CPCFModule(f3_ch, stage=3, f1_ch=f1_ch, f2_ch=f2_ch, f3_ch=f3_ch, f4_ch=f4_ch)
        self.cpcf4 = CPCFModule(f4_ch, stage=4, f1_ch=f1_ch, f2_ch=f2_ch, f3_ch=f3_ch, f4_ch=f4_ch)

        # -------------------------
        # Decoder
        # -------------------------
        self.up5 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBNReLU(base_ch * 16, base_ch * 8, k=3),
        )

        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBNReLU(base_ch * 8, base_ch * 4, k=3),
        )

        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBNReLU(base_ch * 4, base_ch * 2, k=3),
        )

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBNReLU(base_ch * 2, base_ch, k=3),
        )

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBNReLU(base_ch, base_ch, k=3),
        )

        # Final output
        self.final_upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.final_conv = nn.Conv2d(base_ch, out_channels, 1)

    def forward(self, x):
        # Initial stem
        x = self.stem(x)

        # Encoder forward pass
        f1 = self.stage1(x)  # base_ch
        f2 = self.stage2(self.pool1(f1))  # base_ch * 2
        f3 = self.stage3(self.pool2(f2))  # base_ch * 4
        f4 = self.stage4(self.pool3(f3))  # base_ch * 8
        f5 = self.stage5(self.pool4(f4))  # base_ch * 16

        # CPCF fusion
        m1 = self.cpcf1(f1, f2, f3, f4)
        m2 = self.cpcf2(f1, f2, f3, f4)
        m3 = self.cpcf3(f1, f2, f3, f4)
        m4 = self.cpcf4(f1, f2, f3, f4)

        # Decoder with skip connections
        d4 = self.up5(f5) + m4
        d3 = self.up4(d4) + m3
        d2 = self.up3(d3) + m2
        d1 = self.up2(d2) + m1
        d0 = self.up1(d1)

        # Final output
        out = self.final_upsample(d0)
        out = self.final_conv(out)
        return out