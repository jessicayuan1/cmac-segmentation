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
        drop_path_rate=0.15,
    ):
        super().__init__()

        if depths is None:
            depths = [2, 2, 6, 2]

        base_ch = embed_dim // 3
        self.img_size = img_size

        # -------------------------
        # Linear DropPath schedule
        # -------------------------
        total_blocks = sum(depths) + 3
        dpr = torch.linspace(0, drop_path_rate, total_blocks).tolist()
        dp_idx = 0

        # -------------------------
        # Stem
        # -------------------------
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_ch, 3, stride=2, padding=1),
            nn.Conv2d(base_ch, base_ch, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_ch),
            nn.GELU(),
        )

        # -------------------------
        # Encoder (MAC / MMAC)
        # -------------------------
        self.stage1 = nn.Sequential(*[
            MAC(base_ch, base_ch, drop_path=dpr[dp_idx + i])
            for i in range(depths[0])
        ])
        dp_idx += depths[0]
        self.pool1 = nn.MaxPool2d(2)

        self.stage2 = nn.Sequential(
            MAC(base_ch, base_ch * 2, drop_path=dpr[dp_idx]),
            *[
                MAC(base_ch * 2, base_ch * 2, drop_path=dpr[dp_idx + i + 1])
                for i in range(depths[1] - 1)
            ]
        )
        dp_idx += depths[1]
        self.pool2 = nn.MaxPool2d(2)

        self.stage3 = nn.Sequential(
            MMAC(base_ch * 2, base_ch * 4, drop_path=dpr[dp_idx]),
            *[
                MMAC(base_ch * 4, base_ch * 4, drop_path=dpr[dp_idx + i + 1])
                for i in range(depths[2] - 1)
            ]
        )
        dp_idx += depths[2]
        self.pool3 = nn.MaxPool2d(2)

        self.stage4 = nn.Sequential(
            MMAC(base_ch * 4, base_ch * 8, drop_path=dpr[dp_idx]),
            *[
                MMAC(base_ch * 8, base_ch * 8, drop_path=dpr[dp_idx + i + 1])
                for i in range(depths[3] - 1)
            ]
        )
        dp_idx += depths[3]
        self.pool4 = nn.MaxPool2d(2)

        # -------------------------
        # Bottleneck (fixed 3 MMAC)
        # -------------------------
        self.stage5 = nn.Sequential(
            MMAC(base_ch * 8, base_ch * 16, drop_path=dpr[dp_idx]),
            MMAC(base_ch * 16, base_ch * 16, drop_path=dpr[dp_idx + 1]),
            MMAC(base_ch * 16, base_ch * 16, drop_path=dpr[dp_idx + 2]),
        )

        # -------------------------
        # CPCF modules
        # -------------------------
        f1_ch, f2_ch, f3_ch, f4_ch = base_ch, base_ch * 2, base_ch * 4, base_ch * 8

        self.cpcf1 = CPCFModule(f1_ch, 1, f1_ch, f2_ch, f3_ch, f4_ch)
        self.cpcf2 = CPCFModule(f2_ch, 2, f1_ch, f2_ch, f3_ch, f4_ch)
        self.cpcf3 = CPCFModule(f3_ch, 3, f1_ch, f2_ch, f3_ch, f4_ch)
        self.cpcf4 = CPCFModule(f4_ch, 4, f1_ch, f2_ch, f3_ch, f4_ch)

        # -------------------------
        # Decoder
        # -------------------------
        self.up5 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBNReLU(base_ch * 16, base_ch * 8),
        )
        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBNReLU(base_ch * 8, base_ch * 4),
        )
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBNReLU(base_ch * 4, base_ch * 2),
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBNReLU(base_ch * 2, base_ch),
        )
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBNReLU(base_ch, base_ch),
        )

        self.final_upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.final_conv = nn.Conv2d(base_ch, out_channels, 1)

    def forward(self, x):
        x = self.stem(x)

        f1 = self.stage1(x)
        f2 = self.stage2(self.pool1(f1))
        f3 = self.stage3(self.pool2(f2))
        f4 = self.stage4(self.pool3(f3))
        f5 = self.stage5(self.pool4(f4))

        m1 = self.cpcf1(f1, f2, f3, f4)
        m2 = self.cpcf2(f1, f2, f3, f4)
        m3 = self.cpcf3(f1, f2, f3, f4)
        m4 = self.cpcf4(f1, f2, f3, f4)

        d4 = self.up5(f5) + m4
        d3 = self.up4(d4) + m3
        d2 = self.up3(d3) + m2
        d1 = self.up2(d2) + m1
        d0 = self.up1(d1)

        out = self.final_upsample(d0)
        return self.final_conv(out)
