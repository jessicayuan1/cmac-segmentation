import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import (
    ConvBNReLU, HaarWaveletDownsampling, DepthwiseSeparable,
    PolarizedSelfAttention, MAC, MLDA, MMAC,
    ResizeOp, CPCFModule, HydraSegHead
)

class CMACNet(nn.Module):
    """
    CMAC-Net: Cascade Multi-Scale Attention Convolution Network
    for Diabetic Retinopathy Lesion Segmentation
    
    Paper: https://www.sciencedirect.com/science/article/pii/S1746809425009954
    """
    def __init__(
        self,
        in_channels = 3,
        out_channels = 4,
        base_channels = 32,
        depths = [1, 2, 3, 6],  # [stage1, stage2, stage3, stage4]
        img_size = 512,
        drop_path_rate = 0.15,
        apply_sigmoid = True
    ):
        super().__init__()

        C = base_channels
        self.img_size = img_size
        self.apply_sigmoid = apply_sigmoid

        # -------------------------
        # Linear DropPath schedule
        # -------------------------
        # Total blocks: stage1 + stage2 + stage3 + stage4 + stage5(3)
        total_blocks = sum(depths) + 3
        dpr = torch.linspace(0, drop_path_rate, total_blocks).tolist()
        dp_idx = 0

        # -------------------------
        # Stem with Haar Wavelet Downsampling
        # -------------------------
        self.hwd = HaarWaveletDownsampling(in_channels)
        
        # Project from 12 channels (3 * 4) to base_channels
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels * 4, C, 3, stride = 1, padding = 1),
            nn.BatchNorm2d(C),
            nn.GELU(),
        )

        # -------------------------
        # Encoder Stages
        # - Stage 1: 1 MAC block, output C×1, H/2, W/2
        # - Stage 2: 2 MAC blocks, output C×2, H/4, W/4  
        # - Stage 3: 3 MMAC blocks, output C×4, H/8, W/8
        # - Stage 4: 6 MMAC blocks, output C×8, H/16, W/16
        # - Stage 5: 3 MMAC blocks, output C×16, H/32, W/32 (bottleneck)
        # -------------------------

        # Stage 1: MAC blocks, channels stay at C
        self.stage1 = nn.Sequential(*[
            MAC(C, C, drop_path = dpr[dp_idx + i])
            for i in range(depths[0])
        ])
        dp_idx += depths[0]
        self.down1 = nn.Sequential(
            nn.Conv2d(C, C, kernel_size = 3, stride = 2, padding = 1, bias = False),
            nn.BatchNorm2d(C),
            nn.GELU()
        )  # H/2 -> H/4

        # Stage 2: MAC blocks, C -> C*2
        stage2_blocks = []
        stage2_blocks.append(MAC(C, C * 2, drop_path = dpr[dp_idx]))
        for i in range(depths[1] - 1):
            stage2_blocks.append(MAC(C * 2, C * 2, drop_path = dpr[dp_idx + i + 1]))
        self.stage2 = nn.Sequential(*stage2_blocks)
        dp_idx += depths[1]
        self.down2 = nn.Sequential(
            nn.Conv2d(C * 2, C * 2, kernel_size = 3, stride = 2, padding = 1, bias = False),
            nn.BatchNorm2d(C * 2),
            nn.GELU()
        )  # H/4 -> H/8

        # Stage 3: MMAC blocks, C*2 -> C*4
        stage3_blocks = []
        stage3_blocks.append(MMAC(C * 2, C * 4, drop_path = dpr[dp_idx]))
        for i in range(depths[2] - 1):
            stage3_blocks.append(MMAC(C * 4, C * 4, drop_path = dpr[dp_idx + i + 1]))
        self.stage3 = nn.Sequential(*stage3_blocks)
        dp_idx += depths[2]
        self.down3 = nn.Sequential(
            nn.Conv2d(C * 4, C * 4, kernel_size = 3, stride = 2, padding = 1, bias = False),
            nn.BatchNorm2d(C * 4),
            nn.GELU()
        )  # H/8 -> H/16

        # Stage 4: MMAC blocks, C*4 -> C*8
        stage4_blocks = []
        stage4_blocks.append(MMAC(C * 4, C * 8, drop_path = dpr[dp_idx]))
        for i in range(depths[3] - 1):
            stage4_blocks.append(MMAC(C * 8, C * 8, drop_path = dpr[dp_idx + i + 1]))
        self.stage4 = nn.Sequential(*stage4_blocks)
        dp_idx += depths[3]
        self.down4 = nn.Sequential(
            nn.Conv2d(C * 8, C * 8, kernel_size = 3, stride = 2, padding = 1, bias = False),
            nn.BatchNorm2d(C * 8),
            nn.GELU()
        )  # H/16 -> H/32

        # Stage 5 (Bottleneck): 3 MMAC blocks, C*8 -> C*16
        self.stage5 = nn.Sequential(
            MMAC(C * 8, C * 16, drop_path = dpr[dp_idx]),
            MMAC(C * 16, C * 16, drop_path = dpr[dp_idx + 1]),
            MMAC(C * 16, C * 16, drop_path = dpr[dp_idx + 2]),
        )

        # -------------------------
        # CPCF Modules
        # -------------------------
        f1_ch = C         # C×1
        f2_ch = C * 2     # C×2
        f3_ch = C * 4     # C×4
        f4_ch = C * 8     # C×8

        self.cpcf1 = CPCFModule(f1_ch, stage = 1, f1_ch = f1_ch, f2_ch = f2_ch, f3_ch = f3_ch, f4_ch = f4_ch)
        self.cpcf2 = CPCFModule(f2_ch, stage = 2, f1_ch = f1_ch, f2_ch = f2_ch, f3_ch = f3_ch, f4_ch = f4_ch)
        self.cpcf3 = CPCFModule(f3_ch, stage = 3, f1_ch = f1_ch, f2_ch = f2_ch, f3_ch = f3_ch, f4_ch = f4_ch)
        self.cpcf4 = CPCFModule(f4_ch, stage = 4, f1_ch = f1_ch, f2_ch = f2_ch, f3_ch = f3_ch, f4_ch = f4_ch)

        # -------------------------
        # Decoder
        # -------------------------
        self.dec5 = nn.Sequential(
            nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners = False),
            nn.Conv2d(C * 16, C * 8, 3, padding = 1),
            nn.BatchNorm2d(C * 8),
            nn.GELU(),
        )
        
        self.dec4 = nn.Sequential(
            nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners = False),
            nn.Conv2d(C * 8, C * 4, 3, padding = 1),
            nn.BatchNorm2d(C * 4),
            nn.GELU(),
        )
        
        self.dec3 = nn.Sequential(
            nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners = False),
            nn.Conv2d(C * 4, C * 2, 3, padding = 1),
            nn.BatchNorm2d(C * 2),
            nn.GELU(),
        )
        
        self.dec2 = nn.Sequential(
            nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners = False),
            nn.Conv2d(C * 2, C, 3, padding = 1),
            nn.BatchNorm2d(C),
            nn.GELU(),
        )

        # -------------------------
        # Segmentation Head
        # Upsample back to original resolution (×2 due to HWD)
        # -------------------------
        self.final_upsample = nn.Upsample(
            scale_factor = 2, 
            mode = 'bilinear', 
            align_corners = False
        )
        self.seg_head = HydraSegHead(
            in_ch = C,
            num_classes = out_channels
        )

    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor (B, 3, H, W)
            
        Returns:
            Segmentation logits (B, out_channels, H, W)
        """
        # Haar Wavelet Downsampling + Stem
        x = self.hwd(x)   # (B, 3, H, W) -> (B, 12, H/2, W/2)
        x = self.stem(x)  # (B, 12, H/2, W/2) -> (B, C, H/2, W/2)

        # Encoder - extract multi-scale features
        f1 = self.stage1(x)              # (B, C, H/2, W/2)
        f2 = self.stage2(self.down1(f1)) # (B, C*2, H/4, W/4)
        f3 = self.stage3(self.down2(f2)) # (B, C*4, H/8, W/8)
        f4 = self.stage4(self.down3(f3)) # (B, C*8, H/16, W/16)
        f5 = self.stage5(self.down4(f4)) # (B, C*16, H/32, W/32)

        # CPCF - fuse multi-level features
        m1 = self.cpcf1(f1, f2, f3, f4)  # (B, C, H/2, W/2)
        m2 = self.cpcf2(f1, f2, f3, f4)  # (B, C*2, H/4, W/4)
        m3 = self.cpcf3(f1, f2, f3, f4)  # (B, C*4, H/8, W/8)
        m4 = self.cpcf4(f1, f2, f3, f4)  # (B, C*8, H/16, W/16)

        # Decoder - progressive upsampling with skip connections
        de4 = m4 + self.dec5(f5)  # (B, C*8, H/16, W/16)
        de3 = m3 + self.dec4(de4) # (B, C*4, H/8, W/8)
        de2 = m2 + self.dec3(de3) # (B, C*2, H/4, W/4)
        de1 = m1 + self.dec2(de2) # (B, C, H/2, W/2)

        # Segmentation head - upsample to original resolution
        out = self.final_upsample(de1)  # (B, C, H, W)
        out = self.seg_head(out)        # (B, out_channels, H, W)

        if self.apply_sigmoid:
            out = torch.sigmoid(out)

        return out