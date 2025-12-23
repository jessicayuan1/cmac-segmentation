import torch
from model.swin_unet import SwinUNet

def main():
    # Dummy 1024×1024 RGB input
    x = torch.randn(2, 3, 512, 512)
    # Initialize Model
    model = SwinUNet(
        img_size = 512,
        in_channels = 3,
        out_channels = 5,
        embed_dim = 96,
        depths = [2, 2, 6, 2],
        num_heads = [3, 6, 12, 24],
        window_size = 8
    )
    # Forward Pass
    out = model(x)
    # Print Shapes
    print("Input shape :", x.shape)
    print("Output shape:", out.shape)
if __name__ == "__main__":
    main()