import torch
from model.CMAC import CMACNet

def main():
    # Dummy 768x768 RGB input
    x = torch.randn(1, 3, 768, 768)
    # Initialize Model
    model = CMACNet(in_channels=3, out_channels=5, embed_dim=96, img_size=768, depths=[2, 2, 6, 2])
    # Forward Pass
    out = model(x)
    # Print Shapes
    print("Input shape :", x.shape)
    print("Output shape:", out.shape)
    #Input shape : torch.Size([1, 3, 768, 768])
    #Output shape: torch.Size([1, 5, 768, 768])
if __name__ == "__main__":
    main()