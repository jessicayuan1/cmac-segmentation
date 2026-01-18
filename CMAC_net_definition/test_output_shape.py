import torch
from model.CMAC import CMACNet

def main():
    # Dummy RGB input
    x = torch.randn(4, 3, 512, 512)
    # Initialize Model
    model = CMACNet(in_channels = 3, 
                    out_channels = 4, 
                    embed_dim = 96, 
                    img_size = 512, 
                    depths = [1, 2, 3, 6])
    # Forward Pass
    out = model(x)
    # Print Shapes
    print("Input shape :", x.shape)
    print("Output shape:", out.shape)
if __name__ == "__main__":
    main()