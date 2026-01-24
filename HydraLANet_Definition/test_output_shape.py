import torch
from HydraLANet_Definition.model.hydralanet import HydraLANet

def main():
    # Dummy RGB input
    x = torch.randn(4, 3, 1024, 1024)

    # Initialize Model
    model = HydraLANet(
        n_channels = 3
    )

    model.eval()

    # Forward Pass
    with torch.no_grad():
        out = model(x)

    # Print Shapes
    print("Input shape :", x.shape)
    print("Output shape:", out.shape)

    # Print prediction statistics
    print("\n[Prediction stats @ random init]")
    print(f"Min  : {out.min().item():.6f}")
    print(f"Max  : {out.max().item():.6f}")
    print(f"Mean : {out.mean().item():.6f}")

if __name__ == "__main__":
    main()
