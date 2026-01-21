# This file is not related to training at all
# Computes parameter counts and FLOPs for the models

import torch
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis
from torchinfo import summary
def count_parameters(model: nn.Module):
    """
    Count total and trainable parameters in a model.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    return total_params, trainable_params

def compute_flops(
    model: nn.Module,
    input_shape,
    device = "cpu"
):
    """
    Compute FLOPs for a single forward pass (batch size = 1).
    Args:
        model: PyTorch model
        input_shape: tuple, e.g. (1, 3, 512, 512)
        device: "cpu" or "cuda"
    Returns:
        flops (int)
    """
    model = model.to(device)
    model.eval()
    dummy_input = torch.randn(*input_shape).to(device)
    with torch.no_grad():
        flops = FlopCountAnalysis(model, dummy_input).total()
    return flops

def analyze_model(
    model: nn.Module,
    input_shape,
    device = "cpu"
):
    """
    Print parameter count and FLOPs for a model.
    """
    total_params, trainable_params = count_parameters(model)
    flops = compute_flops(model, input_shape, device)
    print("Model Complexity Analysis")
    print(f"Total parameters:      {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"FLOPs (forward pass): {flops:,}")

if __name__ == "__main__":
    from CMAC_net_definition.model.CMAC import CMACNet
    model = CMACNet(
        in_channels = 3,
        out_channels = 5,
        embed_dim = 96,
        depths = [1, 2, 3, 6],
        img_size = 512
    )


    input_shape = (1, 3, 512, 512)

    summary(model, input_size=input_shape)
    print(count_parameters(model))