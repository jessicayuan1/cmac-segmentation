"""
This file is not related to training at all.
It simply computes parameter counts for models.
"""
import torch
import torch.nn as nn

def count_parameters(model: nn.Module):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    return total_params, trainable_params

def analyze_model(model: nn.Module):
    total_params, trainable_params = count_parameters(model)

    print("Model Parameter Analysis")
    print(f"Total parameters:      {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

if __name__ == "__main__":
    from HydraLANet_Definition.model.hydralanet import HydraLANet
    model = HydraLANet()

    analyze_model(model = model)
