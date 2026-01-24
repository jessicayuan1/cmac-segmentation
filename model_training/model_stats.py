# This file is not related to training at all
# Computes parameter counts for models

import torch
import torch.nn as nn


def count_parameters(model: nn.Module):
    """
    Count total and trainable parameters in a model.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    return total_params, trainable_params


def analyze_model(model: nn.Module):
    """
    Print parameter count for a model.
    """
    total_params, trainable_params = count_parameters(model)

    print("Model Parameter Analysis")
    print(f"Total parameters:      {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")


if __name__ == "__main__":
    from CMAC_net_definition.model.CMAC import CMACNet
    from HydraLANet_Definition.model.hydralanet import HydraLANet
    """
    model = CMACNet(
        in_channels = 3,
        out_channels = 4,
        base_channels = 16,
        depths = [1, 2, 3, 6],
        img_size = 512,
        drop_path_rate = 0.15
    )"""
    model = HydraLANet()

    analyze_model(model = model)
