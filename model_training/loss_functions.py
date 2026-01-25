"""
This file contains implementation for 3 loss functions.
1. Focal Tversky
2. Binary Cross Entropy with Predication Probabilities (Not with Logits)
3. Dual Loss (A combination of the previous 2)

This file contains built in mechanisms to adjust hyperparameters for Focal Tversky and Dual Loss.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn

class FocalTverskyLoss(nn.Module):
    """
    Multi-Label Focal Tversky Loss for Semantic Segmentation (Expect Probabilities)

    Expected Shapes:
        probs: (B, C, H, W) values in [0, 1]
        targets: (B, C, H, W) values in {0, 1}

    alpha: controls penalty on false postives (Can be a vector to adjust values per class)
    beta: controls penalty on false negatives (Can be a vector to adjust values per class)
    gamma: focal parameter
    smooth: used to prevent division by 0
    class_weights: explicit scalar reweighting per class using an vector
    """
    def __init__(
        self, alpha = 0.3, beta = 0.7, gamma = 1.3, smooth = 1e-6, class_weights = None,
    ):
        super().__init__()

        self.register_buffer("alpha", torch.as_tensor(alpha, dtype = torch.float32))
        self.register_buffer("beta", torch.as_tensor(beta, dtype = torch.float32))
        self.register_buffer("gamma", torch.as_tensor(gamma, dtype = torch.float32))

        if class_weights is None: # Equal weighting if no input weights
            class_weights = torch.ones(4)
        self.register_buffer(
            "class_weights",
            torch.as_tensor(class_weights, dtype = torch.float32)
        )
        self.register_buffer(
            "smooth",
            torch.tensor(smooth, dtype = torch.float32)
        )

    def forward(self, probs, targets):
        # Safety checks
        assert probs.ndim == 4, f"Expected probs (B,C,H,W), got {probs.shape}"
        assert targets.shape == probs.shape, "Targets must match probs shape"

        assert torch.all(probs >= 0) and torch.all(probs <= 1), \
            "FocalTverskyLoss expects probabilities in [0, 1]"

        # Enforce Binary Targets
        assert torch.all((targets == 0) | (targets == 1)), \
            "Targets must be binary (0 or 1)"

        targets = targets.float()

        B, C, H, W = probs.shape

        # Flatten spatial dims
        probs_flat = probs.view(B, C, -1)
        targets_flat = targets.view(B, C, -1)

        # TP / FP / FN
        TP = (probs_flat * targets_flat).sum(dim = 2)
        FP = (probs_flat * (1 - targets_flat)).sum(dim = 2)
        FN = ((1 - probs_flat) * targets_flat).sum(dim = 2)

        alpha = self.alpha if self.alpha.ndim > 0 else self.alpha.expand(C)
        beta = self.beta if self.beta.ndim > 0 else self.beta.expand(C)
        gamma = self.gamma if self.gamma.ndim > 0 else self.gamma.expand(C)

        alpha = alpha.view(1, C)
        beta = beta.view(1, C)
        gamma = gamma.view(1, C)

        # Tversky + focal term
        tversky = (TP + self.smooth) / (
            TP + alpha * FP + beta * FN + self.smooth
        )

        loss_per_class = (1 - tversky) ** gamma 

        loss_per_class = loss_per_class.mean(dim = 0) 
        weighted_loss = (loss_per_class * self.class_weights).mean()
        return weighted_loss

class BCELossMultiLabel(nn.Module):
    """
    Multi-Label Binary Cross Entropy (BCE) Loss (Expects Probabilities)

    Expected Shapes:
        probs:   (B, C, H, W) values in [0, 1]
        targets: (B, C, H, W) values in {0, 1}
    
    class_weight: explicit scalar reweighting per class using a vector
    """
    def __init__(self, class_weights = None, pos_weight = None):
        super().__init__()

        if class_weights is None:
            class_weights = torch.ones(4)
        self.register_buffer(
            "class_weights",
            torch.as_tensor(class_weights, dtype = torch.float32)
        )

        if pos_weight is not None:
            if not isinstance(pos_weight, torch.Tensor):
                pos_weight = torch.tensor([pos_weight] * 4)
            self.register_buffer("pos_weight", pos_weight.float())
        else:
            self.pos_weight = None

    def forward(self, probs, targets):
        # Safety checks
        assert probs.ndim == 4, f"Expected probs (B,C,H,W), got {probs.shape}"
        assert probs.shape == targets.shape, "Targets must match probs shape"

        assert torch.all(probs >= 0) and torch.all(probs <= 1), \
            "BCELossMultiLabel expects probabilities in [0, 1]"

        assert torch.all((targets == 0) | (targets == 1)), \
            "Targets must be binary (0 or 1)"

        targets = targets.float()

        eps = 1e-7
        probs = torch.clamp(probs, eps, 1 - eps)

        if self.pos_weight is not None:
            pw = self.pos_weight.view(1, -1, 1, 1)
            loss = (
                -pw * targets * torch.log(probs)
                - (1 - targets) * torch.log(1 - probs)
            )
        else:
            loss = (
                -targets * torch.log(probs)
                - (1 - targets) * torch.log(1 - probs)
            )

        loss = loss.mean(dim = (2, 3)).mean(dim = 0)
        weighted_loss = (loss * self.class_weights).mean()

        return weighted_loss
    
class DualLoss(nn.Module):
    """
    Combined Loss Function (Mixed Focal Tversky and BCE) for Multi-Label Semantic Segmentation 

    Final Loss:
        L = w_ft * FocalTverskyLoss + w_bce * BCELossMultiLabel

    Expected Shapes:
        probs:   (B, C, H, W) values in [0, 1]
        targets: (B, C, H, W) values in {0, 1}
    
    w_ft = weight on Focal Tversky
    w_bce = weight on BCE
    alpha/beta/gamma/smooth = Focal Tversky Hyperparameters
    """

    def __init__(
        self, class_weights = None, w_ft = 0.5, w_bce = 0.5, alpha = 0.3, beta = 0.7,
        gamma = 1.3, smooth = 1e-6, pos_weight = None
    ):
        super().__init__()
        self.w_ft = w_ft
        self.w_bce = w_bce

        self.ft = FocalTverskyLoss(
            alpha = alpha,
            beta = beta,
            gamma = gamma,
            smooth = smooth,
            class_weights = class_weights,
        )

        self.bce = BCELossMultiLabel(
            class_weights = class_weights,
            pos_weight = pos_weight,
        )

    def forward(self, probs, targets):
        # Safety Checkes
        assert probs.ndim == 4, f"Expected probs (B,C,H,W), got {probs.shape}"
        assert probs.shape == targets.shape, "Targets must match probs shape"

        # Combined Loss
        loss_ft = self.ft(probs, targets)
        loss_bce = self.bce(probs, targets)

        return self.w_ft * loss_ft + self.w_bce * loss_bce