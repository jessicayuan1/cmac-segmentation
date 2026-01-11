import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalTverskyLoss(nn.Module):
    """
    Multi-Label Focal Tversky Loss for Semantic Segmentation.
    Extends on Tversky Index by adding a Focal Parameter.

    Tversky Index:
        For predicted probablities p and ground truth t:
            TP = sum(p * t)
            FP = sum(p * (1 - t))
            FN = sum((1 - p) * t)
        Tversky Index is given by:

                                        TP + smooth
            Tversky Index = -------------------------------------
                            TP + alpha * FP + beta * FN + smooth
            where alpha controls the penalty on false positives
            and beta controls the penalty on false negatives.

    Focal Tversky Loss
        To focus on harder tasks, the loss becomes:

            Focal Tversky Loss = (1 - Tversky Index)^gamma
        Where gamma >= 1 increases the emphases on harder classes

    Expected Shapes:
        logits: (B, C, H, W)
            Raw model outputs before sigmoid.
        targets: (B, C, H, W)
            Binary ground truth masks for each class/channel.

    Arguments:
        alpha (float or Tensor): False Positive penalty coefficients
        beta  (float or Tensor): False Negative penalty coefficients
        gamma (float or Tensor): Focal exponents
        smooth (float): Stability Constant

    Returns:
        Scalar Focal Tversky Loss (torch.Tensor)
    """

    def __init__(self, alpha = 0.3, beta = 0.7, gamma = 1.3, smooth = 1e-6):
        super().__init__()

        # alpha / beta / gamma can be:
        # - scalars (same for all classes)
        # - tensors of shape (C,) for class-specific weighting
        # They are registered as buffers so they:
        # - move with the model across devices
        # - are saved in state_dict
        # - are not trainable parameters
        self.register_buffer("alpha", torch.tensor(alpha))
        self.register_buffer("beta",  torch.tensor(beta))
        self.register_buffer("gamma", torch.tensor(gamma))

        self.smooth = smooth

    def forward(self, logits, targets):
        # Sigmoid over logits -> probabilities per channel
        probs = torch.sigmoid(logits)

        # Flatten height and width: (B, C, H*W)
        probs_flat = probs.view(probs.size(0), probs.size(1), -1)
        targets_flat = targets.view(targets.size(0), targets.size(1), -1).float()

        # Compute TP, FP, FN per class
        TP = (probs_flat * targets_flat).sum(dim = 2)
        FP = (probs_flat * (1 - targets_flat)).sum(dim = 2)
        FN = ((1 - probs_flat) * targets_flat).sum(dim = 2)

        alpha = self.alpha
        beta  = self.beta
        gamma = self.gamma

        # If they are scalars, broadcast to class dimension
        if alpha.ndim == 0:
            alpha = alpha.expand(TP.size(1))
        if beta.ndim == 0:
            beta = beta.expand(TP.size(1))
        if gamma.ndim == 0:
            gamma = gamma.expand(TP.size(1))

        # Reshape for broadcasting: (1, C)
        alpha = alpha.view(1, -1)
        beta  = beta.view(1, -1)
        gamma = gamma.view(1, -1)

        # Compute Tversky Index per class
        Tversky = (TP + self.smooth) / (
            TP + alpha * FP + beta * FN + self.smooth
        )

        # Focal Tversky Loss
        focal_tversky = (1 - Tversky) ** gamma

        # Return mean across batch and classes
        return focal_tversky.mean()

class BCEwithLogitsLossMultiLabel(nn.Module):
    """
    Multi-Label Binary Cross Entropy (BCE) Loss for Semantic Segmentation.
    Loss is applied to multi-channel segmentation masks where each channel represents a separate binary mask.

    For a predicted probability p ∈ [0, 1] and target t ∈ {0, 1}:
        BCE(p, t) = - [ t * log(p) + (1 - t) * log(1 - p) ]
    where p is the probability and t is the ground-truth

    Expected Shapes:
        logits: (B, C, H, W)
            Raw model outputs before sigmoid.
        targets: (B, C, H, W)
            Binary ground-truth masks for each class.
            Each channel must be 0 or 1.
    Returns:
        Scalar Binary Cross Entropy Loss (torch.Tensor)
    """
    def __init__(self, weight = None):
        """
        Arguments:
            weight (Tensor or None):
                Optional per-class weighting tensor of shape (C,).
                Allows you to upweight certain classes.
        """
        super().__init__()
        self.weight = weight
        self.bce = nn.BCEWithLogitsLoss(weight = weight)

    def forward(self, logits, targets):
        """
        Compute BCE loss.
        Arguments:
            logits: (B, C, H, W)
                No sigmoid before this.
            targets: (B, C, H, W)
                Binary ground-truth
        Returns:
            Tensor: BCE loss scalar value.
        """
        return self.bce(logits, targets.float())
    
class DualLoss(nn.Module):
    """
    Combined Loss Function for Semantic Segmentation.

    Final Loss:
        L = w_ft * Focal Tversky Loss + w_bce * BCEWithLogits Loss

    Arguments:
        w_ft (float): Weight for Focal Tversky loss
        w_bce (float): Weight for BCE loss
        alpha, beta, gamma: Focal Tversky hyperparameters
        smooth (float): Stability constant
        bce_weight (Tensor or None): Optional element-wise BCE weight (usually None)

    Expected Shapes:
        logits: (B, C, H, W)
        targets: (B, C, H, W)

    Returns:
        Scalar combined loss value (torch.Tensor)
    """
    def __init__(
        self,
        w_ft = 0.5,
        w_bce = 0.5,
        alpha = 0.3,
        beta = 0.7,
        gamma = 1.3,
        smooth = 1e-6,
        bce_weight = None
    ):
        super().__init__()

        self.w_ft = w_ft
        self.w_bce = w_bce
        self.ft = FocalTverskyLoss(
            alpha = alpha,
            beta = beta,
            gamma = gamma,
            smooth = smooth
        )
        self.bce = BCEwithLogitsLossMultiLabel(
            weight = bce_weight
        )
    def forward(self, logits, targets):
        L_ft = self.ft(logits, targets)
        L_bce = self.bce(logits, targets)

        return self.w_ft * L_ft + self.w_bce * L_bce