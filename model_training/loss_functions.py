import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalTverskyLoss(nn.Module):
    """
    Multi-Label Focal Tversky Loss for Semantic Segmentation.
    Extends on Tversky Index by adding a Focal Parameter.

    Tversky Index:
        For predicted probabilities p and ground truth t:
            TP = sum(p * t)
            FP = sum(p * (1 - t))
            FN = sum((1 - p) * t)
        Tversky Index is given by:

                                        TP + smooth
            Tversky Index = -------------------------------------
                            TP + alpha * FP + beta * FN + smooth
            where alpha controls the penalty on false positives
            and beta controls the penalty on false negatives.

    Focal Tversky Loss:
        To focus on harder tasks, the loss becomes:
            Focal Tversky Loss = (1 - Tversky Index) ^ gamma
        Where gamma >= 1 increases the emphasis on harder classes

    Expected Shapes:
        logits: (B, C, H, W)
            Raw model outputs before sigmoid.
        targets: (B, C, H, W)
            Binary ground truth masks for each class/channel.

    Arguments:
        alpha (float or Tensor): False Positive penalty coefficient(s)
            - If float: same alpha for all classes
            - If Tensor of shape (C,): per-class alphas
        beta (float or Tensor): False Negative penalty coefficient(s)
            - If float: same beta for all classes
            - If Tensor of shape (C,): per-class betas
        gamma (float or Tensor): Focal exponent(s)
            - If float: same gamma for all classes
            - If Tensor of shape (C,): per-class gammas
        smooth (float): Stability constant
        class_weights (Tensor or None): Per-class scalar multiplier of shape (C,)
            Example for (EX, HE, MA, SE): [1.0, 1.5, 2.0, 1.0]

    Returns:
        Scalar Focal Tversky Loss (torch.Tensor)
    """
    def __init__(
        self,
        alpha = 0.3,           # Can be float or Tensor(C,)
        beta = 0.7,            # Can be float or Tensor(C,)
        gamma = 1.3,           # Can be float or Tensor(C,)
        smooth = 1e-6,
        class_weights = None,  # Tensor shape (C,) or None
    ):
        super().__init__()
        
        # Convert to tensors and register as buffers
        self.register_buffer("alpha", torch.as_tensor(alpha, dtype = torch.float32))
        self.register_buffer("beta", torch.as_tensor(beta, dtype = torch.float32))
        self.register_buffer("gamma", torch.as_tensor(gamma, dtype = torch.float32))
        
        # Validate shapes if per-class parameters are provided
        if self.alpha.ndim > 0 and self.beta.ndim > 0:
            assert self.alpha.shape == self.beta.shape, \
                f"alpha and beta must have same shape, got {self.alpha.shape} and {self.beta.shape}"
        
        if class_weights is None:
            class_weights = torch.ones(4)
        self.register_buffer("class_weights", torch.as_tensor(class_weights, dtype = torch.float32))
        
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        logits:  (B, C, H, W)
        targets: (B, C, H, W)
        """
        probs = torch.sigmoid(logits)
        
        B, C, H, W = probs.shape
        
        # Flatten spatial dimensions
        probs_flat = probs.view(B, C, -1)      # (B, C, H*W)
        targets_flat = targets.view(B, C, -1).float()  # (B, C, H*W)
        
        # Compute TP, FP, FN for each class
        TP = (probs_flat * targets_flat).sum(dim = 2)               # (B, C)
        FP = (probs_flat * (1 - targets_flat)).sum(dim = 2)         # (B, C)
        FN = ((1 - probs_flat) * targets_flat).sum(dim = 2)         # (B, C)
        
        # Expand alpha, beta, gamma to match (1, C) if they're scalars
        alpha = self.alpha if self.alpha.ndim > 0 else self.alpha.expand(C)
        beta = self.beta if self.beta.ndim > 0 else self.beta.expand(C)
        gamma = self.gamma if self.gamma.ndim > 0 else self.gamma.expand(C)
        
        # Reshape to (1, C) for broadcasting
        alpha = alpha.view(1, C)
        beta = beta.view(1, C)
        gamma = gamma.view(1, C)
        
        # Compute Tversky index
        tversky = (TP + self.smooth) / (
            TP + alpha * FP + beta * FN + self.smooth
        )  # (B, C)
        
        # Apply focal component
        loss_per_class = (1 - tversky) ** gamma   # (B, C)
        
        # Average across batch, then weight and average across classes
        loss_per_class = loss_per_class.mean(dim = 0)  # (C,)
        weighted_loss = (loss_per_class * self.class_weights).mean()
        
        return weighted_loss


class BCEwithLogitsLossMultiLabel(nn.Module):
    """
    Multi-Label Binary Cross Entropy (BCE) Loss for Semantic Segmentation.
    Loss is applied to multi-channel segmentation masks where each channel 
    represents a separate binary mask.

    For a predicted probability p ∈ [0, 1] and target t ∈ {0, 1}:
        BCE(p, t) = - [ t * log(p) + (1 - t) * log(1 - p) ]

    Expected Shapes:
        logits: (B, C, H, W)
            Raw model outputs before sigmoid.
        targets: (B, C, H, W)
            Binary ground-truth masks for each class.
            Each channel must be 0 or 1.
            
    Arguments:
        class_weights (Tensor or None): Per-class scalar multiplier of shape (C,)
        
    Returns:
        Scalar Binary Cross Entropy Loss (torch.Tensor)
    """
    def __init__(self, class_weights = None):
        super().__init__()
        
        if class_weights is None:
            class_weights = torch.ones(4)
        self.register_buffer("class_weights", torch.as_tensor(class_weights, dtype = torch.float32))
        
        self.bce = nn.BCEWithLogitsLoss(reduction = "none")

    def forward(self, logits, targets):
        """
        logits:  (B, C, H, W)
        targets: (B, C, H, W)
        """
        loss = self.bce(logits, targets.float())  # (B, C, H, W)
        
        # Average over spatial dimensions
        loss = loss.mean(dim = (2, 3))   # (B, C)
        # Average over batch
        loss = loss.mean(dim = 0)        # (C,)
        
        # Apply class weights and final averaging
        weighted_loss = (loss * self.class_weights).mean()
        
        return weighted_loss

    
class DualLoss(nn.Module):
    """
    Combined Loss Function for Semantic Segmentation.

    Final Loss:
        L = w_ft * Focal Tversky Loss + w_bce * BCEWithLogits Loss

    Arguments:
        class_weights (Tensor or None): Per-class scalar multiplier of shape (C,)
        w_ft (float): Weight for Focal Tversky loss
        w_bce (float): Weight for BCE loss
        alpha (float or Tensor): Focal Tversky FP penalty
        beta (float or Tensor): Focal Tversky FN penalty
        gamma (float or Tensor): Focal Tversky focal exponent
        smooth (float): Stability constant

    Expected Shapes:
        logits: (B, C, H, W)
        targets: (B, C, H, W)

    Returns:
        Scalar combined loss value (torch.Tensor)
        
    Example Usage:
        # Uniform class weights (all classes treated equally)
        loss_fn = DualLoss(class_weights = torch.tensor([1.0, 1.0, 1.0, 1.0]))
        
        # Non-uniform class weights (emphasize certain classes)
        # E.g., for 4 classes (EX, HE, MA, SE): weight MA (microaneurysms) 2x more
        loss_fn = DualLoss(class_weights = torch.tensor([1.0, 1.0, 2.0, 1.0]))
        
        # Per-class alpha/beta with uniform class weights
        # Higher alpha = penalize FP more, higher beta = penalize FN more
        loss_fn = DualLoss(
            class_weights = torch.tensor([1.0, 1.0, 1.0, 1.0]),  # Uniform weights
            alpha = torch.tensor([0.4, 0.4, 0.2, 0.4]),          # Per-class FP penalty
            beta = torch.tensor([0.6, 0.6, 0.8, 0.6]),           # Per-class FN penalty
    """

    def __init__(
        self,
        class_weights = None,
        w_ft = 0.5,
        w_bce = 0.5,
        alpha = 0.3,
        beta = 0.7,
        gamma = 1.3,
        smooth = 1e-6
    ):
        super().__init__()
        
        self.w_ft = w_ft
        self.w_bce = w_bce
        
        self.ft = FocalTverskyLoss(
            alpha = alpha,
            beta = beta,
            gamma = gamma,
            smooth = smooth,
            class_weights = class_weights
        )
        
        self.bce = BCEwithLogitsLossMultiLabel(
            class_weights = class_weights
        )

    def forward(self, logits, targets):
        """
        logits:  (B, C, H, W)
        targets: (B, C, H, W)
        """
        return (
            self.w_ft * self.ft(logits, targets) +
            self.w_bce * self.bce(logits, targets)
        )