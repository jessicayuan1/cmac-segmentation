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

class DiceLossMultiLabel(nn.Module):
    """
    Multi-Label Dice Loss for Semantic Segmentation.
    Each channel (label) gets its own dice score.

                                 2 * TP + smooth
    Dice Coefficient = ------------------------------------
                        (sum(pred) + sum(target) + smooth)
    Where:
            TP = sum(pred * target)
            sum(pred)   = total predicted foreground pixels
            sum(target) = total ground-truth foreground pixels

    Range: [0, 1] with 0 meaning no overlap and 1 being perfect overlap
    Dice Loss Per Class = 1 - Dice Coefficient
    Dice Loss = Mean(Dice Loss Over All Classes)

    Expected Shapes:
        logits: (B, C, H, W)
            Raw model outputs before sigmoid.
        targets: (B, C, H, W)
            Binary ground truth masks for each class/channel.
    Returns:
        Scalar Dice Loss (torch.Tensor)
    """
    def __init__(self, smooth = 1e-6):
        """
        Arguments:
            smooth (float): Small constant to prevent division by zero.
        """
        super().__init__()
        self.smooth = smooth
    def forward(self, logits, targets):
        """
        Compute Mean Dice loss across all channels.
        Arguments:
            logits: (B, C, H, W)
            targets: (B, C, H, W)
        Returns:
            Scalar Dice Loss (torch.Tensor)
        """
        # Convert logits -> probabilities in [0,1]
        probs = torch.sigmoid(logits)

        # Flatten Height and Width into single dimesnions
        # probs_flat: (B, C, H*W)
        probs_flat = probs.view(probs.size(0), probs.size(1), -1)
        targets_flat = targets.view(targets.size(0), targets.size(1), -1).float()

        # Compute Intersection and Union per class
        intersection = (probs_flat * targets_flat).sum(dim = 2)
        union = probs_flat.sum(dim = 2) + targets_flat.sum(dim = 2)

        # Dice Coefficient per Class
        dice_per_class = (2 * intersection + self.smooth) / (union + self.smooth)

        # 1 - Dice Score
        dice_loss_per_class = 1 - dice_per_class

        # Return mean
        return dice_loss_per_class.mean()

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
    
# Ignore the triloss for now
class TriLoss(nn.Module):
    """
    Combined Loss Function for Semantic Segmentation.
    Final Loss:
        L = w_ft * Focal Tversky Loss + w_dice * Dice Loss + w_bce * Binary Cross Entropy Loss
    Arguments:
        w_ft (float): Weight for Focal Tversky loss
        w_dice (float): Weight for Dice loss
        w_bce (float): Weight for BCE loss
        alpha, beta, gamma: Focal Tversky hyperparameters
        smooth (float): Stability constant for Focal Tversky/Dice Loss
        pos_weight (Tensor or None): Optional per-class weighting for BCEWithLogitsLoss

    Expected Shapes:
        logits: (B, C, H, W)
        targets: (B, C, H, W)

    Returns:
        Scalar Combined Loss Value (torch.Tensor)
    """
    def __init__(self, w_ft = 0.5, w_dice = 0.3, w_bce = 0.2, alpha = 0.3,
                 beta = 0.7, gamma = 1.3, smooth = 1e-6, pos_weight = None):
        super().__init__()
        self.w_ft = w_ft
        self.w_dice = w_dice
        self.w_bce = w_bce
        # Loss components
        self.ft = FocalTverskyLoss(alpha = alpha, beta = beta, gamma = gamma, smooth = smooth)
        self.dice = DiceLossMultiLabel(smooth = smooth)
        self.bce = BCEwithLogitsLossMultiLabel(weight = pos_weight)

    def forward(self, logits, targets):
        L_ft = self.ft(logits, targets)
        L_dice = self.dice(logits, targets)
        L_bce = self.bce(logits, targets)
        return self.w_ft * L_ft + self.w_dice * L_dice + self.w_bce * L_bce