import torch
from torch import nn
from torch.nn import functional as F


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        if logits.shape != targets.shape:
            raise ValueError(f"Method returned {tuple(logits.shape)}, expected {tuple(targets.shape)}")
        # Float32 reductions remain stable under mixed precision.
        logits, targets = logits.float(), targets.float()
        bce = F.binary_cross_entropy_with_logits(logits, targets)
        probabilities = torch.sigmoid(logits)
        dimensions = (1, 2, 3)
        intersection = (probabilities * targets).sum(dim=dimensions)
        denominator = probabilities.sum(dim=dimensions) + targets.sum(dim=dimensions)
        dice_loss = 1 - ((2 * intersection + 1) / (denominator + 1)).mean()
        return self.bce_weight * bce + self.dice_weight * dice_loss
