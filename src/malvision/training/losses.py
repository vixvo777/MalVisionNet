"""
Custom loss functions for training.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedBCELoss(nn.Module):
    """
    Weighted Binary Cross Entropy Loss for imbalanced datasets.
    """
    def __init__(self, pos_weight=1.0):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, logits, labels):
        return F.binary_cross_entropy_with_logits(
            logits, labels, pos_weight=torch.tensor(self.pos_weight, device=logits.device)
        )
