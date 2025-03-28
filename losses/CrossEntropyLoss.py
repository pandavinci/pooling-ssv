import torch.nn as nn
from losses.BaseLoss import BaseLoss


class CrossEntropyLoss(BaseLoss):
    """
    Cross Entropy Loss wrapper that inherits from BaseLoss.
    This is a wrapper around torch.nn.CrossEntropyLoss.
    """
    
    def __init__(self, weight=None, reduction='mean'):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(weight=weight, reduction=reduction)
    
    def forward(self, logits, targets):
        """
        Calculate the cross entropy loss.
        
        Args:
            logits: Output from the model
            targets: Ground truth labels
            
        Returns:
            loss: Cross entropy loss value
        """
        return self.loss_fn(logits, targets) 