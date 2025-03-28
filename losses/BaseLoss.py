import torch.nn as nn


class BaseLoss(nn.Module):
    """
    Base class for all loss functions. Inherits from nn.Module.
    All loss functions should inherit from this class and implement forward() method.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, logits, targets):
        """
        Calculate the loss.
        
        Args:
            logits: Output from the model
            targets: Ground truth labels
            
        Returns:
            loss: Loss value
        """
        raise NotImplementedError("Subclasses must implement forward()") 