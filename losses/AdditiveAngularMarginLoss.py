import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from losses.BaseLoss import BaseLoss


class AdditiveAngularMarginLoss(BaseLoss):
    """
    Additive Angular Margin (ArcFace) Loss implementation.
    
    This loss is commonly used in face recognition and other metric learning tasks.
    It applies an additive angular margin to the cosine similarity between feature vectors and weight vectors.
    
    Reference:
    - Jiankang Deng, Jia Guo, Niannan Xue, Stefanos Zafeiriou. "ArcFace: Additive Angular Margin Loss for Deep Face Recognition"
    - https://github.com/foamliu/InsightFace-v2/blob/master/models.py
        - our code was generated using AI, but I include the original code here for reference
    """
    
    def __init__(self, in_features, out_features, margin=0.2, s=32.0, easy_margin=False):
        """
        Initialize AAM Loss.
        
        Args:
            in_features: Size of the input features (embedding dimension)
            out_features: Number of classes
            margin: Angular margin to penalize distances between embeddings (default: 0.2)
            s: Feature scale/radius (default: 32.0)
            easy_margin: Use easy margin version (default: False)
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.margin = margin
        self.s = s
        self.easy_margin = easy_margin
        
        # Initialize weights for the classifier
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        
        # Calculate cos(pi - margin) for the easy margin option
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin
        
    def forward(self, embeddings, labels):
        """
        Calculate the AAM loss.
        
        Args:
            embeddings: Feature embeddings from the model (N, in_features)
            labels: Ground truth labels (N,)
            
        Returns:
            loss: AAM loss value
        """
        # Normalize embeddings and weights
        embeddings = F.normalize(embeddings)
        weights = F.normalize(self.weight)
        
        # Calculate cosine similarity
        cosine = F.linear(embeddings, weights)
        
        # Get one-hot encoding for labels
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        
        # Calculate sin and cos values
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        
        # Add angular margin
        phi = cosine * self.cos_m - sine * self.sin_m
        
        # Apply easy margin if specified
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        # Create output with margins applied only to target class
        output = torch.where(one_hot.bool(), phi, cosine)
        
        # Apply scaling
        output = output * self.s
        
        # Calculate cross-entropy loss
        loss = F.cross_entropy(output, labels)
        
        return loss 

def main():
    """
    Test function to verify the functionality of AngularAdaptiveMarginLoss.
    Creates random embeddings and labels to simulate a small batch of data.
    """
    import numpy as np
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Test parameters
    batch_size = 8
    embedding_dim = 512
    num_classes = 40
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create random embeddings (simulating model output)
    embeddings = torch.randn(batch_size, embedding_dim).to(device)
    
    # Create random labels
    labels = torch.randint(0, num_classes, (batch_size,)).to(device)
    
    # Initialize the loss function
    criterion = AdditiveAngularMarginLoss(
        in_features=embedding_dim,
        out_features=num_classes,
        margin=0.2,
        s=30.0,
        easy_margin=False
    ).to(device)
    
    # Calculate loss
    loss = criterion(embeddings, labels)
    
    print(f"Device: {device}")
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Labels: {labels}")
    print(f"Loss value: {loss.item():.4f}")
    
    # Test backward pass
    loss.backward()
    print("Backward pass successful")
    
    # Test with different margin values
    margins = [0.3, 0.5, 0.7]
    for margin in margins:
        criterion = AdditiveAngularMarginLoss(
            in_features=embedding_dim,
            out_features=num_classes,
            margin=margin,
            s=32.0,
            easy_margin=False
        ).to(device)
        loss = criterion(embeddings, labels)
        print(f"Loss with margin {margin}: {loss.item():.4f}")

if __name__ == "__main__":
    main() 