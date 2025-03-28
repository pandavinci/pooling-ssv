import torch.nn as nn


class FFBase(nn.Module):
    """
    Base class for feedforward classifiers, inherited by FFConcatBase and FFDiffBase.
    """

    def __init__(self, extractor, feature_processor, in_dim=1024, num_classes=2):
        """
        Initialize the model.

        param extractor: Model to extract features from audio data.
                         Needs to provide method extract_features(input_data)
        param feature_processor: Model to process the extracted features.
                                 Needs to provide method __call__(input_data)
        param in_dim: Dimension of the input data to the classifier, divisible by 4.
        param num_classes: Number of output classes (default: 2)
        """

        super().__init__()

        self.extractor = extractor
        self.feature_processor = feature_processor

        # Allow variable input dimension, mainly for base (768 features), large (1024 features) and extra-large (1920 features) models.
        self.layer1_in_dim = in_dim
        self.layer1_out_dim = in_dim // 2
        self.layer2_in_dim = self.layer1_out_dim
        self.layer2_out_dim = self.layer2_in_dim // 2
        
        # Define feature layers and final classifier separately to allow access to embeddings
        self.feature_layers = nn.Sequential(
            nn.Linear(self.layer1_in_dim, self.layer1_out_dim),
            nn.BatchNorm1d(self.layer1_out_dim),
            nn.ReLU(),
            nn.Linear(self.layer2_in_dim, self.layer2_out_dim),
            nn.BatchNorm1d(self.layer2_out_dim),
            nn.ReLU(),
        )
        
        self.final_classifier = nn.Linear(self.layer2_out_dim, num_classes)
        
        # Combined classifier for backward compatibility
        self.classifier = nn.Sequential(
            self.feature_layers,
            self.final_classifier
        )

    def get_embeddings(self, x):
        """
        Get embeddings from the penultimate layer (before classification)
        Used for metric learning losses like AAM
        """
        return self.feature_layers(x)
        
    def forward(self, input_gt, input_tested):
        raise NotImplementedError("Forward pass not implemented in the base class.")
