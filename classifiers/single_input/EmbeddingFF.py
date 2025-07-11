import torch.nn.functional as F

from classifiers.FFBase import FFBase


class EmbeddingFF(FFBase):
    """
    Feedforward classifier for audio embeddings that returns processed embeddings.
    Used with embedding-based loss functions like AAM which operate on the
    embeddings from the feature processor rather than the final classifier output.
    """

    def __init__(self, extractor, feature_processor, loss_fn, in_dim=1024, num_classes=2):
        """
        Initialize the model.

        param extractor: Model to extract features from audio data.
                         Needs to provide method extract_features(input_data)
        param feature_processor: Model to process the extracted features.
                                 Needs to provide method __call__(input_data)
        param in_dim: Dimension of the input data to the classifier, divisible by 4.
        param num_classes: Number of output classes (default: 2)
        """

        super().__init__(extractor, feature_processor, loss_fn, in_dim=in_dim, num_classes=num_classes)

    def forward(self, waveforms):
        """
        Forward pass through the model.

        Extract features from the audio data and process them through the feature
        processor. Return both the classifier output and the processed embeddings
        for use with the AAM loss.

        param embeddings: Audio waveforms of shape: (batch_size, seq_len)

        return: Tuple of (logits, probabilities, processed_embeddings).
               - logits: Raw model outputs for classification
               - probabilities: Softmax of logits for classification
               - processed_embeddings: Features from the feature processor,
                 to be used with the AAM loss
        """

        # Get raw embeddings from extractor
        raw_embs = self.extractor.extract_features(waveforms)
        
        # Process through feature processor to get processed embeddings
        processed_embeddings = self.feature_processor(raw_embs)
        
        # Get logits and probabilities for classification
        #out = self.classifier(processed_embeddings)
        #prob = F.softmax(out, dim=1)

        return processed_embeddings