import torch
import torch.nn as nn
from speaker_filtration.BaseFilter import BaseFilter

class SpeakerFilterBlock(nn.Module, BaseFilter):
    """
    Speaker Filtration Block.

    This block is intended to be placed after a feature extractor and before a pooling layer.
    It takes features from a main/converted audio and features from a reference speaker's audio
    to produce filtered features.
    The internal architecture (LSTM, attention, etc.) needs to be defined based on the specific
    requirements of the model (e.g., as depicted in the provided diagram).

    It may also be placed after the classification block.
    """
    def __init__(self, input_channels: int, feature_size: int, lstm_hidden_size: int = 400, *args, **kwargs):
        """
        Initializes the SpeakerFilterBlock.

        Args:
            input_channels (int): The number of channels in the input feature maps (C).
            feature_size (int): The size of the features in each channel (F).
            lstm_hidden_size (int, optional): The hidden size for the LSTM. Default is 400.
        """
        super().__init__()
        self.input_channels = input_channels
        self.feature_size = feature_size
        self.epsilon = 1e-6
        
        self.lstm_hidden_size = lstm_hidden_size

        # BiLSTM layer
        # Input features will be [orthogonal_vc, vc] concatenated, so 2 * feature_size
        self.lstm = nn.LSTM(
            input_size=2 * self.feature_size,
            hidden_size=self.lstm_hidden_size,
            num_layers=1, 
            batch_first=False, # Expects (seq_len, batch_overall, feature)
            bidirectional=True
        )

        # Fully Connected layer for mask prediction
        self.fc_mask = nn.Linear(self.lstm_hidden_size * 2, self.feature_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, converted_features: torch.Tensor, reference_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the SpeakerFilterBlock.

        Args:
            converted_features (torch.Tensor): Features from the converted/main audio (vc_features).
                                               Shape: (batch_size, time_steps, input_channels, feature_size).
            reference_features (torch.Tensor): Features from the reference speaker's audio (ref_features).
                                               Shape: (batch_size, time_steps, input_channels, feature_size).

        Returns:
            torch.Tensor: Filtered features.
                          Shape: (batch_size, time_steps, input_channels, feature_size).
        """
        B, T, C, F_dim = converted_features.shape

        # 1. Reference Processing
        # reference_features shape: (B, T, C, F)
        # ref_avg: (B, C, F) - average over time dimension (T)
        # FIXME: If reference_features are padded (e.g., to a max length T in the batch),
        # this mean calculation will include padding values, potentially skewing ref_avg.
        # Consider using a masked mean if original sequence lengths are available.
        ref_avg = torch.mean(reference_features, dim=1)
        
        # Normalization of ref_avg to get n
        # ref_avg_norm: (B, C, 1)
        ref_avg_norm = torch.linalg.norm(ref_avg, ord=2, dim=2, keepdim=True)
        # n: (B, C, F)
        n = ref_avg / (ref_avg_norm + self.epsilon)

        # 2. Feature Projection for converted_features (vc)
        # n is (B, C, F). converted_features is (B, T, C, F)
        # n_expanded: (B, 1, C, F) for broadcasting with converted_features
        n_expanded = n.unsqueeze(1)
        
        # vc_dot_n: (B, T, C, 1) - sum along feature_size dimension (F)
        vc_dot_n = torch.sum(converted_features * n_expanded, dim=3, keepdim=True)
        # parallel_vc: (B, T, C, F)
        parallel_vc = vc_dot_n * n_expanded
        # orthogonal_vc: (B, T, C, F)
        orthogonal_vc = converted_features - parallel_vc

        # 3. Mask Generation
        # lstm_input_features: (B, T, C, 2*F) - concatenate along feature_size dimension (F)
        lstm_input_features = torch.cat((orthogonal_vc, converted_features), dim=3)
        
        # Reshape for LSTM: LSTM expects (seq_len, batch_overall, num_features)
        # Here, seq_len = T, batch_overall = B*C, num_features = 2*F
        # Current shape: (B, T, C, 2*F)
        # Permute to (T, B, C, 2*F)
        lstm_input_reshaped = lstm_input_features.permute(1, 0, 2, 3)
        # Reshape to (T, B*C, 2*F)
        lstm_input_reshaped = lstm_input_reshaped.reshape(T, B * C, 2 * self.feature_size)
        
        # lstm_out: (T, B*C, lstm_hidden_size*2)
        # FIXME: If lstm_input_reshaped contains padded sequences (i.e., T is max_len in batch),
        # the LSTM will process these padding frames. This is inefficient and can affect
        # the quality of the learned representations/mask. Consider using
        # torch.nn.utils.rnn.pack_padded_sequence and pad_packed_sequence if original lengths are known.
        lstm_out, _ = self.lstm(lstm_input_reshaped)
        
        # Reshape LSTM output back: (B, T, C, lstm_hidden_size*2)
        # Current shape: (T, B*C, H*2)
        # Reshape to (T, B, C, H*2)
        lstm_out_reshaped = lstm_out.reshape(T, B, C, self.lstm_hidden_size * 2)
        # Permute to (B, T, C, H*2)
        mask_fc_input = lstm_out_reshaped.permute(1, 0, 2, 3)

        # mask_logits: (B, T, C, F)
        mask_logits = self.fc_mask(mask_fc_input)
        # M (mask): (B, T, C, F)
        M = self.sigmoid(mask_logits)

        # 4. Final Feature Calculation
        # filtered_features_intermediate: (B, T, C, F)
        # Essentially
        filtered_features_output = (M * converted_features) + converted_features
        
        return filtered_features_output

# Example usage (for testing purposes, if you were to implement it):
if __name__ == '__main__':
    batch_size = 4
    time_steps = 100     # Example: T (sequence length after feature extraction)
    input_channels = 12  # Example: C from WavLM (e.g., 12 layers)
    feature_size = 768   # Example: F from WavLM (e.g., 768 features per layer)

    # Create a dummy filter block
    speaker_filter = SpeakerFilterBlock(input_channels=input_channels, feature_size=feature_size)

    # Create dummy input tensors
    dummy_converted_features = torch.randn(batch_size, time_steps, input_channels, feature_size)
    dummy_reference_features = torch.randn(batch_size, time_steps, input_channels, feature_size)

    try:
        filtered_output = speaker_filter(dummy_converted_features, dummy_reference_features)
        print(f"Input converted_features shape: {dummy_converted_features.shape}")
        print(f"Input reference_features shape: {dummy_reference_features.shape}")
        print(f"Output filtered_features shape: {filtered_output.shape}")
        # Expected output shape: (batch_size, time_steps, input_channels, feature_size)
        # e.g., (4, 100, 12, 768)
    except Exception as e:
        print(f"Error during forward pass: {e}")
        import traceback
        traceback.print_exc()