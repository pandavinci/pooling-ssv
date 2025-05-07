import torch

class BaseFilter():
    def forward(self, converted_features: torch.Tensor, reference_features: torch.Tensor) -> torch.Tensor:
        """
        Abstract method for filtering converted features using reference features.

        Args:
            converted_features (torch.Tensor): Features extracted from the converted audio.
                                               Expected shape: (batch_size, channels, time_steps).
            reference_features (torch.Tensor): Features extracted from the reference audio.
                                               Expected shape: (batch_size, channels, time_steps).

        Returns:
            torch.Tensor: The filtered features. The shape might differ from the input,
                          e.g., (batch_size, output_channels, time_steps).
        """
        pass 