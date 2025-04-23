import torch
import torchaudio.transforms as T
import audiomentations as AA

class CodecAugmentations:
    """
    Class for codec augmentations.
    Currently supports mu-law compression and MP3 compression.
    """

    def __init__(self, sample_rate: int = 16000, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.sample_rate = sample_rate
        self.mu_encoder = T.MuLawEncoding().to(self.device)
        self.mu_decoder = T.MuLawDecoding().to(self.device)
        self.mp3_compression = AA.Mp3Compression(p=1.0)

    def mu_law(
        self,
        waveform: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply mu-law compression to the audio waveform.

        param waveform: The audio waveform to apply mu-law compression to.

        return: The audio waveform with mu-law compression and decompression applied.
        """
        waveform = waveform.to(self.device)
        enc = self.mu_encoder(waveform)
        dec = self.mu_decoder(enc)
        return dec

    def mp3(
        self,
        waveform: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply MP3 compression to the audio waveform.

        param waveform: The audio waveform to apply MP3 compression to.

        return: The audio waveform with MP3 compression applied.
        """
        # Convert tensor to numpy array
        waveform_np = waveform.to(self.device).numpy()
        
        # Apply MP3 compression
        augmented_waveform = self.mp3_compression(
            samples=waveform_np,
            sample_rate=self.sample_rate
        )
        
        # Convert back to tensor
        return torch.tensor(augmented_waveform, device=self.device)

def main():
    import torchaudio

    # Load the test audio file
    waveform, sample_rate = torchaudio.load('augmentation/test.wav')
    print(waveform.shape)
    # Initialize the Codec class
    codec = CodecAugmentations(sample_rate=sample_rate)

    # Test mu-law compression
    mu_law_waveform = codec.mu_law(waveform)
    print("Mu-law compression applied.")
    print(mu_law_waveform.shape)
    # Test MP3 compression
    mp3_waveform = codec.mp3(waveform)
    print("MP3 compression applied.")
    print(mp3_waveform.shape)

    # Save the results for verification
    torchaudio.save('augmentation/test_mu_law.wav', mu_law_waveform, sample_rate)
    torchaudio.save('augmentation/test.mp3', mp3_waveform, sample_rate)
    print("Compressed files saved.")

if __name__ == "__main__":
    main()
