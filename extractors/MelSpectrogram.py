import torchaudio
import torch

class MelSpectrogram():
    def __init__(self, n_fft=2048, hop_length=512, n_mels=128, device="cuda" if torch.cuda.is_available() else "cpu"):
        super(MelSpectrogram, self).__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.feature_size = n_mels
        self.device = device
        self.transform = torchaudio.transforms.MelSpectrogram(n_fft=self.n_fft, hop_length=self.hop_length, n_mels=self.feature_size).to(device)

    def extract_features(self, x):
        x = self.transform(x)
        x = torch.permute(x, (0, 2, 1)) # (batch_size, n_time, n_mels)
        return x