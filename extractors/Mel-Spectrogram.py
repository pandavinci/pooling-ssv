import torchaudio
import torch.nn as nn

class MelSpectrogram(nn.Module):
    def __init__(self, n_fft=2048, hop_length=512, n_mels=128):
        super(MelSpectrogram, self).__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.feature_size = n_mels

    def extract_features(self, x):
        return torchaudio.transforms.MelSpectrogram(n_fft=self.n_fft, hop_length=self.hop_length, n_mels=self.feature_size)(x)