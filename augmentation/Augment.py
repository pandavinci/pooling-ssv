from math import floor
import torch
import random

from augmentation.Codec import CodecAugmentations
from augmentation.General import GeneralAugmentations
from augmentation.NoiseFilter import NoiseFilterAugmentations
from augmentation.RawBoost import process_Rawboost_feature
from augmentation.RIR import RIRAugmentations

from torch.profiler import record_function


class Augmentor:
    """
    Class to define the waveform augmentation pipeline.
    """

    def __init__(self, rir_root, device="cpu"):
        self.device = device

        self.Codec = CodecAugmentations(device=self.device)
        self.General = GeneralAugmentations(device=self.device)
        self.NoiseFilter = NoiseFilterAugmentations()  # is cpu only as of now
        self.RIR = RIRAugmentations(rir_root, device=self.device)

    def augment(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        The waveform augmentation pipeline.
        """

        with torch.no_grad(): # No need to compute gradients for augmentation
            # Using augmentations according to this paper: https://www.isca-archive.org/asvspoof_2024/xu24_asvspoof.pdf
            trim_starting_silence: bool = random.random() < 0.5  # 50% chance of removing the starting silence
            apply_timemask: bool = random.random() < 0.3  # 30% chance of applying time-mask
            apply_rir: bool = random.random() < 0.3  # 30% chance of applying RIR augmentations
            apply_mu_law: bool = random.random() < 0.3  # 30% chance of applying mu-law enc-dec augmentation
            apply_LnL_ISD: bool = random.random() < 0.3  # 30% chance of applying RawBoost augmentations
            apply_noise_filter: bool = random.random() < 0.3  # 30% chance of applying noise augmentations

            waveform = waveform.squeeze()
            waveform = waveform.to(self.device)

            if trim_starting_silence:  # 50% chance of removing the starting silence
                trimmed_waveform = self.General.trim_starting_silence(waveform)
                # print("Trimmed starting silence")
                # Use the trimmed waveform only if it is not empty
                if len(trimmed_waveform) != 0:
                    waveform = trimmed_waveform
                    del trimmed_waveform  # Clean up intermediate tensor

            if apply_timemask:
                wf_len = len(waveform)
                time_mask_duration = random.uniform(floor(wf_len * 0.2), floor(wf_len * 0.5))
                time_mask_start = random.uniform(0, wf_len - time_mask_duration)
                waveform = self.General.mask_time(
                    waveform, mask_time=(time_mask_start, time_mask_start + time_mask_duration)
                )
                # print(f"Applied time mask {time_mask_start} to {time_mask_start + time_mask_duration}")

            if apply_rir: # TODO: separate RIR and noise augmentations
                rir_intesity = random.uniform(0.2, 0.8)
                waveform = self.RIR.apply_rir(waveform, scale_factor=rir_intesity)
                # print(f"Applied RIR with intensity {rir_intesity}")

            if apply_mu_law:
                waveform = self.Codec.mu_law(waveform)
                # print("Applied mu-law enc-dec")

            if apply_LnL_ISD:
                # Convert to CPU only when needed for RawBoost
                waveform_cpu = waveform.squeeze().cpu().numpy()
                waveform = process_Rawboost_feature(waveform_cpu, 16000, algo=5)
                del waveform_cpu  # Clean up CPU tensor
                # print("Applied RawBoost: LnL-ISD")

            if apply_noise_filter:
                waveform = self.NoiseFilter.apply_noise_filter(waveform)
                # print("Applied noise filter")

            waveform = waveform.unsqueeze(0).cpu().detach()
            # GPU memory is released in the collate function in dataloader
            return waveform
