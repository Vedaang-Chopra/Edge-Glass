"""Factory for vision/audio/text transforms."""

from __future__ import annotations

import torch
import torchaudio
import torchvision.transforms as T
from torchvision.transforms import functional as F


def build_transforms(image_size: int = 224, sample_rate: int = 16000):
    vision = T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.ConvertImageDtype(torch.float32),
            T.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ]
    )
    mel_spec = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_mels=128)

    def audio_loader(path: str):
        waveform, sr = torchaudio.load(path)
        if sr != sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
        return waveform

    def audio_transform(waveform):
        return mel_spec(waveform)

    return {"vision": vision, "audio": audio_transform, "audio_loader": audio_loader}


def get_image_transforms(image_size: int = 224, is_training: bool = True):
    """Return image transforms for training or eval."""
    if is_training:
        return T.Compose(
            [
                T.Resize((image_size + 32, image_size + 32)),
                T.RandomCrop((image_size, image_size)),
                T.RandomHorizontalFlip(),
                T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                T.ConvertImageDtype(torch.float32),
                T.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )
    else:
        return T.Compose(
            [
                T.Resize((image_size, image_size)),
                T.ConvertImageDtype(torch.float32),
                T.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )
