"""Factory for vision/audio/text transforms."""

from __future__ import annotations

from typing import Callable

import torch
import torchaudio
import torchvision.transforms as T


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
