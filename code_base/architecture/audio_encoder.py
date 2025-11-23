"""
audio_encoder.py

OOP wrapper for an audio encoder (Whisper-style).
"""

from typing import Optional, Union

import numpy as np
import torch
import torchaudio
from transformers import WhisperProcessor, WhisperModel

from .base import ModelConfig, BaseEncoder


class AudioEncoder(BaseEncoder):
    """
    Generic audio encoder using Whisper as an encoder.

    Usage:
        cfg = ModelConfig("openai/whisper-base")
        enc = AudioEncoder(cfg)
        emb = enc.encode_waveform(waveform, sr=16000)
    """

    def __init__(self, cfg: ModelConfig, target_sr: int = 16000):
        super().__init__(cfg)

        self.processor: WhisperProcessor = WhisperProcessor.from_pretrained(
            cfg.model_name
        )
        self.model: WhisperModel = WhisperModel.from_pretrained(
            cfg.model_name, torch_dtype=cfg.dtype
        ).to(self.device)

        # Enable hidden states for analysis / future MRL
        if hasattr(self.model.config, "output_hidden_states"):
            self.model.config.output_hidden_states = True

        # Freeze (Phase 1: frozen encoder)
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

        self.target_sr = target_sr

    @torch.no_grad()
    def encode_waveform(
        self,
        waveform: Union[torch.Tensor, np.ndarray],
        sr: int,
    ) -> torch.Tensor:
        """
        Encode a single waveform into an embedding.

        waveform: np.ndarray or torch.Tensor of shape (T,) or (1, T)
        sr: sampling rate of waveform
        Returns: (1, D) embedding (mean-pooled last_hidden_state)
        """
        if isinstance(waveform, np.ndarray):
            waveform = torch.from_numpy(waveform)

        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)  # (1, T)

        if sr != self.target_sr:
            waveform = torchaudio.functional.resample(
                waveform, orig_freq=sr, new_freq=self.target_sr
            )

        inputs = self.processor(
            waveform.squeeze(0).numpy(),
            sampling_rate=self.target_sr,
            return_tensors="pt",
        ).to(self.device)

        outputs = self.model(**inputs)
        hidden = outputs.last_hidden_state  # (B, T, D)
        audio_emb = hidden.mean(dim=1)      # (B, D)
        return audio_emb

    def encode(self, waveform, sr: int) -> torch.Tensor:
        return self.encode_waveform(waveform, sr)


def load_audio_encoder(
    model_name: str,
    device: Optional[str] = None,
    dtype: torch.dtype = torch.float16,
    target_sr: int = 16000,
) -> AudioEncoder:
    """
    Convenience loader for scripts / notebooks.
    """
    cfg = ModelConfig(model_name=model_name, device=device, dtype=dtype)
    return AudioEncoder(cfg, target_sr=target_sr)
