"""
encoders.py

Frozen vision & audio encoder wrappers for alignment experiments.

Usage (in a notebook):
    from encoders import VisionEncoder, AudioEncoder
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vision_enc = VisionEncoder(
        model_name="facebook/dinov2-base",
        device=device,
        dtype=torch.float16,
    )

    audio_enc = AudioEncoder(
        model_name="openai/whisper-base",
        device=device,
        dtype=torch.float16,
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import numpy as np

import torch
import torch.nn as nn

from transformers import (
    AutoImageProcessor,
    AutoModel,
    WhisperProcessor,
    WhisperModel,
)

# Optional: only needed if you want to pass raw PIL images
try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None  # type: ignore


Tensor = torch.Tensor


# ---------------------------------------------------------------------------
# Utility: simple to_device helper
# ---------------------------------------------------------------------------

def move_to_device(x: Tensor, device: torch.device, dtype: Optional[torch.dtype] = None) -> Tensor:
    if dtype is not None:
        return x.to(device=device, dtype=dtype)
    return x.to(device=device)


# ---------------------------------------------------------------------------
# Vision Encoder
# ---------------------------------------------------------------------------

@dataclass
class VisionEncoderConfig:
    model_name: str = "facebook/dinov2-base"
    """
    Any HF vision transformer model that supports:
      - AutoImageProcessor
      - AutoModel(..., output_hidden_states=True)
    """

    use_hidden_layers: Tuple[int, int] = (1, -2)
    """
    Indices of hidden states to concatenate.
    For a ViT, (1, -2) means: 2nd layer and 2nd-to-last layer.
    """

    remove_cls_token: bool = True
    """
    If True, drops the first token (CLS) and keeps only patch tokens.
    """

    dtype: torch.dtype = torch.float16


class VisionEncoder(nn.Module):
    """
    Wrapper around a frozen HF vision encoder (e.g. DINOv2).

    Provides:
        encode_images(images) -> dict with:
            {
                "feats": (B, T_patches, D_feat),
                "mask":  (B, T_patches) boolean
            }
    """

    def __init__(
        self,
        model_name: str = "facebook/dinov2-base",
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float16,
        use_hidden_layers: Tuple[int, int] = (1, -2),
        remove_cls_token: bool = True,
    ) -> None:
        super().__init__()

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.cfg = VisionEncoderConfig(
            model_name=model_name,
            use_hidden_layers=use_hidden_layers,
            remove_cls_token=remove_cls_token,
            dtype=dtype,
        )

        self.device = device

        # Load processor + model
        self.processor = AutoImageProcessor.from_pretrained(self.cfg.model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(
            self.cfg.model_name,
            output_hidden_states=True,
        )

        # Move & freeze
        self.model.to(self.device, dtype=self.cfg.dtype)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        # Will be set after first forward
        self._feat_dim: Optional[int] = None

    @property
    def feat_dim(self) -> int:
        """Dimension of the returned patch features (after concatenation)."""
        if self._feat_dim is None:
            raise RuntimeError("feat_dim is not set yet — run encode_images once.")
        return self._feat_dim

    @torch.no_grad()
    def encode_images(
        self,
        images: Union[List["Image.Image"], Tensor],
        return_mask: bool = True,
    ) -> dict:
        """
        Encode a batch of images into patch-level features.

        Args:
            images: list of PIL.Image or preprocessed pixel_values tensor
                    If list of PIL, processor will be applied automatically.
                    If Tensor, assumed shape is (B, C, H, W) in model's expected format.
            return_mask: whether to return a boolean mask.

        Returns:
            dict with:
                feats: Tensor (B, T_patches, D_feat)
                mask:  Tensor (B, T_patches) bool
        """
        # Preprocess
        if isinstance(images, list):
            if Image is None:
                raise ImportError("PIL is required to pass raw images to VisionEncoder.")
            inputs = self.processor(
                images=images,
                return_tensors="pt",
            )
            pixel_values = inputs["pixel_values"]  # (B, C, H, W)
        else:
            # already preprocessed tensor
            pixel_values = images

        pixel_values = move_to_device(pixel_values, self.device, self.cfg.dtype)

        # Forward through the model
        outputs = self.model(pixel_values=pixel_values)
        hidden_states = outputs.hidden_states  # tuple of (B, T, D)

        h_idx1, h_idx2 = self.cfg.use_hidden_layers
        h1 = hidden_states[h_idx1]  # (B, T, D)
        h2 = hidden_states[h_idx2]  # (B, T, D)

        # Remove CLS token if requested
        if self.cfg.remove_cls_token:
            h1 = h1[:, 1:, :]  # (B, T_patches, D)
            h2 = h2[:, 1:, :]  # (B, T_patches, D)

        feats = torch.cat([h1, h2], dim=-1)  # (B, T_patches, 2D)

        if self._feat_dim is None:
            self._feat_dim = feats.size(-1)

        if not return_mask:
            return {"feats": feats}

        B, T, _ = feats.shape
        mask = torch.ones(B, T, dtype=torch.bool, device=self.device)

        return {
            "feats": feats,  # (B, T_patches, D_feat)
            "mask": mask,    # (B, T_patches)
        }


# ---------------------------------------------------------------------------
# Audio Encoder (Whisper)
# ---------------------------------------------------------------------------

@dataclass
class AudioEncoderConfig:
    model_name: str = "openai/whisper-base"
    dtype: torch.dtype = torch.float16
    target_sampling_rate: int = 16000


class AudioEncoder(nn.Module):
    """
    Wrapper around a frozen Whisper encoder used as a generic audio encoder.

    Provides:
        encode_waveforms(wavs, sample_rates) -> dict with:
            {
                "feats": (B, T_audio, D_audio),
                "mask":  (B, T_audio) boolean
            }

    Note: This uses WhisperModel (encoder part only) and WhisperProcessor.
    """

    def __init__(
        self,
        model_name: str = "openai/whisper-base",
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float16,
        target_sampling_rate: int = 16000,
    ) -> None:
        super().__init__()

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.cfg = AudioEncoderConfig(
            model_name=model_name,
            dtype=dtype,
            target_sampling_rate=target_sampling_rate,
        )

        self.device = device

        # Load processor + model
        self.processor = WhisperProcessor.from_pretrained(self.cfg.model_name, use_fast=True)
        self.model = WhisperModel.from_pretrained(self.cfg.model_name)

        # Move & freeze
        self.model.to(self.device, dtype=self.cfg.dtype)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        self._feat_dim: Optional[int] = None

    @property
    def feat_dim(self) -> int:
        """Dimension of the returned audio frame features."""
        if self._feat_dim is None:
            raise RuntimeError("feat_dim is not set yet — run encode_waveforms once.")
        return self._feat_dim

    @torch.no_grad()
    def encode_waveforms(
        self,
        waveforms: Union[Tensor, List[Tensor]],
        sample_rates: Union[int, List[int]],
        return_mask: bool = True,
    ) -> dict:
        """
        Encode raw audio waveforms into frame-level features.

        Args:
            waveforms:
                - Tensor of shape (B, T) or (B, 1, T)
                - List of 1D or multi-D Tensors / numpy arrays (we flatten to 1D)
            sample_rates:
                - Single int if all waveforms share the same sampling rate
                - List[int] of length B otherwise. (Currently ignored, we resample to cfg.target_sampling_rate.)

        Returns:
            dict with:
                feats: (B, T_audio, D_audio)
                mask:  (B, T_audio) boolean
        """
        # ---- Normalize inputs to list-of-1D float32 numpy arrays ----
        if isinstance(waveforms, torch.Tensor):
            # Ensure batch dim exists
            if waveforms.ndim == 1:
                waveforms = waveforms.unsqueeze(0)  # (1, T)

            if waveforms.ndim == 2:      # (B, T)
                wav_list = [
                    w.detach().cpu().numpy().astype("float32").reshape(-1)
                    for w in waveforms
                ]
            elif waveforms.ndim == 3:    # (B, C, T) -> flatten all dims to 1D
                wav_list = [
                    w.detach().cpu().numpy().astype("float32").reshape(-1)
                    for w in waveforms
                ]
            else:
                raise ValueError(f"Unexpected waveform tensor shape: {waveforms.shape}")
        else:
            # list[Tensor] or list[np.ndarray]
            wav_list = []
            for w in waveforms:
                if isinstance(w, torch.Tensor):
                    arr = w.detach().cpu().numpy().astype("float32")
                else:
                    arr = np.asarray(w, dtype="float32")
                wav_list.append(arr.reshape(-1))  # ensure 1D

        # (Optional) we could respect per-sample sample_rates, but for now
        # WhisperProcessor will resample to self.cfg.target_sampling_rate.
        if isinstance(sample_rates, int):
            sr = sample_rates
        else:
            # if a list is given, we just pick the first; Whisper will resample anyway
            sr = sample_rates[0]

        # ---- WhisperProcessor: raw audio -> log-Mel features ----
        inputs = self.processor(
            wav_list,
            sampling_rate=sr,
            return_tensors="pt",
        )
        input_features = inputs["input_features"]  # (B, n_frames, feat_dim)

        input_features = move_to_device(input_features, self.device, self.cfg.dtype)

        # ---- Forward through Whisper encoder (encoder-only path) ----
        outputs = self.model.encoder(
            input_features=input_features,
        )
        feats = outputs.last_hidden_state  # (B, T_audio, D_audio)

        if self._feat_dim is None:
            self._feat_dim = feats.size(-1)

        if not return_mask:
            return {"feats": feats}

        B, T, _ = feats.shape
        mask = torch.ones(B, T, dtype=torch.bool, device=self.device)

        return {
            "feats": feats,  # (B, T_audio, D_audio)
            "mask": mask,    # (B, T_audio)
        }


__all__ = [
    "VisionEncoder",
    "VisionEncoderConfig",
    "AudioEncoder",
    "AudioEncoderConfig",
    "move_to_device",
]
