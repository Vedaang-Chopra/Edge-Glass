"""Audio encoder module using Whisper."""

import torch
import torch.nn as nn
from transformers import WhisperModel, WhisperFeatureExtractor
from dataclasses import dataclass
from typing import Optional

from .perceiver import PerceiverResampler
from .mrl import MatryoshkaProjection


@dataclass
class AudioEncoderOutput:
    """Output from audio encoder."""

    pooled: torch.Tensor  # (batch_size, projection_dim)
    sequence: Optional[torch.Tensor] = None  # (batch_size, seq_len, projection_dim)
    mrl_embeddings: Optional[dict] = None  # {dim: embedding} for MRL


class AudioEncoder(nn.Module):
    """Audio encoder with optional Perceiver and MRL.

    Architecture:
        Audio waveform (B, samples)
          ↓
        Whisper Encoder (frozen)
          ↓ (B, time_steps, hidden_dim)
        Linear Projection
          ↓ (B, time_steps, projection_dim)
        [Optional] Perceiver Resampler
          ↓ (B, num_latents, perceiver_dim)
        [Optional] MRL Projection
          ↓ (B, projection_dim) + MRL outputs

    Args:
        model_name: HuggingFace model name (e.g., 'openai/whisper-large-v3')
        projection_dim: Output dimension for projected embeddings
        freeze: Whether to freeze the base encoder
        use_perceiver: Whether to use Perceiver resampler
        perceiver_num_latents: Number of latent query tokens for Perceiver
        perceiver_latent_dim: Dimension of Perceiver latents
        perceiver_num_layers: Number of Perceiver layers
        perceiver_num_heads: Number of attention heads in Perceiver
        use_mrl: Whether to use Matryoshka Representation Learning
        mrl_dimensions: List of MRL dimensions
    """

    def __init__(
        self,
        model_name: str = "openai/whisper-large-v3",
        projection_dim: int = 1024,
        freeze: bool = True,
        use_perceiver: bool = False,
        perceiver_num_latents: int = 64,
        perceiver_latent_dim: int = 512,
        perceiver_num_layers: int = 3,
        perceiver_num_heads: int = 8,
        use_mrl: bool = False,
        mrl_dimensions: list = None,
    ):
        super().__init__()

        self.model_name = model_name
        self.projection_dim = projection_dim
        self.use_perceiver = use_perceiver
        self.use_mrl = use_mrl

        # Load Whisper encoder
        self.encoder = WhisperModel.from_pretrained(model_name).encoder
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)

        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Get hidden dimension
        self.hidden_dim = self.encoder.config.d_model

        # Projection layer
        if use_perceiver:
            self.projector = nn.Linear(self.hidden_dim, perceiver_latent_dim)

            # Perceiver resampler
            self.perceiver = PerceiverResampler(
                dim=perceiver_latent_dim,
                num_latents=perceiver_num_latents,
                num_layers=perceiver_num_layers,
                num_heads=perceiver_num_heads,
            )

            # Final projection
            self.final_projector = nn.Linear(perceiver_latent_dim, projection_dim)
        else:
            self.projector = nn.Linear(self.hidden_dim, projection_dim)
            self.perceiver = None
            self.final_projector = None

        # MRL projection
        if use_mrl:
            if mrl_dimensions is None:
                mrl_dimensions = [512, 256, 128]
            self.mrl = MatryoshkaProjection(
                input_dim=projection_dim, mrl_dimensions=mrl_dimensions
            )
        else:
            self.mrl = None

    def forward(
        self,
        input_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_sequence: bool = False,
    ) -> AudioEncoderOutput:
        """Encode audio to embeddings.

        Args:
            input_features: Preprocessed audio features (B, n_mels, time)
            attention_mask: Attention mask for variable-length audio
            return_sequence: Whether to return sequence embeddings

        Returns:
            AudioEncoderOutput with pooled and optional sequence embeddings
        """
        # Encode with Whisper
        outputs = self.encoder(
            input_features=input_features,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        # Get last hidden state (B, time_steps, hidden_dim)
        hidden_states = outputs.last_hidden_state

        # Project
        projected = self.projector(hidden_states)

        # Apply Perceiver if enabled
        if self.use_perceiver:
            latents = self.perceiver(projected)  # (B, num_latents, latent_dim)
            latents = self.final_projector(latents)  # (B, num_latents, projection_dim)

            # Pool latents
            pooled = latents.mean(dim=1)  # (B, projection_dim)
            sequence_output = latents if return_sequence else None

        else:
            # Mean pooling over time dimension
            if attention_mask is not None:
                # Mask-aware pooling
                mask_expanded = attention_mask.unsqueeze(-1).expand(projected.size())
                sum_embeddings = torch.sum(projected * mask_expanded, dim=1)
                sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
                pooled = sum_embeddings / sum_mask
            else:
                pooled = projected.mean(dim=1)

            sequence_output = projected if return_sequence else None

        # Apply MRL if enabled
        mrl_embeddings = None
        if self.use_mrl:
            mrl_embeddings = self.mrl(pooled)

        # L2 normalize
        pooled = nn.functional.normalize(pooled, p=2, dim=-1)

        return AudioEncoderOutput(
            pooled=pooled, sequence=sequence_output, mrl_embeddings=mrl_embeddings
        )

    def preprocess(self, audio_arrays, sampling_rate=16000):
        """Preprocess audio using Whisper feature extractor.

        Args:
            audio_arrays: List of audio waveforms
            sampling_rate: Sampling rate of audio

        Returns:
            Preprocessed features and attention mask
        """
        features = self.feature_extractor(
            audio_arrays, sampling_rate=sampling_rate, return_tensors="pt", padding=True
        )
        return features["input_features"], features.get("attention_mask")

    @property
    def num_parameters(self) -> int:
        """Return number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def num_total_parameters(self) -> int:
        """Return total number of parameters."""
        return sum(p.numel() for p in self.parameters())
