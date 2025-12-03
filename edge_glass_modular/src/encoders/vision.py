"""Vision encoder module using CLIP/ViT."""

import torch
import torch.nn as nn
from transformers import CLIPVisionModel, CLIPImageProcessor
from dataclasses import dataclass
from typing import Optional

from .perceiver import PerceiverResampler
from .mrl import MatryoshkaProjection
from .pooling import AttentionPooling, SimpleAttentionPooling


@dataclass
class VisionEncoderOutput:
    """Output from vision encoder."""

    pooled: torch.Tensor  # (batch_size, projection_dim)
    sequence: Optional[torch.Tensor] = None  # (batch_size, seq_len, projection_dim)
    mrl_embeddings: Optional[dict] = None  # {dim: embedding} for MRL


class VisionEncoder(nn.Module):
    """Vision encoder with optional Perceiver and MRL.

    Architecture:
        Image (B, 3, 224, 224)
          ↓
        CLIP Vision Encoder (frozen)
          ↓ (B, num_patches, hidden_dim)
        Linear Projection
          ↓ (B, num_patches, projection_dim)
        [Optional] Perceiver Resampler
          ↓ (B, num_latents, perceiver_dim)
        [Optional] MRL Projection
          ↓ (B, projection_dim) + MRL outputs

    Args:
        model_name: HuggingFace model name (e.g., 'openai/clip-vit-large-patch14')
        projection_dim: Output dimension for projected embeddings
        freeze: Whether to freeze the base encoder
        use_perceiver: Whether to use Perceiver resampler
        perceiver_num_latents: Number of latent query tokens for Perceiver
        perceiver_latent_dim: Dimension of Perceiver latents
        perceiver_num_layers: Number of Perceiver layers
        perceiver_num_heads: Number of attention heads in Perceiver
        use_mrl: Whether to use Matryoshka Representation Learning
        mrl_dimensions: List of MRL dimensions (e.g., [512, 256, 128])
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-large-patch14",
        projection_dim: int = 4096,  # Updated to 4096 for top MRL dim
        freeze: bool = True,
        use_perceiver: bool = False,
        perceiver_num_latents: int = 64,
        perceiver_latent_dim: int = 512,
        perceiver_num_layers: int = 3,
        perceiver_num_heads: int = 8,
        use_mrl: bool = False,
        mrl_dimensions: list = None,
        use_attention_pooling: bool = True,  # New: use learnable attention pooling
        pooling_type: str = "simple",  # "simple" or "multihead"
    ):
        super().__init__()

        self.model_name = model_name
        self.projection_dim = projection_dim
        self.use_perceiver = use_perceiver
        self.use_mrl = use_mrl
        self.use_attention_pooling = use_attention_pooling

        # Load CLIP vision encoder
        self.encoder = CLIPVisionModel.from_pretrained(model_name)
        self.processor = CLIPImageProcessor.from_pretrained(model_name)

        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Get hidden dimension from encoder
        self.hidden_dim = self.encoder.config.hidden_size

        # Projection layer
        if use_perceiver:
            # Project to perceiver latent dim first
            self.projector = nn.Linear(self.hidden_dim, perceiver_latent_dim)
            current_dim = perceiver_latent_dim

            # Perceiver resampler
            self.perceiver = PerceiverResampler(
                dim=perceiver_latent_dim,
                num_latents=perceiver_num_latents,
                num_layers=perceiver_num_layers,
                num_heads=perceiver_num_heads,
            )

            # Final projection from perceiver to output dim
            self.final_projector = nn.Linear(perceiver_latent_dim, projection_dim)
        else:
            # Direct projection
            self.projector = nn.Linear(self.hidden_dim, projection_dim)
            self.perceiver = None
            self.final_projector = None
            current_dim = projection_dim

        # Learnable attention pooling (replaces mean/CLS pooling)
        if use_attention_pooling and not use_perceiver:
            if pooling_type == "multihead":
                self.attention_pool = AttentionPooling(
                    input_dim=projection_dim,
                    num_queries=1,
                    num_heads=8,
                    dropout=0.1,
                )
            else:  # simple
                self.attention_pool = SimpleAttentionPooling(
                    input_dim=projection_dim,
                    dropout=0.1,
                )
        else:
            self.attention_pool = None

        # MRL projection with updated dimensions for 4096
        if use_mrl:
            if mrl_dimensions is None:
                # Updated MRL dimensions: 4096 (top) -> 2048 -> 1024 -> 512 -> 256 -> 128
                mrl_dimensions = [2048, 1024, 512, 256, 128]
            self.mrl = MatryoshkaProjection(
                input_dim=projection_dim, mrl_dimensions=mrl_dimensions
            )
        else:
            self.mrl = None

    def forward(
        self, images: torch.Tensor, return_sequence: bool = False
    ) -> VisionEncoderOutput:
        """Encode images to embeddings.

        Args:
            images: Input images (B, 3, H, W)
            return_sequence: Whether to return sequence embeddings

        Returns:
            VisionEncoderOutput with pooled and optional sequence embeddings
        """
        # Encode with CLIP
        outputs = self.encoder(pixel_values=images, output_hidden_states=True)

        # Get last hidden state (B, num_patches + 1, hidden_dim)
        # Note: includes CLS token at position 0
        hidden_states = outputs.last_hidden_state

        # Project
        projected = self.projector(hidden_states)  # (B, num_patches + 1, proj_dim or latent_dim)

        # Apply Perceiver if enabled
        if self.use_perceiver:
            # Remove CLS token for perceiver input
            sequence = projected[:, 1:, :]  # (B, num_patches, latent_dim)
            latents = self.perceiver(sequence)  # (B, num_latents, latent_dim)

            # Project to final dimension
            latents = self.final_projector(latents)  # (B, num_latents, projection_dim)

            # Pool latents (mean pooling for perceiver)
            pooled = latents.mean(dim=1)  # (B, projection_dim)
            sequence_output = latents if return_sequence else None

        else:
            # Use attention pooling or CLS token
            if self.attention_pool is not None:
                # Learnable attention pooling over all tokens (including CLS)
                pooled = self.attention_pool(projected)  # (B, projection_dim)
                sequence_output = projected if return_sequence else None
            else:
                # Fallback to CLS token as pooled representation
                pooled = projected[:, 0, :]  # (B, projection_dim)
                sequence_output = projected[:, 1:, :] if return_sequence else None

        # L2 normalize pooled embedding BEFORE MRL
        pooled = nn.functional.normalize(pooled, p=2, dim=-1)

        # Apply MRL if enabled (MRL expects normalized input)
        mrl_embeddings = None
        if self.use_mrl:
            mrl_embeddings = self.mrl(pooled)

        return VisionEncoderOutput(
            pooled=pooled, sequence=sequence_output, mrl_embeddings=mrl_embeddings
        )

    def preprocess(self, images):
        """Preprocess images using CLIP processor.

        Args:
            images: List of PIL Images or numpy arrays

        Returns:
            Preprocessed tensor (B, 3, 224, 224)
        """
        return self.processor(images=images, return_tensors="pt")["pixel_values"]

    @property
    def num_parameters(self) -> int:
        """Return number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def num_total_parameters(self) -> int:
        """Return total number of parameters."""
        return sum(p.numel() for p in self.parameters())
