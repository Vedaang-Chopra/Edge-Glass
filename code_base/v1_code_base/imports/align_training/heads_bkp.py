"""
alignment_heads.py

Feature-level alignment modules built on top of:
- PerceiverLatentEncoder
- ProjectorMLP

These heads take pre-extracted sequence features + masks from any encoder
(e.g., DINO for vision, Whisper for audio) and map them into a shared
alignment space (d_align), optionally via a Perceiver latent bottleneck.

Intended usage in Stage-1 alignment:

    from imports.alignment_heads import FeatureAlignmentHead, TextAlignmentHead
    from imports.losses import matryoshka_contrastive_loss

    vision_head = FeatureAlignmentHead(d_feat=D_FEAT_V, ...)
    audio_head  = FeatureAlignmentHead(d_feat=D_FEAT_A, ...)
    text_head   = TextAlignmentHead(d_text=D_TEXT, d_align=D_ALIGN)

    z_img = vision_head(feats_img, mask_img)["pooled"]   # (B, d_align)
    z_txt = text_head(text_embs)                         # (B, d_align)

    loss = matryoshka_contrastive_loss(z_img, z_txt, radii=[256, 512, 1024])
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from imports.perceiver import PerceiverLatentEncoder, ProjectorMLP


Tensor = torch.Tensor


# ============================================================
# Masked mean pooling
# ============================================================

def masked_mean(
    x: Tensor,        # (B, T, D)
    mask: Tensor,     # (B, T) bool, True = valid, False = pad
    eps: float = 1e-6,
) -> Tensor:
    """
    Compute mean over time dimension with a boolean mask.

    Args:
        x:    (B, T, D)
        mask: (B, T) bool, True where there is real data.

    Returns:
        pooled: (B, D)
    """
    if mask.dtype != torch.float32 and mask.dtype != torch.float16 and mask.dtype != torch.bfloat16:
        mask_f = mask.float()
    else:
        mask_f = mask

    # (B, T, 1)
    mask_f = mask_f.unsqueeze(-1)

    summed = (x * mask_f).sum(dim=1)  # (B, D)
    denom = mask_f.sum(dim=1).clamp_min(eps)  # (B, 1)

    return summed / denom


# ============================================================
# Config container (optional, but handy)
# ============================================================

@dataclass
class AlignmentHeadConfig:
    """
    Small helper dataclass for constructing FeatureAlignmentHead.

    Typically you tie these to cfg.architecture from config.py:
        d_shared       ← cfg.architecture.perceiver_dim or a chosen value
        d_latent       ← same as d_shared or separate
        d_align        ← global alignment dim (MRL dim)
        num_latents    ← cfg.architecture.num_latents
        num_layers     ← cfg.architecture.num_perceiver_layers
        num_heads      ← cfg.architecture.num_attn_heads
        mlp_ratio      ← cfg.architecture.mlp_ratio
    """
    d_shared: int = 512
    d_latent: int = 512
    d_align: int = 1024
    num_latents: int = 64
    num_layers: int = 4
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    use_perceiver: bool = True


# ============================================================
# Feature Alignment Head (for vision OR audio)
# ============================================================

class FeatureAlignmentHead(nn.Module):
    """
    Feature-level alignment head for a *single* modality.

    Expected inputs:
        feats: (B, T, D_feat)
        mask:  (B, T) bool (True = valid token)

    Pipeline:
        1) Linear adapter: D_feat → d_shared
        2) If use_perceiver:
             PerceiverLatentEncoder: (B, L, d_latent)
             ProjectorMLP:          (B, L, d_align)
             pooled = mean over L   → (B, d_align)
           Else:
             pooled_in = masked_mean(feats_sh, mask)  # (B, d_shared)
             ProjectorMLP: pooled_in → (B, d_align)

    Outputs (dict):
        {
          "tokens": (B, L, d_align) or None if use_perceiver=False,
          "pooled": (B, d_align),
        }
    """

    def __init__(
        self,
        d_feat: int,                # input feature dim from encoder
        cfg: AlignmentHeadConfig,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
    ):
        super().__init__()

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.d_feat = d_feat
        self.cfg = cfg
        self.dtype = dtype
        self.device = device

        # Modality-specific adapter into shared Perceiver space
        self.adapter = nn.Linear(d_feat, cfg.d_shared).to(device, dtype=dtype)

        self.use_perceiver = cfg.use_perceiver

        if self.use_perceiver:
            self.perceiver = PerceiverLatentEncoder(
                num_latents=cfg.num_latents,
                d_latent=cfg.d_latent,
                d_input=cfg.d_shared,
                num_layers=cfg.num_layers,
                num_heads=cfg.num_heads,
                mlp_ratio=cfg.mlp_ratio,
                dropout=cfg.dropout,
            ).to(device, dtype=dtype)

            # Project each latent token into alignment space
            self.projector = ProjectorMLP(
                d_in=cfg.d_latent,
                d_out=cfg.d_align,
                hidden_factor=cfg.mlp_ratio,
                dropout=cfg.dropout,
            ).to(device, dtype=dtype)
        else:
            # No Perceiver: project pooled features directly into alignment space
            self.perceiver = None
            self.projector = ProjectorMLP(
                d_in=cfg.d_shared,
                d_out=cfg.d_align,
                hidden_factor=cfg.mlp_ratio,
                dropout=cfg.dropout,
            ).to(device, dtype=dtype)

    def forward(
        self,
        feats: Tensor,   # (B, T, D_feat)
        mask: Tensor,    # (B, T) bool
    ) -> Dict[str, Tensor]:
        """
        Returns:
            {
              "tokens": (B, L, d_align) or None,
              "pooled": (B, d_align),
            }
        """
        # Move to correct device/dtype if not already there
        feats = feats.to(self.device, dtype=self.dtype)
        mask = mask.to(self.device)

        # Step 1: adapter to shared dim
        feats_sh = self.adapter(feats)  # (B, T, d_shared)

        if self.use_perceiver:
            # Step 2: Perceiver over time dimension
            latents = self.perceiver(encoder_feats=feats_sh, encoder_mask=mask)  # (B, L, d_latent)

            # Step 3: Project to alignment space token-wise
            tokens = self.projector(latents)   # (B, L, d_align)

            # Step 4: Pooled representation
            pooled = tokens.mean(dim=1)        # (B, d_align)

            return {
                "tokens": tokens,
                "pooled": pooled,
            }
        else:
            # No Perceiver: masked mean over time then project
            pooled_in = masked_mean(feats_sh, mask)  # (B, d_shared)
            pooled = self.projector(pooled_in)       # (B, d_align)
            return {
                "tokens": None,
                "pooled": pooled,
            }


# ============================================================
# Text Alignment Head (project CLS/pooled text into d_align)
# ============================================================

class TextAlignmentHead(nn.Module):
    """
    Simple projector that maps text encoder outputs into the shared
    alignment space (d_align).

    Assumes you already have a text encoder that produces (B, D_text)
    pooled embeddings (e.g., CLS token or mean pooled).

    Usage:

        text_encoder: BaseTextEncoder  # your own wrapper
        text_head = TextAlignmentHead(d_text=text_encoder.d_model, d_align=1024)

        z_txt_raw = text_encoder.encode(text_batch)        # (B, D_text)
        z_txt = text_head(z_txt_raw)                       # (B, d_align)
    """

    def __init__(
        self,
        d_text: int,
        d_align: int,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
    ):
        super().__init__()

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.d_text = d_text
        self.d_align = d_align

        self.projector = ProjectorMLP(
            d_in=d_text,
            d_out=d_align,
            hidden_factor=mlp_ratio,
            dropout=dropout,
        ).to(device, dtype=dtype)

        self.device = device
        self.dtype = dtype

    def forward(self, text_embs: Tensor) -> Tensor:
        """
        text_embs: (B, D_text) pooled embeddings from a text encoder.

        Returns:
            z_text: (B, d_align) alignment-space embeddings.
        """
        text_embs = text_embs.to(self.device, dtype=self.dtype)
        z = self.projector(text_embs)
        return z
