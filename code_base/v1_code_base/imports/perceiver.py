"""
Perciever.py

Perceiver-style latent encoder + simple MLP projector for alignment.

Intended usage (in a notebook):

    from Perciever import PerceiverLatentEncoder, ProjectorMLP
    from encoders import VisionEncoder

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vision_enc = VisionEncoder(...)

    perceiver = PerceiverLatentEncoder(
        num_latents=64,
        d_latent=512,
        d_input=vision_enc.feat_dim,   # after one forward
        num_layers=4,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.1,
    ).to(device)

    proj = ProjectorMLP(
        d_in=512,
        d_out=1024,
        hidden_factor=2.0,
        dropout=0.1,
    ).to(device)

    # encoder_out["feats"]: (B, T, D_in), encoder_out["mask"]: (B, T)
    encoder_out = vision_enc.encode_images(images)
    feats, mask = encoder_out["feats"], encoder_out["mask"]

    latents = perceiver(feats, encoder_mask=mask)  # (B, L, d_latent)
    z = proj(latents)  # (B, L, d_out)   # for alignment / LLM prefix, etc.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor


# ---------------------------------------------------------------------------
# Core building blocks
# ---------------------------------------------------------------------------


class MultiHeadAttention(nn.Module):
    """
    Simple multi-head attention with separate Q/K/V projections.

    Works for both:
        - cross-attention: Q = latents, K/V = encoder_feats
        - self-attention:  Q = K = V = latents
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x_q: Tensor,                 # (B, L_q, d_model)
        x_kv: Tensor,                # (B, L_kv, d_model)
        mask_kv: Optional[Tensor] = None,  # (B, L_kv) bool, True = keep
    ) -> Tensor:
        B, L_q, _ = x_q.shape
        B2, L_kv, _ = x_kv.shape
        assert B == B2, "Batch size mismatch between x_q and x_kv"

        # Linear projections
        q = self.q_proj(x_q)   # (B, L_q, d_model)
        k = self.k_proj(x_kv)  # (B, L_kv, d_model)
        v = self.v_proj(x_kv)  # (B, L_kv, d_model)

        # Reshape to (B, heads, L, head_dim)
        q = q.view(B, L_q, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, L_q, Hd)
        k = k.view(B, L_kv, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, L_kv, Hd)
        v = v.view(B, L_kv, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, L_kv, Hd)

        # Scaled dot-product attention
        attn_logits = torch.matmul(q, k.transpose(-1, -2))  # (B, H, L_q, L_kv)
        attn_logits = attn_logits / math.sqrt(self.head_dim)

        if mask_kv is not None:
            # mask_kv: (B, L_kv) bool, True = keep, False = pad
            mask_kv = mask_kv.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, L_kv)
            attn_logits = attn_logits.masked_fill(~mask_kv, float("-inf"))

        attn_weights = F.softmax(attn_logits, dim=-1)  # (B, H, L_q, L_kv)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)  # (B, H, L_q, Hd)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L_q, self.d_model)  # (B, L_q, d_model)

        out = self.out_proj(attn_output)
        return out


class FeedForward(nn.Module):
    """
    Simple Transformer-style feed-forward MLP:
        x -> LN -> Linear -> GELU -> Dropout -> Linear -> Dropout
    """

    def __init__(
        self,
        d_model: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_dim = int(d_model * mlp_ratio)
        self.lin1 = nn.Linear(d_model, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, d_model)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.lin1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.lin2(x)
        x = self.dropout(x)
        return x


class PerceiverBlock(nn.Module):
    """
    One Perceiver-style block:

        1) Cross-attention: latents attend over encoder tokens
        2) Self-attention: latents attend over themselves
        3) Feed-forward MLP

    All with pre-LayerNorm + residual connections.
    """

    def __init__(
        self,
        d_latent: int,
        d_input: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        # Project encoder features to d_latent for cross-attn
        self.enc_proj = nn.Linear(d_input, d_latent)

        self.ln_latents_1 = nn.LayerNorm(d_latent)
        self.ln_latents_2 = nn.LayerNorm(d_latent)
        self.ln_latents_3 = nn.LayerNorm(d_latent)

        self.cross_attn = MultiHeadAttention(
            d_model=d_latent,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.self_attn = MultiHeadAttention(
            d_model=d_latent,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.mlp = FeedForward(
            d_model=d_latent,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        latents: Tensor,          # (B, L, d_latent)
        encoder_feats: Tensor,    # (B, T, d_input)
        encoder_mask: Optional[Tensor] = None,  # (B, T) bool
    ) -> Tensor:
        # 1) Cross-attention over encoder tokens
        enc = self.enc_proj(encoder_feats)   # (B, T, d_latent)

        latents_norm = self.ln_latents_1(latents)
        cross_out = self.cross_attn(latents_norm, enc, mask_kv=encoder_mask)
        latents = latents + self.dropout(cross_out)

        # 2) Self-attention on latents
        latents_norm = self.ln_latents_2(latents)
        self_out = self.self_attn(latents_norm, latents_norm, mask_kv=None)
        latents = latents + self.dropout(self_out)

        # 3) Feed-forward
        latents_norm = self.ln_latents_3(latents)
        mlp_out = self.mlp(latents_norm)
        latents = latents + self.dropout(mlp_out)

        return latents


# ---------------------------------------------------------------------------
# Perceiver Latent Encoder
# ---------------------------------------------------------------------------


@dataclass
class PerceiverConfig:
    num_latents: int = 64
    d_latent: int = 512
    d_input: int = 1024  # must be set to encoder feature dim
    num_layers: int = 4
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.0


class PerceiverLatentEncoder(nn.Module):
    """
    Perceiver-style latent encoder.

    - Start from a learned latent array Z_0 ∈ R^{L × d_latent}
    - For each block:
        CrossAttn(latents, encoder_feats) -> SelfAttn(latents) -> MLP(latents)
    - Returns Z_final ∈ R^{B × L × d_latent}

    Inputs:
        encoder_feats: (B, T, d_input)
        encoder_mask:  (B, T) bool, True = valid, False = pad
    """

    def __init__(
        self,
        num_latents: int,
        d_latent: int,
        d_input: int,
        num_layers: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.cfg = PerceiverConfig(
            num_latents=num_latents,
            d_latent=d_latent,
            d_input=d_input,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

        # Learned latents (L, d_latent)
        self.latents = nn.Parameter(
            torch.randn(self.cfg.num_latents, self.cfg.d_latent) * 0.02
        )

        # Stack of Perceiver blocks
        self.blocks = nn.ModuleList(
            [
                PerceiverBlock(
                    d_latent=self.cfg.d_latent,
                    d_input=self.cfg.d_input,
                    num_heads=self.cfg.num_heads,
                    mlp_ratio=self.cfg.mlp_ratio,
                    dropout=self.cfg.dropout,
                )
                for _ in range(self.cfg.num_layers)
            ]
        )

    def forward(
        self,
        encoder_feats: Tensor,          # (B, T, d_input)
        encoder_mask: Optional[Tensor] = None,  # (B, T) bool
    ) -> Tensor:
        B, T, D_in = encoder_feats.shape
        assert D_in == self.cfg.d_input, (
            f"encoder_feats dim {D_in} != d_input {self.cfg.d_input}. "
            f"Make sure d_input matches your encoder feature dimension."
        )

        # Broadcast learned latents to batch: (B, L, d_latent)
        latents = self.latents.unsqueeze(0).expand(B, -1, -1)

        for block in self.blocks:
            latents = block(latents, encoder_feats, encoder_mask)

        return latents  # (B, L, d_latent)


# ---------------------------------------------------------------------------
# Simple MLP Projector
# ---------------------------------------------------------------------------


class ProjectorMLP(nn.Module):
    """
    Generic MLP projector:

        x -> LayerNorm -> Linear -> GELU -> Dropout -> Linear

    Works on either:
        - sequence of latents: (B, L, d_in)
        - pooled embeddings:  (B, d_in)
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        hidden_factor: float = 2.0,
        dropout: float = 0.0,
        use_layernorm: bool = True,
    ) -> None:
        super().__init__()

        hidden_dim = int(d_in * hidden_factor)
        self.use_layernorm = use_layernorm

        self.ln = nn.LayerNorm(d_in) if use_layernorm else nn.Identity()
        self.fc1 = nn.Linear(d_in, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, d_out)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, L, d_in) or (B, d_in)
        x = self.ln(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


__all__ = [
    "PerceiverConfig",
    "PerceiverLatentEncoder",
    "PerceiverBlock",
    "MultiHeadAttention",
    "FeedForward",
    "ProjectorMLP",
]
