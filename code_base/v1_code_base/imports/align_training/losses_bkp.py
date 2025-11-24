"""
losses.py

Implements:
- L2 normalization utilities
- Pairwise cosine similarity
- CLIP-style symmetric contrastive loss
- Matryoshka Representation Learning (MRL) contrastive loss

These are encoder-agnostic and operate only on (B, D) embeddings.
"""

from __future__ import annotations

from typing import Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Utility Functions
# ============================================================

def l2_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Normalize embeddings along the last dimension.
    x: (..., D)
    """
    return x / (x.norm(dim=-1, keepdim=True).clamp_min(eps))


def pairwise_cosine_sim(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise cosine similarity between two batches of embeddings.

    Args:
        z1: (B, D)
        z2: (B, D)

    Returns:
        sim: (B, B)
    """
    z1 = l2_normalize(z1)
    z2 = l2_normalize(z2)
    return z1 @ z2.T  # (B, B)


# ============================================================
# CLIP Contrastive Loss (base)
# ============================================================

def clip_contrastive_loss(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    temperature: float = 0.07,
    symmetric: bool = True,
) -> torch.Tensor:
    """
    CLIP-style InfoNCE loss between two embedding sets.

    Args:
        z_a: (B, D) modality A embeddings
        z_b: (B, D) modality B embeddings
        temperature: contrastive temperature
        symmetric: compute both A->B and B->A

    Returns:
        scalar loss
    """
    assert z_a.shape == z_b.shape, f"Shape mismatch: {z_a.shape} vs {z_b.shape}"
    B = z_a.size(0)
    device = z_a.device

    logits = pairwise_cosine_sim(z_a, z_b) / temperature
    targets = torch.arange(B, device=device)

    # A → B
    loss_ab = F.cross_entropy(logits, targets)

    if not symmetric:
        return loss_ab

    # B → A
    loss_ba = F.cross_entropy(logits.T, targets)

    return 0.5 * (loss_ab + loss_ba)


# ============================================================
# Matryoshka Contrastive Loss (MRL)
# ============================================================

def matryoshka_contrastive_loss(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    radii: Sequence[int],
    temperature: float = 0.07,
    symmetric: bool = True,
) -> torch.Tensor:
    """
    Matryoshka (nested) contrastive loss.
    Enforces good alignment at multiple embedding truncation radii.

    Args:
        z_a: (B, D) modality A embeddings
        z_b: (B, D) modality B embeddings
        radii: list of radii, e.g. [256, 512, 1024]
        temperature: contrastive temp
        symmetric: CLIP symmetric loss

    Returns:
        scalar loss
    """
    assert z_a.shape == z_b.shape
    B, D = z_a.shape
    max_r = max(radii)
    assert max_r <= D, f"Max radius {max_r} > embedding dim {D}"

    losses = []

    for r in radii:
        # truncate
        za = z_a[:, :r]
        zb = z_b[:, :r]

        # normalize after truncation
        za = l2_normalize(za)
        zb = l2_normalize(zb)

        # compute logits
        logits = (za @ zb.T) / temperature
        targets = torch.arange(B, device=z_a.device)

        loss_ab = F.cross_entropy(logits, targets)
        if symmetric:
            loss_ba = F.cross_entropy(logits.T, targets)
            loss_r = 0.5 * (loss_ab + loss_ba)
        else:
            loss_r = loss_ab

        losses.append(loss_r)

    # Average over radii
    return sum(losses) / len(losses)
