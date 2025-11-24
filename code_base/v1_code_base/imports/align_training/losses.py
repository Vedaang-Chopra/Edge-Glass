# alignment/alignment_losses.py

from typing import Iterable
import torch
import torch.nn.functional as F


def l2_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    L2-normalize along the last dimension.
    x: (..., D)
    """
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


def matryoshka_contrastive_loss(
    z_mod: torch.Tensor,
    z_txt: torch.Tensor,
    trunc_dims: Iterable[int],
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Multi-radius CLIP-style contrastive loss.

    Args:
        z_mod: (B, D) modality embeddings (vision or audio)
        z_txt: (B, D) text embeddings (same space)
        trunc_dims: list/tuple of radii (e.g. [256, 512, 1024])
        temperature: softmax temperature

    Returns:
        Scalar loss (mean over radii).
    """
    assert z_mod.shape == z_txt.shape, "z_mod and z_txt must have same shape"
    B, D = z_mod.shape
    device = z_mod.device

    trunc_dims = list(trunc_dims)
    targets = torch.arange(B, device=device)

    losses = []

    for d in trunc_dims:
        assert d <= D, f"Requested radius {d} > embedding dim {D}"
        zm = l2_normalize(z_mod[:, :d])
        zt = l2_normalize(z_txt[:, :d])

        # (B, B)
        logits = (zm @ zt.T) / temperature

        # symmetric CLIP loss
        loss_mod_to_txt = F.cross_entropy(logits, targets)
        loss_txt_to_mod = F.cross_entropy(logits.T, targets)

        losses.append(0.5 * (loss_mod_to_txt + loss_txt_to_mod))

    return sum(losses) / len(losses)
