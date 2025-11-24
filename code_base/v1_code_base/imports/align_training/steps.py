from dataclasses import dataclass
from typing import Callable, Dict, Any, Optional

import torch
import torch.nn as nn

from imports.align_training.losses import (
    matryoshka_contrastive_loss,
    clip_contrastive_loss,
)


@dataclass
class AlignmentConfig:
    """
    Thin config wrapper for alignment training.

    You normally construct this in your notebook from the global `cfg`,
    then pass it into `train_alignment` and `eval_alignment`.

    Args
    ----
    mrl_dims:
        Tuple of radii for Matryoshka loss (e.g. (256, 512, 1024)).
    mrl_temperature:
        Temperature for the MRL contrastive loss.
    clip_temperature:
        Temperature for the CLIP-style contrastive loss.
    mrl_weight:
        Weight for the MRL term in the total loss.
    clip_weight:
        Weight for the CLIP term in the total loss.
    max_text_length:
        Max token length for the text encoder.
    """
    mrl_dims: tuple[int, ...]
    mrl_temperature: float = 0.07

    clip_temperature: float = 0.07
    mrl_weight: float = 1.0
    clip_weight: float = 1.0

    max_text_length: int = 128


@dataclass
class AlignmentModules:
    """
    Simple container bundling all trainable alignment modules.

    All fields are `nn.Module` or `None`.
    Only these are optimized by `build_alignment_optimizer`.

    Attributes
    ----------
    vision_adapter:
        Linear (or small MLP) that maps vision encoder feats -> shared dim.
    audio_adapter:
        Linear (or small MLP) that maps audio encoder feats -> shared dim.
    perceiver:
        Optional PerceiverLatentEncoder over shared tokens.
    projector:
        MLP mapping from shared / latent dim -> final aligned dim.
    """
    vision_adapter: Optional[nn.Module]
    audio_adapter: Optional[nn.Module]
    perceiver: Optional[nn.Module]
    projector: nn.Module


def pooled_modality_embedding(latent_tokens: torch.Tensor) -> torch.Tensor:
    """
    Pool a sequence of latent tokens into a single embedding.

    latent_tokens: (B, L, D)
    returns: (B, D)
    """
    return latent_tokens.mean(dim=1)


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Masked mean pooling over the sequence dimension.

    x: (B, T, D)
    mask: (B, T) bool

    returns: (B, D)
    """
    mask = mask.to(x.dtype)
    denom = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
    return (x * mask.unsqueeze(-1)).sum(dim=1) / denom


def forward_alignment_step(
    batch: Dict[str, Any],
    modality: str,
    modules: AlignmentModules,
    cfg: AlignmentConfig,
    text_embed_fn: Callable[[list[str], int], torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, Dict[str, float]]:
    """
    One full forward/backward step for alignment training or evaluation.

    Expected `batch` format (from `collate_alignment`):
        {
            "features":       (B, T, D_feat)   # CPU float
            "feature_mask":   (B, T)           # CPU bool
            "input_ids":      (B, T_txt)       # CPU (unused here)
            "attention_mask": (B, T_txt)       # CPU (unused here)
            "modality_ids":   (B,)             # CPU (unused here)
            "raw_text":       list[str],
        }

    Args
    ----
    batch:
        Mini-batch from a feature dataset.
    modality:
        Either "vision" or "audio". Selects the corresponding adapter.
    modules:
        AlignmentModules container.
    cfg:
        AlignmentConfig with loss hyperparameters.
    text_embed_fn:
        Callable(texts, max_length) -> (B, D_text) pooled text embeddings.
        Normally wraps HFTextEncoder (and possibly a text projector).
    device:
        torch.device used for tensors.

    Returns
    -------
    loss:
        Scalar tensor suitable for backward().
    metrics:
        Dict of scalar floats for logging.
    """
    # ------------------------------------------------------------------
    # 1) Move features & masks to device
    # ------------------------------------------------------------------
    encoder_feats: torch.Tensor = batch["features"].to(device)       # (B, T, D_feat)
    encoder_mask: torch.Tensor = batch["feature_mask"].to(device)    # (B, T)
    texts: list[str] = batch["raw_text"]

    if modality == "vision":
        adapter = modules.vision_adapter
    elif modality == "audio":
        adapter = modules.audio_adapter
    else:
        raise ValueError(f"Unknown modality: {modality}")

    if adapter is None:
        raise RuntimeError(f"{modality}_adapter is None in AlignmentModules")

    # ------------------------------------------------------------------
    # 2) Map raw encoder feats into shared space
    # ------------------------------------------------------------------
    encoder_feats = encoder_feats.to(dtype=cfg.torch_dtype)  # for stability
    shared_tokens = adapter(encoder_feats)  # (B, T, d_shared)

    # ------------------------------------------------------------------
    # 3) Optional Perceiver latent bottleneck, then projector
    # ------------------------------------------------------------------
    if modules.perceiver is not None:
        latents = modules.perceiver(shared_tokens, encoder_mask=encoder_mask)  # (B, L, d_latent)
        latent_tokens = modules.projector(latents)                             # (B, L, d_align)
        h_mod = pooled_modality_embedding(latent_tokens)                       # (B, d_align)
    else:
        pooled_shared = masked_mean(shared_tokens, encoder_mask)               # (B, d_shared)
        h_mod = modules.projector(pooled_shared)                               # (B, d_align)

    # ------------------------------------------------------------------
    # 4) Text embeddings (using frozen text encoder wrapper)
    # ------------------------------------------------------------------
    h_txt = text_embed_fn(texts, cfg.max_text_length)                          # (B, d_align) or (B, D_txt)

    # Cast to float32 for contrastive losses (stability on fp16 / bfloat16)
    h_mod = h_mod.to(torch.float32)
    h_txt = h_txt.to(torch.float32)

    # ------------------------------------------------------------------
    # 5) Contrastive losses: Matryoshka + CLIP
    # ------------------------------------------------------------------
    loss_mrl = matryoshka_contrastive_loss(
        h_mod,
        h_txt,
        radii=cfg.mrl_dims,
        temperature=cfg.mrl_temperature,
        symmetric=True,
    )

    loss_clip = clip_contrastive_loss(
        h_mod,
        h_txt,
        temperature=cfg.clip_temperature,
        symmetric=True,
    )

    loss = cfg.mrl_weight * loss_mrl + cfg.clip_weight * loss_clip

    metrics: Dict[str, float] = {
        f"{modality}/loss": float(loss.detach().cpu().item()),
        f"{modality}/mrl_loss": float(loss_mrl.detach().cpu().item()),
        f"{modality}/clip_loss": float(loss_clip.detach().cpu().item()),
    }

    return loss, metrics
