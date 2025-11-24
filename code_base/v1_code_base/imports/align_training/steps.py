# alignment/alignment_steps.py

from dataclasses import dataclass
from typing import Callable, Dict, Any

import torch
import torch.nn as nn

from imports.align_training.losses import matryoshka_contrastive_loss


@dataclass
class AlignmentConfig:
    """
    Thin config wrapper so we don't depend on your big notebook dataclass directly.
    You can construct this from your existing `cfg` in the notebook.
    """
    mrl_dims: tuple[int, ...]
    mrl_temperature: float = 0.07
    max_text_length: int = 64   # for pooling text embeddings


@dataclass
class AlignmentModules:
    """
    Bundle all modules needed for the alignment step.
    """
    vision_adapter: nn.Module
    audio_adapter: nn.Module
    perceiver: nn.Module
    projector: nn.Module  # Perceiver dim -> LLM/Qwen dim


def pooled_modality_embedding(latent_tokens_llm: torch.Tensor) -> torch.Tensor:
    """
    Simple mean-pooling over the latent sequence.
    latent_tokens_llm: (B, L, D_llm)
    Returns:
        (B, D_llm)
    """
    return latent_tokens_llm.mean(dim=1)


def forward_alignment_step(
    batch: Dict[str, Any],
    modality: str,
    modules: AlignmentModules,
    cfg: AlignmentConfig,
    text_embed_fn: Callable[[list[str], int], torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, Dict[str, float]]:
    """
    One alignment step for a batch.

    Args:
        batch: dict with keys:
            - "encoder_feats": (B, T, D_enc)
            - "encoder_mask": (B, T) bool or 0/1
            - "texts": list[str]
        modality: "vision" or "audio"
        modules: AlignmentModules bundle
        cfg: AlignmentConfig
        text_embed_fn: function(texts, max_length) -> (B, D_llm) pooled text embeddings
        device: torch.device

    Returns:
        loss: scalar tensor
        metrics: dict[str, float]
    """
    encoder_feats = batch["encoder_feats"].to(device)
    encoder_mask = batch["encoder_mask"].to(device)
    texts = batch["texts"]

    if modality == "vision":
        adapter = modules.vision_adapter
    elif modality == "audio":
        adapter = modules.audio_adapter
    else:
        raise ValueError(f"Unknown modality: {modality}")

    # 1) Adapter: raw encoder feats -> Perceiver input
    tokens = adapter(encoder_feats)  # (B, T, D_perceiver)

    # 2) Perceiver: (B, T, D_perceiver) -> (B, L, D_perceiver)
    latents = modules.perceiver(tokens, encoder_mask=encoder_mask)  # (B, L, D_perceiver)

    # 3) Projector: Perceiver -> LLM dim
    latent_tokens_llm = modules.projector(latents)  # (B, L, D_llm)

    # 4) Pool modality embedding
    h_mod = pooled_modality_embedding(latent_tokens_llm)  # (B, D_llm)

    # 5) Text embeddings from Qwen (or any LLM text encoder)
    h_txt = text_embed_fn(texts, cfg.max_text_length)  # (B, D_llm)

    # 6) Matryoshka contrastive loss
    loss = matryoshka_contrastive_loss(
        h_mod,
        h_txt,
        trunc_dims=cfg.mrl_dims,
        temperature=cfg.mrl_temperature,
    )

    metrics = {
        f"{modality}/mrl_loss": float(loss.detach().cpu().item()),
    }

    return loss, metrics
