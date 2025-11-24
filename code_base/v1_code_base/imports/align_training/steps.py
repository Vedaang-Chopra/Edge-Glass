from dataclasses import dataclass
from typing import Callable, Dict, Any

import torch
import torch.nn as nn

from imports.align_training.losses import matryoshka_contrastive_loss


@dataclass
class AlignmentConfig:
    """Thin config wrapper for alignment training.

    You can construct this from your existing `cfg` in the notebook.

    Args:
        mrl_dims: tuple of Matryoshka radii (e.g., (256, 512, 1024))
        mrl_temperature: contrastive temperature
        max_text_length: (optional) max text length hint passed to text_embed_fn
    """
    mrl_dims: tuple[int, ...]
    mrl_temperature: float = 0.07
    max_text_length: int = 64   # for pooling text embeddings


@dataclass
class AlignmentModules:
    """Bundle all modules needed for the alignment step.

    Each attribute can be an nn.Module or None (e.g., audio_adapter if you
    are only training vision).
    """
    vision_adapter: nn.Module | None
    audio_adapter: nn.Module | None
    perceiver: nn.Module | None
    projector: nn.Module  # Perceiver dim -> LLM/text dim


def pooled_modality_embedding(latent_tokens_llm: torch.Tensor) -> torch.Tensor:
    """Simple mean-pooling over the latent sequence.

    Args:
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
    """One alignment step for a batch.

    Expects the output format of `collate_alignment` from dataset.py:

        batch = {
            "features":       (B, T, D_feat)  # CPU
            "feature_mask":   (B, T)          # CPU bool
            "input_ids":      (B, T_txt)      # CPU (unused here)
            "attention_mask": (B, T_txt)      # CPU (unused here)
            "modality_ids":   (B,)            # CPU (unused here)
            "raw_text":       list[str],
        }

    Args:
        batch: dict as above
        modality: "vision" or "audio"
        modules: AlignmentModules bundle
        cfg: AlignmentConfig
        text_embed_fn: function(texts, max_length) -> (B, D_llm) pooled text embeddings
        device: torch.device

    Returns:
        loss: scalar tensor
        metrics: dict[str, float]
    """
    # --- unpack & move features to device ---
    encoder_feats = batch["features"].to(device)         # (B, T, D_feat)
    encoder_mask = batch["feature_mask"].to(device)      # (B, T)
    texts = batch["raw_text"]                            # list[str]

    if modality == "vision":
        if modules.vision_adapter is None:
            raise ValueError("vision_adapter is None but modality='vision'")
        adapter = modules.vision_adapter
    elif modality == "audio":
        if modules.audio_adapter is None:
            raise ValueError("audio_adapter is None but modality='audio'")
        adapter = modules.audio_adapter
    else:
        raise ValueError(f"Unknown modality: {modality}")

    # 1) Adapter: raw encoder feats -> Perceiver input
    tokens = adapter(encoder_feats)  # (B, T, D_perceiver)

    # 2) Perceiver: (B, T, D_perceiver) -> (B, L, D_perceiver)
    if modules.perceiver is not None:
        latents = modules.perceiver(tokens, encoder_mask=encoder_mask)  # (B, L, D_perceiver)
    else:
        # If no Perceiver, treat tokens as latents
        latents = tokens

    # 3) Projector: Perceiver -> LLM/text dim
    latent_tokens_llm = modules.projector(latents)  # (B, L, D_llm)

    # 4) Pool modality embedding
    h_mod = pooled_modality_embedding(latent_tokens_llm)  # (B, D_llm)

    # 5) Text embeddings from HFTextEncoder (or any other)
    h_txt = text_embed_fn(texts, cfg.max_text_length)  # (B, D_llm)

    # 6) Matryoshka contrastive loss
    loss = matryoshka_contrastive_loss(
        h_mod,
        h_txt,
        radii=cfg.mrl_dims,
        temperature=cfg.mrl_temperature,
        symmetric=True,
    )

    metrics = {
        f"{modality}/mrl_loss": float(loss.detach().cpu().item()),
    }

    return loss, metrics
