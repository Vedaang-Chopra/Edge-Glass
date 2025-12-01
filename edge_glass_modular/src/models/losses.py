"""Loss functions for multimodal alignment."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional


def contrastive_loss(
    embeddings_a: torch.Tensor,
    embeddings_b: torch.Tensor,
    temperature: float = 0.07,
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute contrastive (InfoNCE/CLIP-style) loss.

    Args:
        embeddings_a: First set of embeddings (batch_size, dim)
        embeddings_b: Second set of embeddings (batch_size, dim)
        temperature: Temperature parameter for softmax
        reduction: Reduction method ('mean', 'sum', 'none')

    Returns:
        Contrastive loss value
    """
    # Normalize embeddings
    embeddings_a = F.normalize(embeddings_a, p=2, dim=-1)
    embeddings_b = F.normalize(embeddings_b, p=2, dim=-1)

    # Compute similarity matrix
    similarity_matrix = torch.matmul(embeddings_a, embeddings_b.t()) / temperature

    batch_size = embeddings_a.shape[0]
    labels = torch.arange(batch_size, device=embeddings_a.device)

    # Symmetric cross-entropy loss
    loss_a_to_b = F.cross_entropy(similarity_matrix, labels, reduction=reduction)
    loss_b_to_a = F.cross_entropy(similarity_matrix.t(), labels, reduction=reduction)

    return (loss_a_to_b + loss_b_to_a) / 2


def mrl_loss(
    embeddings_a_mrl: Dict[int, torch.Tensor],
    embeddings_b_mrl: Dict[int, torch.Tensor],
    temperature: float = 0.07,
    weights: Optional[Dict[int, float]] = None,
) -> Dict[int, torch.Tensor]:
    """Compute Matryoshka Representation Learning loss at multiple dimensions.

    Args:
        embeddings_a_mrl: Dict of {dim: embeddings} for first modality
        embeddings_b_mrl: Dict of {dim: embeddings} for second modality
        temperature: Temperature for contrastive loss
        weights: Optional weights for each dimension

    Returns:
        Dictionary of {dim: loss_value}
    """
    losses = {}

    for dim in embeddings_a_mrl.keys():
        if dim not in embeddings_b_mrl:
            continue

        emb_a = embeddings_a_mrl[dim]
        emb_b = embeddings_b_mrl[dim]

        loss = contrastive_loss(emb_a, emb_b, temperature=temperature)

        # Apply weight if provided
        if weights is not None and dim in weights:
            loss = loss * weights[dim]

        losses[dim] = loss

    return losses


class AlignmentLoss(nn.Module):
    """Combined alignment loss with contrastive and MRL components.

    Args:
        contrastive_weight: Weight for main contrastive loss
        mrl_weight: Weight for MRL losses
        mrl_dimension_weights: Optional per-dimension weights for MRL
        temperature: Temperature for contrastive loss
    """

    def __init__(
        self,
        contrastive_weight: float = 1.0,
        mrl_weight: float = 0.05,
        mrl_dimension_weights: Optional[Dict[int, float]] = None,
        temperature: float = 0.07,
    ):
        super().__init__()

        self.contrastive_weight = contrastive_weight
        self.mrl_weight = mrl_weight
        self.mrl_dimension_weights = mrl_dimension_weights
        self.temperature = temperature

    def forward(
        self,
        embeddings_a: torch.Tensor,
        embeddings_b: torch.Tensor,
        embeddings_a_mrl: Optional[Dict[int, torch.Tensor]] = None,
        embeddings_b_mrl: Optional[Dict[int, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute combined alignment loss.

        Args:
            embeddings_a: Main embeddings from first modality (batch_size, dim)
            embeddings_b: Main embeddings from second modality (batch_size, dim)
            embeddings_a_mrl: Optional MRL embeddings for first modality
            embeddings_b_mrl: Optional MRL embeddings for second modality

        Returns:
            Dictionary with 'total_loss', 'contrastive_loss', and optional MRL losses
        """
        losses = {}

        # Main contrastive loss
        contrastive = contrastive_loss(embeddings_a, embeddings_b, temperature=self.temperature)
        losses["contrastive_loss"] = contrastive

        total_loss = self.contrastive_weight * contrastive

        # MRL losses if provided
        if embeddings_a_mrl is not None and embeddings_b_mrl is not None:
            mrl_losses = mrl_loss(
                embeddings_a_mrl,
                embeddings_b_mrl,
                temperature=self.temperature,
                weights=self.mrl_dimension_weights,
            )

            # Add MRL losses
            for dim, loss_val in mrl_losses.items():
                losses[f"mrl_loss_{dim}"] = loss_val
                total_loss += self.mrl_weight * loss_val

        losses["total_loss"] = total_loss

        return losses


class TriModalAlignmentLoss(nn.Module):
    """Alignment loss for three modalities (vision, audio, text).

    Computes pairwise contrastive losses:
    - Vision-Text
    - Audio-Text
    - Vision-Audio

    Args:
        vision_text_weight: Weight for vision-text alignment
        audio_text_weight: Weight for audio-text alignment
        vision_audio_weight: Weight for vision-audio alignment
        mrl_weight: Weight for MRL losses
        temperature: Temperature for contrastive loss
    """

    def __init__(
        self,
        vision_text_weight: float = 1.0,
        audio_text_weight: float = 1.0,
        vision_audio_weight: float = 0.5,
        mrl_weight: float = 0.05,
        temperature: float = 0.07,
    ):
        super().__init__()

        self.vision_text_weight = vision_text_weight
        self.audio_text_weight = audio_text_weight
        self.vision_audio_weight = vision_audio_weight
        self.mrl_weight = mrl_weight
        self.temperature = temperature

    def forward(
        self,
        vision_emb: torch.Tensor,
        audio_emb: torch.Tensor,
        text_emb: torch.Tensor,
        vision_mrl: Optional[Dict[int, torch.Tensor]] = None,
        audio_mrl: Optional[Dict[int, torch.Tensor]] = None,
        text_mrl: Optional[Dict[int, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute tri-modal alignment loss.

        Args:
            vision_emb: Vision embeddings (batch_size, dim)
            audio_emb: Audio embeddings (batch_size, dim)
            text_emb: Text embeddings (batch_size, dim)
            vision_mrl: Optional vision MRL embeddings
            audio_mrl: Optional audio MRL embeddings
            text_mrl: Optional text MRL embeddings

        Returns:
            Dictionary with all losses
        """
        losses = {}

        # Vision-Text loss
        vt_loss = contrastive_loss(vision_emb, text_emb, temperature=self.temperature)
        losses["vision_text_loss"] = vt_loss

        # Audio-Text loss
        at_loss = contrastive_loss(audio_emb, text_emb, temperature=self.temperature)
        losses["audio_text_loss"] = at_loss

        # Vision-Audio loss
        va_loss = contrastive_loss(vision_emb, audio_emb, temperature=self.temperature)
        losses["vision_audio_loss"] = va_loss

        # Total loss
        total_loss = (
            self.vision_text_weight * vt_loss
            + self.audio_text_weight * at_loss
            + self.vision_audio_weight * va_loss
        )

        # MRL losses if provided
        if vision_mrl is not None and text_mrl is not None:
            vt_mrl_losses = mrl_loss(vision_mrl, text_mrl, temperature=self.temperature)
            for dim, loss_val in vt_mrl_losses.items():
                losses[f"vision_text_mrl_{dim}"] = loss_val
                total_loss += self.mrl_weight * loss_val

        if audio_mrl is not None and text_mrl is not None:
            at_mrl_losses = mrl_loss(audio_mrl, text_mrl, temperature=self.temperature)
            for dim, loss_val in at_mrl_losses.items():
                losses[f"audio_text_mrl_{dim}"] = loss_val
                total_loss += self.mrl_weight * loss_val

        if vision_mrl is not None and audio_mrl is not None:
            va_mrl_losses = mrl_loss(vision_mrl, audio_mrl, temperature=self.temperature)
            for dim, loss_val in va_mrl_losses.items():
                losses[f"vision_audio_mrl_{dim}"] = loss_val
                total_loss += self.mrl_weight * loss_val

        losses["total_loss"] = total_loss

        return losses
