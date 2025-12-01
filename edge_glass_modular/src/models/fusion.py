"""Multimodal fusion strategies."""

import torch
import torch.nn as nn
from typing import Dict, Optional, List
import math


class MultimodalFusion(nn.Module):
    """Fusion module for combining multiple modalities.

    Supports multiple fusion strategies:
    - 'concat': Simple concatenation + projection
    - 'cross_attention': Cross-modal attention
    - 'gated': Gated fusion with learned weights

    Args:
        modality_dims: Dictionary of {modality_name: dimension}
        fusion_dim: Output fusion dimension
        strategy: Fusion strategy ('concat', 'cross_attention', 'gated')
        num_layers: Number of fusion layers (for cross_attention)
        num_heads: Number of attention heads (for cross_attention)
        dropout: Dropout probability
    """

    def __init__(
        self,
        modality_dims: Dict[str, int],
        fusion_dim: int,
        strategy: str = "concat",
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.modality_dims = modality_dims
        self.fusion_dim = fusion_dim
        self.strategy = strategy
        self.modalities = list(modality_dims.keys())

        if strategy == "concat":
            # Simple concatenation + projection
            total_dim = sum(modality_dims.values())
            self.projector = nn.Sequential(
                nn.Linear(total_dim, fusion_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(fusion_dim * 2, fusion_dim),
                nn.LayerNorm(fusion_dim),
            )

        elif strategy == "cross_attention":
            # Project each modality to fusion_dim first
            self.modality_projectors = nn.ModuleDict({
                name: nn.Linear(dim, fusion_dim)
                for name, dim in modality_dims.items()
            })

            # Cross-attention layers
            self.fusion_layers = nn.ModuleList([
                CrossModalAttentionLayer(fusion_dim, num_heads, dropout)
                for _ in range(num_layers)
            ])

            self.norm = nn.LayerNorm(fusion_dim)

        elif strategy == "gated":
            # Gated fusion with learned importance
            self.modality_projectors = nn.ModuleDict({
                name: nn.Linear(dim, fusion_dim)
                for name, dim in modality_dims.items()
            })

            # Gate network
            self.gate = nn.Sequential(
                nn.Linear(fusion_dim * len(modality_dims), len(modality_dims)),
                nn.Softmax(dim=-1),
            )

            self.norm = nn.LayerNorm(fusion_dim)

        else:
            raise ValueError(f"Unknown fusion strategy: {strategy}")

    def forward(self, modality_embeddings: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Fuse multimodal embeddings.

        Args:
            modality_embeddings: Dict of {modality_name: embeddings (batch_size, dim)}

        Returns:
            Fused embeddings (batch_size, fusion_dim)
        """
        if self.strategy == "concat":
            # Concatenate all modalities
            embeddings_list = [modality_embeddings[name] for name in self.modalities if name in modality_embeddings]
            concat_emb = torch.cat(embeddings_list, dim=-1)
            fused = self.projector(concat_emb)

        elif self.strategy == "cross_attention":
            # Project each modality
            projected = {
                name: self.modality_projectors[name](emb)
                for name, emb in modality_embeddings.items()
            }

            # Stack for attention
            modality_list = [projected[name] for name in self.modalities if name in projected]
            stacked = torch.stack(modality_list, dim=1)  # (batch_size, num_modalities, fusion_dim)

            # Apply cross-modal attention
            for layer in self.fusion_layers:
                stacked = layer(stacked)

            # Pool across modalities
            fused = stacked.mean(dim=1)  # (batch_size, fusion_dim)
            fused = self.norm(fused)

        elif self.strategy == "gated":
            # Project each modality
            projected = {
                name: self.modality_projectors[name](emb)
                for name, emb in modality_embeddings.items()
            }

            # Compute gates
            modality_list = [projected[name] for name in self.modalities if name in projected]
            concat_for_gate = torch.cat(modality_list, dim=-1)
            gates = self.gate(concat_for_gate)  # (batch_size, num_modalities)

            # Weighted sum
            stacked = torch.stack(modality_list, dim=1)  # (batch_size, num_modalities, fusion_dim)
            fused = (stacked * gates.unsqueeze(-1)).sum(dim=1)  # (batch_size, fusion_dim)
            fused = self.norm(fused)

        return fused


class CrossModalAttentionLayer(nn.Module):
    """Cross-modal attention layer for fusion."""

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        assert dim % num_heads == 0

        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )

        # Feed-forward
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input (batch_size, num_modalities, dim)

        Returns:
            Output (batch_size, num_modalities, dim)
        """
        # Self-attention (cross-modal)
        residual = x
        x = self.norm1(x)
        attn_out, _ = self.self_attn(x, x, x)
        x = residual + attn_out

        # Feed-forward
        residual = x
        x = self.norm2(x)
        x = residual + self.ff(x)

        return x
