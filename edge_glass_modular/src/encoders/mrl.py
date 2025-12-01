"""Matryoshka Representation Learning (MRL) projection."""

import torch
import torch.nn as nn
from typing import List, Dict


class MatryoshkaProjection(nn.Module):
    """Matryoshka Representation Learning projection.

    MRL creates nested embeddings at multiple granularities (dimensions).
    This allows using different embedding dimensions at inference time
    without retraining, trading off between speed and accuracy.

    For example, with dimensions [512, 256, 128]:
    - Full embedding: 1024 dims (most accurate, slower)
    - First 512 dims: High quality, faster
    - First 256 dims: Medium quality, even faster
    - First 128 dims: Lower quality, fastest

    During training, losses are computed at all MRL dimensions to ensure
    each prefix is informative.

    Args:
        input_dim: Dimension of input embeddings
        mrl_dimensions: List of MRL dimensions to supervise (e.g., [512, 256, 128])

    Example:
        >>> mrl = MatryoshkaProjection(input_dim=1024, mrl_dimensions=[512, 256, 128])
        >>> embeddings = torch.randn(32, 1024)
        >>> mrl_outputs = mrl(embeddings)
        >>> # mrl_outputs = {1024: (32, 1024), 512: (32, 512), 256: (32, 256), 128: (32, 128)}
    """

    def __init__(self, input_dim: int, mrl_dimensions: List[int]):
        super().__init__()

        self.input_dim = input_dim
        self.mrl_dimensions = sorted(mrl_dimensions, reverse=True)  # Largest to smallest

        # Validate dimensions
        for dim in self.mrl_dimensions:
            if dim > input_dim:
                raise ValueError(
                    f"MRL dimension {dim} cannot be larger than input_dim {input_dim}"
                )

        # Add full dimension if not included
        if input_dim not in self.mrl_dimensions:
            self.all_dimensions = [input_dim] + self.mrl_dimensions
        else:
            self.all_dimensions = [input_dim] + [d for d in self.mrl_dimensions if d != input_dim]

    def forward(self, x: torch.Tensor) -> Dict[int, torch.Tensor]:
        """Generate MRL embeddings at multiple dimensions.

        Args:
            x: Input embeddings (batch_size, input_dim)

        Returns:
            Dictionary mapping dimension -> embedding tensor
            {
                1024: (batch_size, 1024),  # Full embedding
                512: (batch_size, 512),     # First 512 dims
                256: (batch_size, 256),     # First 256 dims
                128: (batch_size, 128),     # First 128 dims
            }
        """
        mrl_embeddings = {}

        for dim in self.all_dimensions:
            # Take first `dim` dimensions
            embedding = x[:, :dim]

            # L2 normalize
            embedding = nn.functional.normalize(embedding, p=2, dim=-1)

            mrl_embeddings[dim] = embedding

        return mrl_embeddings

    def get_embedding(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        """Get embedding at specific MRL dimension.

        Args:
            x: Input embeddings (batch_size, input_dim)
            dim: Desired MRL dimension

        Returns:
            Embedding at specified dimension (batch_size, dim)
        """
        if dim > self.input_dim:
            raise ValueError(f"Dimension {dim} exceeds input_dim {self.input_dim}")

        embedding = x[:, :dim]
        return nn.functional.normalize(embedding, p=2, dim=-1)


def matryoshka_loss(
    embeddings_a: torch.Tensor,
    embeddings_b: torch.Tensor,
    mrl_dimensions: List[int],
    temperature: float = 0.07,
) -> Dict[int, torch.Tensor]:
    """Compute MRL contrastive loss at multiple dimensions.

    Args:
        embeddings_a: First set of embeddings (batch_size, dim)
        embeddings_b: Second set of embeddings (batch_size, dim)
        mrl_dimensions: List of MRL dimensions
        temperature: Temperature for contrastive loss

    Returns:
        Dictionary mapping dimension -> loss value
    """
    mrl_losses = {}
    batch_size = embeddings_a.shape[0]

    for dim in mrl_dimensions:
        # Take first `dim` dimensions
        emb_a = nn.functional.normalize(embeddings_a[:, :dim], p=2, dim=-1)
        emb_b = nn.functional.normalize(embeddings_b[:, :dim], p=2, dim=-1)

        # Compute similarity matrix
        sim_matrix = torch.matmul(emb_a, emb_b.t()) / temperature  # (B, B)

        # Labels: diagonal elements are positives
        labels = torch.arange(batch_size, device=emb_a.device)

        # Cross-entropy loss (row-wise and column-wise)
        loss_a = nn.functional.cross_entropy(sim_matrix, labels)
        loss_b = nn.functional.cross_entropy(sim_matrix.t(), labels)

        mrl_losses[dim] = (loss_a + loss_b) / 2

    return mrl_losses
