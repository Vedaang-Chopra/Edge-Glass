"""Learnable attention pooling for embeddings."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class AttentionPooling(nn.Module):
    """Learnable attention pooling mechanism.

    Instead of simple mean or CLS pooling, this module learns to attend
    to the most relevant tokens in a sequence using a query-based attention.

    Args:
        input_dim: Dimension of input embeddings
        num_queries: Number of attention queries (default: 1 for single pooled output)
        num_heads: Number of attention heads
        dropout: Dropout probability
    """

    def __init__(
        self,
        input_dim: int,
        num_queries: int = 1,
        num_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.num_queries = num_queries
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads

        assert (
            self.head_dim * num_heads == input_dim
        ), "input_dim must be divisible by num_heads"

        # Learnable query tokens
        self.queries = nn.Parameter(torch.randn(1, num_queries, input_dim))

        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Layer norm
        self.layer_norm = nn.LayerNorm(input_dim)

        # Initialize queries
        nn.init.normal_(self.queries, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply attention pooling.

        Args:
            x: Input sequence (batch_size, seq_len, input_dim)
            attention_mask: Optional mask (batch_size, seq_len)

        Returns:
            Pooled output (batch_size, num_queries, input_dim) or
                          (batch_size, input_dim) if num_queries == 1
        """
        batch_size = x.shape[0]

        # Expand queries for batch
        queries = self.queries.expand(batch_size, -1, -1)  # (B, num_queries, dim)

        # Apply layer norm to input
        x = self.layer_norm(x)

        # Multi-head attention: queries attend to input sequence
        # key_padding_mask: True for positions to ignore
        if attention_mask is not None:
            # Convert to key padding mask (True = ignore)
            key_padding_mask = ~attention_mask.bool()
        else:
            key_padding_mask = None

        pooled, _ = self.attention(
            query=queries,
            key=x,
            value=x,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )  # (B, num_queries, dim)

        # If single query, squeeze
        if self.num_queries == 1:
            pooled = pooled.squeeze(1)  # (B, dim)

        return pooled


class SimpleAttentionPooling(nn.Module):
    """Simple single-layer attention pooling.

    More lightweight than multi-head attention, using a simple
    attention mechanism with a learnable query vector.

    Args:
        input_dim: Dimension of input embeddings
        dropout: Dropout probability
    """

    def __init__(self, input_dim: int, dropout: float = 0.0):
        super().__init__()

        self.input_dim = input_dim

        # Learnable query vector
        self.query = nn.Parameter(torch.randn(input_dim))

        # Attention projection
        self.attn_proj = nn.Linear(input_dim, 1)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Layer norm
        self.layer_norm = nn.LayerNorm(input_dim)

        # Initialize
        nn.init.normal_(self.query, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply simple attention pooling.

        Args:
            x: Input sequence (batch_size, seq_len, input_dim)
            attention_mask: Optional mask (batch_size, seq_len)

        Returns:
            Pooled output (batch_size, input_dim)
        """
        # Layer norm
        x = self.layer_norm(x)  # (B, L, D)

        # Compute attention scores
        attn_scores = self.attn_proj(x).squeeze(-1)  # (B, L)

        # Apply mask if provided
        if attention_mask is not None:
            attn_scores = attn_scores.masked_fill(~attention_mask.bool(), float('-inf'))

        # Softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)  # (B, L)
        attn_weights = self.dropout(attn_weights)

        # Weighted sum
        pooled = torch.bmm(
            attn_weights.unsqueeze(1),  # (B, 1, L)
            x,  # (B, L, D)
        ).squeeze(1)  # (B, D)

        return pooled
