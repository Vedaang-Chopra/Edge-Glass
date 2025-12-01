"""Perceiver Resampler for compressing variable-length sequences."""

import torch
import torch.nn as nn
import math


class PerceiverAttention(nn.Module):
    """Cross-attention layer for Perceiver."""

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_kv = nn.Linear(dim, dim * 2, bias=False)
        self.to_out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input sequence (batch_size, seq_len, dim)
            latents: Latent queries (batch_size, num_latents, dim)

        Returns:
            Updated latents (batch_size, num_latents, dim)
        """
        batch_size, num_latents, _ = latents.shape

        # Queries from latents
        q = self.to_q(latents)  # (B, num_latents, dim)

        # Keys and values from input
        kv = self.to_kv(x).chunk(2, dim=-1)  # 2 x (B, seq_len, dim)
        k, v = kv

        # Reshape for multi-head attention
        q = q.view(batch_size, num_latents, self.num_heads, self.head_dim)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim)

        # Transpose to (B, num_heads, num_latents/seq_len, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, num_latents, seq_len)
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = attn @ v  # (B, num_heads, num_latents, head_dim)

        # Reshape back
        out = out.transpose(1, 2).contiguous()  # (B, num_latents, num_heads, head_dim)
        out = out.view(batch_size, num_latents, -1)  # (B, num_latents, dim)

        return self.to_out(out)


class FeedForward(nn.Module):
    """Feed-forward network with GELU activation."""

    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PerceiverResamplerLayer(nn.Module):
    """Single Perceiver layer with cross-attention and feed-forward."""

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()

        self.attn = PerceiverAttention(dim, num_heads, dropout)
        self.ff = FeedForward(dim, dropout=dropout)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input sequence (batch_size, seq_len, dim)
            latents: Latent queries (batch_size, num_latents, dim)

        Returns:
            Updated latents (batch_size, num_latents, dim)
        """
        # Cross-attention from latents to input
        latents = latents + self.attn(self.norm1(x), self.norm1(latents))

        # Feed-forward
        latents = latents + self.ff(self.norm2(latents))

        return latents


class PerceiverResampler(nn.Module):
    """Perceiver Resampler: Compress variable-length sequences to fixed-size latents.

    The Perceiver uses learned latent queries that attend to the input sequence,
    compressing variable-length inputs into a fixed number of latent representations.

    Architecture:
        Input: (batch_size, seq_len, dim)
        Latents: (num_latents, dim) - learned parameters

        For each layer:
            - Cross-attention: latents attend to input
            - Feed-forward: process latents

        Output: (batch_size, num_latents, dim)

    This provides O(num_latents * seq_len) complexity instead of O(seq_len^2)
    for standard self-attention, making it efficient for long sequences.

    Args:
        dim: Dimension of input and latent embeddings
        num_latents: Number of latent query tokens (e.g., 64)
        num_layers: Number of Perceiver layers (e.g., 3)
        num_heads: Number of attention heads (e.g., 8)
        dropout: Dropout probability
    """

    def __init__(
        self,
        dim: int = 512,
        num_latents: int = 64,
        num_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.dim = dim
        self.num_latents = num_latents

        # Learned latent queries
        self.latents = nn.Parameter(torch.randn(num_latents, dim))
        nn.init.normal_(self.latents, std=0.02)

        # Perceiver layers
        self.layers = nn.ModuleList(
            [PerceiverResamplerLayer(dim, num_heads, dropout) for _ in range(num_layers)]
        )

        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Resample input sequence to fixed-size latents.

        Args:
            x: Input sequence (batch_size, seq_len, dim)

        Returns:
            Latent representations (batch_size, num_latents, dim)
        """
        batch_size = x.shape[0]

        # Expand latents for batch
        latents = self.latents.unsqueeze(0).expand(batch_size, -1, -1)

        # Apply Perceiver layers
        for layer in self.layers:
            latents = layer(x, latents)

        # Final normalization
        latents = self.norm(latents)

        return latents

    @property
    def num_parameters(self) -> int:
        """Return number of parameters."""
        return sum(p.numel() for p in self.parameters())
