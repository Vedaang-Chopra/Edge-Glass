"\"\"\"Lightweight Perceiver resampler for ablations.\"\"\""

from __future__ import annotations

import torch
from torch import nn


class PerceiverAdapter(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, num_layers: int):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(1, 64, latent_dim))
        layers = []
        for _ in range(num_layers):
            layers.append(nn.MultiheadAttention(latent_dim, num_heads=8, batch_first=True))
            layers.append(nn.LayerNorm(latent_dim))
        self.layers = nn.ModuleList(layers)
        self.in_proj = nn.Linear(input_dim, latent_dim)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(sequence)
        latents = self.latents.expand(sequence.size(0), -1, -1)
        for attn, norm in zip(self.layers[::2], self.layers[1::2]):
            attended, _ = attn(latents, x, x)
            latents = norm(attended + latents)
        return latents
