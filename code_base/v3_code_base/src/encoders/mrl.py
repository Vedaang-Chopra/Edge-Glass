"""Matryoshka representation learning projection."""

from __future__ import annotations

import torch
from torch import nn


class MatryoshkaProjection(nn.Module):
    def __init__(self, dim: int, radii):
        super().__init__()
        self.dim = dim
        self.radii = radii or [dim, dim // 2, dim // 4]
        self.projectors = nn.ModuleList([nn.Linear(dim, r, bias=False) for r in self.radii])

    def forward(self, embeddings: torch.Tensor):
        projections = [proj(embeddings) for proj in self.projectors]
        return torch.cat(projections, dim=-1)
