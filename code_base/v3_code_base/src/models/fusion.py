"""Modal fusion module for decoder alignment."""

from __future__ import annotations

from typing import Dict

import torch
from torch import nn

from .projector import ProjectionHead


class MultimodalFusion(nn.Module):
    def __init__(self, dim: int, decoder_dim: int):
        super().__init__()
        self.vision_proj = ProjectionHead(dim, decoder_dim)
        self.audio_proj = ProjectionHead(dim, decoder_dim)
        self.text_proj = ProjectionHead(dim, decoder_dim)

    def forward(self, embeddings: Dict[str, torch.Tensor]):
        fused = []
        if "vision" in embeddings:
            fused.append(self.vision_proj(embeddings["vision"]))
        if "audio" in embeddings:
            fused.append(self.audio_proj(embeddings["audio"]))
        if "text" in embeddings:
            fused.append(self.text_proj(embeddings["text"]))
        return torch.stack(fused).mean(0)
