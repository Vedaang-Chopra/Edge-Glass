"""Vision encoder factory wrapping CLIP/ViT style models."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from transformers import AutoModel, AutoProcessor

from ..config import EncoderConfig
from ..utils.registry import ENCODER_REGISTRY
from .perceiver import PerceiverAdapter
from .mrl import MatryoshkaProjection


@dataclass
class VisionEncoderOutput:
    pooled: torch.Tensor
    sequence: torch.Tensor


class VisionEncoder(nn.Module):
    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.model = AutoModel.from_pretrained(cfg.model_name, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(cfg.model_name, trust_remote_code=True)
        self.projector = nn.Linear(self.model.config.hidden_size, cfg.projection_dim)
        self.perceiver = (
            PerceiverAdapter(cfg.projection_dim, cfg.perceiver_dim, cfg.perceiver_layers)
            if cfg.use_perceiver
            else None
        )
        self.mrl = MatryoshkaProjection(cfg.projection_dim, cfg.mrl_dimensions) if cfg.use_mrl else None
        for param in self.model.parameters():
            param.requires_grad = cfg.trainable

    def forward(self, images: torch.Tensor) -> VisionEncoderOutput:
        outputs = self.model(pixel_values=images)
        pooled = outputs.pooler_output
        seq = outputs.last_hidden_state
        proj_seq = self.projector(seq)
        pooled = self.projector(pooled)
        if self.perceiver:
            proj_seq = self.perceiver(proj_seq)
        if self.mrl:
            pooled = self.mrl(pooled)
        return VisionEncoderOutput(pooled=pooled, sequence=proj_seq)


@ENCODER_REGISTRY.register("vision")
def build_vision_encoder(cfg: EncoderConfig) -> VisionEncoder:
    return VisionEncoder(cfg)
