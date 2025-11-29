"""Text encoder wrapping sentence-transformers / LLM encoders."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

from ..config import EncoderConfig
from ..utils.registry import ENCODER_REGISTRY
from .mrl import MatryoshkaProjection


@dataclass
class TextEncoderOutput:
    pooled: torch.Tensor
    sequence: torch.Tensor


class TextEncoder(nn.Module):
    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.model = AutoModel.from_pretrained(cfg.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
        self.projector = nn.Linear(self.model.config.hidden_size, cfg.projection_dim)
        self.mrl = MatryoshkaProjection(cfg.projection_dim, cfg.mrl_dimensions) if cfg.use_mrl else None
        for param in self.model.parameters():
            param.requires_grad = cfg.trainable

    def forward(self, input_texts):
        tokens = self.tokenizer(
            input_texts, padding=True, truncation=True, max_length=512, return_tensors="pt"
        ).to(self.projector.weight.device)
        outputs = self.model(**tokens)
        pooled = outputs.pooler_output
        seq = self.projector(outputs.last_hidden_state)
        pooled = self.projector(pooled)
        if self.mrl:
            pooled = self.mrl(pooled)
        return TextEncoderOutput(pooled=pooled, sequence=seq)


@ENCODER_REGISTRY.register("text")
def build_text_encoder(cfg: EncoderConfig) -> TextEncoder:
    return TextEncoder(cfg)
