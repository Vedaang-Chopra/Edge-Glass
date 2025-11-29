"\"\"\"Audio encoder (Whisper/AST) wrapper.\"\"\""

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
class AudioEncoderOutput:
    pooled: torch.Tensor
    sequence: torch.Tensor


class AudioEncoder(nn.Module):
    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.model = AutoModel.from_pretrained(cfg.model_name)
        self.processor = AutoProcessor.from_pretrained(cfg.model_name)
        self.projector = nn.Linear(self.model.config.hidden_size, cfg.projection_dim)
        self.perceiver = (
            PerceiverAdapter(cfg.projection_dim, cfg.perceiver_dim, cfg.perceiver_layers)
            if cfg.use_perceiver
            else None
        )
        self.mrl = MatryoshkaProjection(cfg.projection_dim, cfg.mrl_dimensions) if cfg.use_mrl else None
        for param in self.model.parameters():
            param.requires_grad = cfg.trainable

    def forward(self, audios: torch.Tensor) -> AudioEncoderOutput:
        inputs = self.processor(audios, sampling_rate=16000, return_tensors="pt").to(self.projector.weight.device)
        outputs = self.model(**inputs)
        pooled = self.projector(outputs.last_hidden_state[:, 0])
        seq = self.projector(outputs.last_hidden_state)
        if self.perceiver:
            seq = self.perceiver(seq)
        if self.mrl:
            pooled = self.mrl(pooled)
        return AudioEncoderOutput(pooled=pooled, sequence=seq)


@ENCODER_REGISTRY.register("audio")
def build_audio_encoder(cfg: EncoderConfig) -> AudioEncoder:
    return AudioEncoder(cfg)
