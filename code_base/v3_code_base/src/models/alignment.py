"""High-level multimodal alignment module."""

from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import nn

from ..config import ExperimentConfig
from ..encoders.audio import AudioEncoder
from ..encoders.text import TextEncoder
from ..encoders.vision import VisionEncoder
from ..models.losses import AlignmentLoss
from ..utils.registry import DECODER_REGISTRY, ENCODER_REGISTRY


class MultimodalAlignmentModel(nn.Module):
    def __init__(self, cfg: ExperimentConfig):
        super().__init__()
        self.cfg = cfg
        self.vision_encoder: Optional[VisionEncoder] = None
        self.audio_encoder: Optional[AudioEncoder] = None
        if cfg.vision_encoder:
            self.vision_encoder = ENCODER_REGISTRY.get("vision")(cfg.vision_encoder)
        if cfg.audio_encoder:
            self.audio_encoder = ENCODER_REGISTRY.get("audio")(cfg.audio_encoder)
        self.text_encoder = ENCODER_REGISTRY.get("text")(cfg.text_encoder)
        self.loss_fn = AlignmentLoss(cfg.losses)
        self.decoder = None
        if cfg.decoder:
            builder = DECODER_REGISTRY.get(cfg.decoder.type)
            self.decoder = builder(cfg.decoder)

    def encode_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        outputs: Dict[str, torch.Tensor] = {}
        if self.vision_encoder and "image" in batch:
            v = self.vision_encoder(batch["image"])
            outputs["vision"] = v.pooled
            if self.cfg.losses.get("mrl"):
                outputs["mrl_embeddings"] = v.pooled
        if self.audio_encoder and "audio" in batch:
            a = self.audio_encoder(batch["audio"])
            outputs["audio"] = a.pooled
        if "text" in batch:
            t = self.text_encoder(batch["text"])
            outputs["text"] = t.pooled
        return outputs

    def forward(self, batch: Dict[str, torch.Tensor]):
        embeddings = self.encode_batch(batch)
        loss = self.loss_fn(embeddings)
        outputs = {"loss": loss, **embeddings}
        if self.decoder and "instruction" in batch:
            outputs["decoder"] = self.decoder(batch["instruction"])
        return outputs
