"""
text_encoder.py

Thin wrapper around a Hugging Face text encoder for Stage-1 alignment.

We keep it very generic: any HF model that returns last_hidden_state
can be used (BERT, MiniLM, etc.). We pool over the CLS token by default.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel


@dataclass
class HFTextEncoderConfig:
    model_name: str
    max_length: int = 128
    trainable: bool = False   # Stage-1: usually keep frozen


class HFTextEncoder(nn.Module):
    """
    Simple wrapper around a Hugging Face text encoder.

    Usage:
        cfg = HFTextEncoderConfig(model_name="sentence-transformers/all-MiniLM-L6-v2")
        txt_enc = HFTextEncoder(cfg, device=device, dtype=dtype)

        z = txt_enc.encode(["a caption", "another caption"])  # (B, D_text)
    """

    def __init__(
        self,
        cfg: HFTextEncoderConfig,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.dtype = dtype

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        self.model = AutoModel.from_pretrained(cfg.model_name)

        self.model.to(device)
        self.model.eval()

        if not cfg.trainable:
            for p in self.model.parameters():
                p.requires_grad = False

        # Expose hidden size for alignment heads
        self.hidden_size = self.model.config.hidden_size

    @torch.no_grad()
    def encode(self, texts: List[str]) -> torch.Tensor:
        """
        Encode a list of texts into pooled embeddings.

        Returns:
            (B, D_text) tensor on self.device with dtype self.dtype.
        """
        if len(texts) == 0:
            raise ValueError("encode() called with empty texts list")

        toks = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.cfg.max_length,
            return_tensors="pt",
        ).to(self.device)

        outputs = self.model(**toks)  # last_hidden_state: (B, T, D)
        last_hidden = outputs.last_hidden_state

        # CLS pooling: take first token
        pooled = last_hidden[:, 0, :]  # (B, D_text)
        return pooled.to(self.dtype)
