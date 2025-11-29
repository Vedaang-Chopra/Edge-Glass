"""Qwen decoder wrapper with optional LoRA."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from ..config import DecoderConfig
from ..utils.registry import DECODER_REGISTRY


class QwenDecoder(nn.Module):
    def __init__(self, cfg: DecoderConfig):
        super().__init__()
        self.cfg = cfg
        quantization = BitsAndBytesConfig(load_in_8bit=True) if torch.cuda.is_available() else None
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name, device_map="auto" if torch.cuda.is_available() else None, quantization_config=quantization
        )
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def forward(self, inputs, labels=None):
        outputs = self.model(**inputs, labels=labels)
        return outputs

    def generate(self, **kwargs):
        return self.model.generate(**kwargs)


@DECODER_REGISTRY.register("qwen")
def build_qwen_decoder(cfg: DecoderConfig) -> QwenDecoder:
    return QwenDecoder(cfg)
