"""Tiny Recursive Model decoder for ablation experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import DecoderConfig
from ..utils.registry import DECODER_REGISTRY


@dataclass
class TRMConfig:
    vocab_size: int = 32000
    hidden_size: int = 512
    num_heads: int = 8
    num_layers: int = 2
    max_seq_len: int = 512
    dropout: float = 0.1
    n_recursions: int = 6
    t_cycles: int = 3


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_x = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm_x * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len: int, device=None):
        positions = torch.arange(seq_len, device=device).float()
        freqs = torch.einsum("i,j->ij", positions, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()


def apply_rotary(q, k, cos, sin):
    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)
    q1, q2 = q.chunk(2, dim=-1)
    k1, k2 = k.chunk(2, dim=-1)
    q = torch.cat((q1 * cos - q2 * sin, q2 * cos + q1 * sin), dim=-1)
    k = torch.cat((k1 * cos - k2 * sin, k2 * cos + k1 * sin), dim=-1)
    return q, k


class TRMDecoder(nn.Module):
    def __init__(self, cfg: TRMConfig):
        super().__init__()
        self.cfg = cfg
        self.token_embed = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.pos_embed = RotaryEmbedding(cfg.hidden_size // cfg.num_heads * 2)
        self.blocks = nn.ModuleList(
            [nn.TransformerEncoderLayer(cfg.hidden_size, cfg.num_heads, 4 * cfg.hidden_size, cfg.dropout) for _ in range(cfg.num_layers)]
        )
        self.norm = RMSNorm(cfg.hidden_size)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

    def forward(self, input_ids, labels=None, past_key_values: Optional[torch.Tensor] = None):
        hidden = self.token_embed(input_ids)
        cos, sin = self.pos_embed(hidden.size(1), device=hidden.device)
        for block in self.blocks:
            q = k = hidden
            q, k = apply_rotary(q, k, cos, sin)
            hidden = block(hidden)
        hidden = self.norm(hidden)
        logits = self.lm_head(hidden)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
        return {"logits": logits, "loss": loss}


@DECODER_REGISTRY.register("trm")
def build_trm_decoder(cfg: DecoderConfig) -> TRMDecoder:
    trm_cfg = TRMConfig(
        vocab_size=cfg.max_seq_len,
        hidden_size=cfg.max_seq_len // 2,
    )
    return TRMDecoder(trm_cfg)
