"""Alignment and instruction losses."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


class AlignmentLoss:
    def __init__(self, weights: Dict[str, float]):
        self.weights = weights

    def contrastive(self, vision, text):
        temp = 0.07
        vision = F.normalize(vision, dim=-1)
        text = F.normalize(text, dim=-1)
        logits = vision @ text.t() / temp
        labels = torch.arange(logits.size(0), device=logits.device)
        loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)) / 2
        return loss

    def mrl(self, embeddings):
        splits = torch.chunk(embeddings, chunks=3, dim=-1)
        losses = [split.norm(dim=-1).mean() for split in splits]
        return sum(losses) / len(losses)

    def __call__(self, outputs: Dict[str, torch.Tensor]):
        loss = 0.0
        if "contrastive" in self.weights:
            loss = loss + self.weights["contrastive"] * self.contrastive(outputs["vision"], outputs["text"])
        if "mrl" in self.weights and "mrl_embeddings" in outputs:
            loss = loss + self.weights["mrl"] * self.mrl(outputs["mrl_embeddings"])
        return loss
