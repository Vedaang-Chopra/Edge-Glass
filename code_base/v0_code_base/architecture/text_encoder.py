"""
text_encoder.py

OOP wrapper for a text encoder (MiniLM / BERT / sentence-transformer style).
"""

from typing import List, Optional

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

from .base import ModelConfig, BaseEncoder


class TextEncoder(BaseEncoder):
    """
    Generic text encoder using Hugging Face transformer models.

    Examples:
        cfg = ModelConfig("sentence-transformers/all-MiniLM-L6-v2")
        enc = TextEncoder(cfg)
        emb = enc.encode(["hello world", "this is a test"])
    """

    def __init__(
        self,
        cfg: ModelConfig,
        pooling: str = "mean",
        max_length: int = 256,
    ):
        super().__init__(cfg)

        self.pooling = pooling
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        self.model = AutoModel.from_pretrained(
            cfg.model_name,
            torch_dtype=cfg.dtype,
        ).to(self.device)

        # Enable hidden states for analysis / MRL-like things if needed
        if hasattr(self.model.config, "output_hidden_states"):
            self.model.config.output_hidden_states = True

        # Freeze encoder (Phase 1)
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

    @staticmethod
    def _mean_pool(
        last_hidden_state: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Standard mean pooling with attention mask, like in sentence-transformers.
        last_hidden_state: (B, T, D)
        attention_mask: (B, T)
        returns: (B, D)
        """
        mask = attention_mask.unsqueeze(-1)  # (B, T, 1)
        masked = last_hidden_state * mask
        summed = masked.sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1)
        return summed / counts

    @torch.no_grad()
    def encode_texts(self, texts: List[str], batch_size: int = 32) -> torch.Tensor:
        """
        Encode a list of texts into embeddings.

        Returns: (N, D) tensor on self.device.
        """
        all_embs = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]

            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)

            outputs = self.model(
                **inputs,
            )
            last_hidden = outputs.last_hidden_state  # (B, T, D)

            if self.pooling == "cls":
                # Many BERT-like models use CLS at position 0
                emb = last_hidden[:, 0, :]
            else:
                # Default: mean pooling with attention mask
                emb = self._mean_pool(last_hidden, inputs["attention_mask"])

            all_embs.append(emb)

        return torch.cat(all_embs, dim=0)  # (N, D)

    def encode(self, texts: List[str]) -> torch.Tensor:
        return self.encode_texts(texts)


def load_text_encoder(
    model_name: str,
    device: Optional[str] = None,
    dtype: torch.dtype = torch.float16,
    pooling: str = "mean",
    max_length: int = 256,
) -> TextEncoder:
    """
    Convenience loader for scripts / notebooks.
    """
    cfg = ModelConfig(model_name=model_name, device=device, dtype=dtype)
    return TextEncoder(cfg, pooling=pooling, max_length=max_length)
