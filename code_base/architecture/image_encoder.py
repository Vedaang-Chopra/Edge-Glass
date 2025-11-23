"""
image_encoder.py

OOP wrapper for an image encoder (CLIP / SigLIP-style).
"""

from typing import List, Optional

import torch
from transformers import AutoProcessor, AutoModel

from .base import ModelConfig, BaseEncoder

try:
    from PIL import Image
except ImportError:
    Image = None  # type: ignore


class ImageEncoder(BaseEncoder):
    """
    Generic image encoder using Hugging Face vision-text models
    such as CLIP / SigLIP.

    Usage:
        cfg = ModelConfig("openai/clip-vit-base-patch32")
        enc = ImageEncoder(cfg)
        emb = enc.encode_pil(list_of_pil_images)
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)

        self.processor = AutoProcessor.from_pretrained(cfg.model_name)
        self.model = AutoModel.from_pretrained(
            cfg.model_name, torch_dtype=cfg.dtype
        ).to(self.device)

        # Enable hidden states for multi-layer feature strategies later
        if hasattr(self.model.config, "output_hidden_states"):
            self.model.config.output_hidden_states = True

        # Freeze encoder (Phase 1: frozen vision encoder)
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

    @torch.no_grad()
    def encode_pil(self, images: List["Image.Image"]) -> torch.Tensor:
        """
        Encode a list of PIL images into embeddings.

        Returns: (B, D) embeddings. If the model exposes `image_embeds`,
        we use that (CLIP-style). Otherwise we mean-pool last_hidden_state.
        """
        if Image is None:
            raise ImportError("PIL is required for encode_pil; install pillow.")

        inputs = self.processor(
            images=images,
            return_tensors="pt",
        ).to(self.device)

        outputs = self.model(**inputs)

        if hasattr(outputs, "image_embeds"):
            img_emb = outputs.image_embeds  # (B, D)
        else:
            # Generic fallback: mean-pool patch tokens
            img_emb = outputs.last_hidden_state.mean(dim=1)

        return img_emb

    def encode(self, images: List["Image.Image"]) -> torch.Tensor:
        return self.encode_pil(images)


def load_image_encoder(
    model_name: str,
    device: Optional[str] = None,
    dtype: torch.dtype = torch.float16,
) -> ImageEncoder:
    """
    Convenience loader for scripts / notebooks.
    """
    cfg = ModelConfig(model_name=model_name, device=device, dtype=dtype)
    return ImageEncoder(cfg)
