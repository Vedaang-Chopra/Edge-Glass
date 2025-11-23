"""
image_encoder.py

OOP wrapper for an image encoder (CLIP / SigLIP-style) with
extra capabilities for Phase-0/1 POCs:

- Flexible feature strategies:
    * "auto"          → prefer image_embeds / pooler_output / mean-pool last_hidden_state
    * "image_embeds"  → use outputs.image_embeds if available
    * "last_hidden"   → mean/cls pooling over last_hidden_state
    * "layers_concat" → concatenate selected hidden layers, then pool

- Configurable pooling:
    * "mean"  → mean over spatial/sequence dimension
    * "cls"   → take first token (e.g. CLS)
    * "none"  → return token-level features (no pooling)

Usage (simple, same as before):

    cfg = ModelConfig("openai/clip-vit-base-patch32")
    enc = ImageEncoder(cfg)
    emb = enc.encode_pil(list_of_pil_images)      # (B, D)

Advanced usage:

    # concat 2nd and 2nd-to-last layer, then mean-pool
    enc = ImageEncoder(
        cfg,
        feature_strategy="layers_concat",
        layer_indices=[2, -2],
        pool="mean",
    )
    emb = enc.encode_pil(images)                  # (B, D_concat)

    # get token-level features (no pooling)
    tokens = enc.encode_pil(images, pooled=False)  # (B, T, D or D_concat)
"""

from typing import List, Optional, Literal

import torch
from transformers import AutoProcessor, AutoModel

from .base import ModelConfig, BaseEncoder

try:
    from PIL import Image
except ImportError:
    Image = None  # type: ignore


FeatureStrategy = Literal[
    "auto",
    "image_embeds",
    "last_hidden",
    "layers_concat",
]

PoolStrategy = Literal[
    "mean",
    "cls",
    "none",
]


class ImageEncoder(BaseEncoder):
    """
    Generic image encoder using Hugging Face vision(-text) models
    such as CLIP / SigLIP.

    The encoder is frozen (Phase 1: frozen vision encoder) and returns
    embeddings suitable for downstream alignment / Perceiver / projector.

    Attributes
    ----------
    feature_strategy : FeatureStrategy
        How to derive features from model outputs.
    layer_indices : List[int]
        Layer indices (into hidden_states) used when feature_strategy=="layers_concat".
    pool : PoolStrategy
        Pooling strategy over the sequence/patch dimension.
    """

    def __init__(
        self,
        cfg: ModelConfig,
        feature_strategy: FeatureStrategy = "auto",
        layer_indices: Optional[List[int]] = None,
        pool: PoolStrategy = "mean",
    ):
        """
        Parameters
        ----------
        cfg : ModelConfig
            Per-model configuration (name, device, dtype).
        feature_strategy : {"auto", "image_embeds", "last_hidden", "layers_concat"}
            Feature extraction strategy (see docstring above).
        layer_indices : list of int, optional
            Indices of hidden layers to use when concatenating hidden states.
            For example: [2, -2] (2nd layer and 2nd-to-last layer).
        pool : {"mean", "cls", "none"}
            Pooling strategy across tokens/patches.
        """
        super().__init__(cfg)

        self.processor = AutoProcessor.from_pretrained(cfg.model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(
            cfg.model_name, dtype=cfg.dtype
        ).to(self.device)

        # Enable hidden states for multi-layer feature strategies later
        if hasattr(self.model.config, "output_hidden_states"):
            self.model.config.output_hidden_states = True

        # Freeze encoder (Phase 1: frozen vision encoder)
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

        self.feature_strategy: FeatureStrategy = feature_strategy
        self.layer_indices: List[int] = layer_indices or []
        self.pool: PoolStrategy = pool

    # ---------------------------
    # Internal helpers
    # ---------------------------

    def _pool_sequence(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pool over the sequence/patch dimension according to self.pool.

        x: (B, T, D)
        returns: (B, D) if pool != "none", else (B, T, D)
        """
        if self.pool == "mean":
            return x.mean(dim=1)
        elif self.pool == "cls":
            # Take first token (e.g., CLS or global token)
            return x[:, 0, :]
        elif self.pool == "none":
            return x
        else:
            raise ValueError(f"Unknown pool strategy: {self.pool}")

    def _extract_features(
        self,
        outputs,
        pooled: bool,
    ) -> torch.Tensor:
        """
        Given model outputs, apply the configured feature strategy and pooling.
        """
        # 1) AUTO strategy: try best available things in a reasonable order.
        if self.feature_strategy == "auto":
            # Prefer dedicated image_embeds (CLIP-style)
            if hasattr(outputs, "image_embeds") and outputs.image_embeds is not None:
                feats = outputs.image_embeds  # (B, D)
                if pooled and feats.ndim == 2:
                    return feats
                elif not pooled and feats.ndim == 2:
                    # treat as sequence of length 1
                    return feats.unsqueeze(1)
                return feats

            # Some models have pooler_output
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                feats = outputs.pooler_output  # (B, D)
                if pooled and feats.ndim == 2:
                    return feats
                elif not pooled and feats.ndim == 2:
                    return feats.unsqueeze(1)
                return feats

            # Fallback: mean-pool last_hidden_state
            hidden = outputs.last_hidden_state  # (B, T, D)
            if pooled:
                return hidden.mean(dim=1)
            else:
                return hidden

        # 2) image_embeds only
        if self.feature_strategy == "image_embeds":
            if not hasattr(outputs, "image_embeds"):
                raise ValueError(
                    "feature_strategy='image_embeds' but outputs.image_embeds not present."
                )
            feats = outputs.image_embeds  # (B, D)
            if pooled:
                return feats
            else:
                # treat as length-1 sequence
                return feats.unsqueeze(1)

        # 3) last_hidden: sequence of patch tokens
        if self.feature_strategy == "last_hidden":
            hidden = outputs.last_hidden_state  # (B, T, D)
            if pooled:
                return self._pool_sequence(hidden)
            else:
                return hidden

        # 4) layers_concat: concat chosen hidden layers along channel dim
        if self.feature_strategy == "layers_concat":
            if not hasattr(outputs, "hidden_states") or outputs.hidden_states is None:
                raise ValueError(
                    "feature_strategy='layers_concat' requires outputs.hidden_states=True."
                )
            if not self.layer_indices:
                raise ValueError(
                    "feature_strategy='layers_concat' but no layer_indices provided."
                )

            hidden_states = outputs.hidden_states  # tuple of (layer, B, T, D)
            # Collect requested layers (support negative indices).
            layers_to_cat = []
            for idx in self.layer_indices:
                layers_to_cat.append(hidden_states[idx])  # (B, T, D_i)

            # Concatenate along the feature dimension
            concat_hidden = torch.cat(layers_to_cat, dim=-1)  # (B, T, sum(D_i))

            if pooled:
                return self._pool_sequence(concat_hidden)
            else:
                return concat_hidden

        raise ValueError(f"Unknown feature strategy: {self.feature_strategy}")

    # ---------------------------
    # Public API
    # ---------------------------

    @torch.no_grad()
    def encode_pil(
        self,
        images: List["Image.Image"],
        pooled: bool = True,
    ) -> torch.Tensor:
        """
        Encode a list of PIL images into embeddings.

        Parameters
        ----------
        images : list of PIL.Image.Image
            Input images.
        pooled : bool, default True
            If True, returns (B, D) or (B, D_concat) embeddings.
            If False, returns token-level features (B, T, D or D_concat).

        Returns
        -------
        torch.Tensor
            Image features according to feature_strategy & pooling.
        """
        if Image is None:
            raise ImportError("PIL is required for encode_pil; install pillow.")

        inputs = self.processor(
            images=images,
            return_tensors="pt",
        ).to(self.device)

        outputs = self.model(**inputs)

        feats = self._extract_features(outputs, pooled=pooled)
        return feats

    def encode(
        self,
        images: List["Image.Image"],
        pooled: bool = True,
    ) -> torch.Tensor:
        """
        Thin wrapper over encode_pil to satisfy BaseEncoder interface.
        """
        return self.encode_pil(images, pooled=pooled)


def load_image_encoder(
    model_name: str,
    device: Optional[str] = None,
    dtype: torch.dtype = torch.float16,
    feature_strategy: FeatureStrategy = "auto",
    layer_indices: Optional[List[int]] = None,
    pool: PoolStrategy = "mean",
) -> ImageEncoder:
    """
    Convenience loader for scripts / notebooks.

    This is what you'll typically call from your main IPYNB:

        from models.image_encoder import load_image_encoder

        img_enc = load_image_encoder(
            cfg.models.vision_model_name,
            device=str(cfg.torch_device),
            dtype=cfg.torch_dtype,
            feature_strategy="layers_concat",
            layer_indices=[2, -2],
            pool="mean",
        )
    """
    cfg = ModelConfig(model_name=model_name, device=device, dtype=dtype)
    return ImageEncoder(
        cfg,
        feature_strategy=feature_strategy,
        layer_indices=layer_indices,
        pool=pool,
    )
