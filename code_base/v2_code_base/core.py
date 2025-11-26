"""
core.py - Core components for multimodal alignment

Simplified, modular implementation with:
- Vision encoder wrapper (CLIP)
- Text encoder wrapper (for alignment targets)
- Simple MLP adapter (no Perceiver initially)
- Contrastive losses (CLIP + MRL)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    CLIPVisionModel,
    CLIPImageProcessor,
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
)

Tensor = torch.Tensor


# ============================================================
# Configuration
# ============================================================

@dataclass
class AlignmentConfig:
    """Configuration for alignment training."""
    # Model names
    vision_model_name: str = "openai/clip-vit-base-patch32"
    text_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"  # smaller for testing
    
    # Architecture
    d_align: int = 512  # alignment embedding dimension
    adapter_hidden_factor: float = 2.0
    dropout: float = 0.1
    
    # Training
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    num_epochs: int = 5
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    
    # Loss
    mrl_dims: Tuple[int, ...] = (128, 256, 512)
    mrl_temperature: float = 0.07
    clip_temperature: float = 0.07
    mrl_weight: float = 1.0
    clip_weight: float = 1.0
    
    # Misc
    seed: int = 42
    log_every: int = 50
    max_text_length: int = 128
    
    # Will be set dynamically
    d_vision: int = 0
    d_text: int = 0
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    dtype: torch.dtype = torch.float32


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ============================================================
# Vision Encoder (Frozen CLIP)
# ============================================================

class VisionEncoder(nn.Module):
    """Frozen CLIP vision encoder wrapper."""
    
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.device = device or get_device()
        self.dtype = dtype
        
        # Load CLIP vision model
        self.processor = CLIPImageProcessor.from_pretrained(model_name)
        self.model = CLIPVisionModel.from_pretrained(model_name)
        
        # Move and freeze
        self.model.to(self.device, dtype=self.dtype)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        
        self.hidden_size = self.model.config.hidden_size
        print(f"[VisionEncoder] Loaded {model_name}, hidden_size={self.hidden_size}")
    
    @torch.no_grad()
    def encode(self, images) -> Dict[str, Tensor]:
        """
        Encode images to patch-level features.
        
        Args:
            images: List of PIL images or preprocessed tensor (B, C, H, W)
        
        Returns:
            dict with:
                feats: (B, num_patches, hidden_size)
                pooled: (B, hidden_size) - CLS token
                mask: (B, num_patches) bool
        """
        if isinstance(images, list):
            inputs = self.processor(images=images, return_tensors="pt")
            pixel_values = inputs["pixel_values"]
        else:
            pixel_values = images
        
        pixel_values = pixel_values.to(self.device, dtype=self.dtype)
        
        outputs = self.model(pixel_values=pixel_values)
        
        # last_hidden_state includes CLS + patches
        hidden = outputs.last_hidden_state  # (B, 1 + num_patches, D)
        
        cls_token = hidden[:, 0, :]           # (B, D)
        patch_feats = hidden[:, 1:, :]        # (B, num_patches, D)
        
        B, T, D = patch_feats.shape
        mask = torch.ones(B, T, dtype=torch.bool, device=self.device)
        
        return {
            "feats": patch_feats,
            "pooled": cls_token,
            "mask": mask,
        }


# ============================================================
# Text Encoder (Frozen, for alignment targets)
# ============================================================

class TextEncoder(nn.Module):
    """Frozen text encoder for alignment targets."""
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        max_length: int = 128,
    ):
        super().__init__()
        self.device = device or get_device()
        self.dtype = dtype
        self.max_length = max_length
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        # Move and freeze
        self.model.to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        
        self.hidden_size = self.model.config.hidden_size
        print(f"[TextEncoder] Loaded {model_name}, hidden_size={self.hidden_size}")
    
    @torch.no_grad()
    def encode(self, texts: List[str]) -> Tensor:
        """
        Encode texts to pooled embeddings.
        
        Args:
            texts: List of strings
        
        Returns:
            (B, hidden_size) tensor
        """
        tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)
        
        outputs = self.model(**tokens)
        
        # Mean pooling over non-padding tokens
        attention_mask = tokens["attention_mask"]
        hidden = outputs.last_hidden_state  # (B, T, D)
        
        mask = attention_mask.unsqueeze(-1).float()  # (B, T, 1)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        
        return pooled.to(self.dtype)


# ============================================================
# Simple MLP Adapter (Trainable)
# ============================================================

class MLPAdapter(nn.Module):
    """
    Simple 2-layer MLP adapter for projecting features.
    
    Can work on:
        - Sequences: (B, T, D_in) -> (B, T, D_out)
        - Pooled: (B, D_in) -> (B, D_out)
    """
    
    def __init__(
        self,
        d_in: int,
        d_out: int,
        hidden_factor: float = 2.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        hidden_dim = int(d_in * hidden_factor)
        
        self.net = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_out),
            nn.Dropout(dropout),
        )
    
    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


# ============================================================
# Contrastive Losses
# ============================================================

def l2_normalize(x: Tensor, eps: float = 1e-6) -> Tensor:
    """L2 normalize along last dimension."""
    return x / (x.norm(dim=-1, keepdim=True).clamp(min=eps))


def clip_contrastive_loss(
    z_a: Tensor,
    z_b: Tensor,
    temperature: float = 0.07,
) -> Tensor:
    """
    Symmetric CLIP-style InfoNCE loss.
    
    Args:
        z_a: (B, D) embeddings from modality A
        z_b: (B, D) embeddings from modality B
        temperature: softmax temperature
    
    Returns:
        scalar loss
    """
    z_a = l2_normalize(z_a)
    z_b = l2_normalize(z_b)
    
    logits = z_a @ z_b.T / temperature  # (B, B)
    B = z_a.size(0)
    targets = torch.arange(B, device=z_a.device)
    
    loss_ab = F.cross_entropy(logits, targets)
    loss_ba = F.cross_entropy(logits.T, targets)
    
    return 0.5 * (loss_ab + loss_ba)


def matryoshka_loss(
    z_a: Tensor,
    z_b: Tensor,
    radii: Tuple[int, ...],
    temperature: float = 0.07,
) -> Tensor:
    """
    Matryoshka Representation Learning loss.
    
    Computes CLIP loss at multiple embedding truncations.
    
    Args:
        z_a: (B, D) full embeddings from modality A
        z_b: (B, D) full embeddings from modality B
        radii: tuple of truncation dimensions
        temperature: softmax temperature
    
    Returns:
        scalar loss (averaged over radii)
    """
    D = z_a.size(-1)
    valid_radii = [r for r in radii if 0 < r <= D]
    
    if not valid_radii:
        return clip_contrastive_loss(z_a, z_b, temperature)
    
    losses = []
    for r in valid_radii:
        loss_r = clip_contrastive_loss(z_a[:, :r], z_b[:, :r], temperature)
        losses.append(loss_r)
    
    return sum(losses) / len(losses)


# ============================================================
# Alignment Model (Phase 1)
# ============================================================

class VisionTextAligner(nn.Module):
    """
    Phase 1: Simple vision-text alignment model.
    
    Architecture:
        Vision Encoder (frozen) -> MLP Adapter (trainable) -> z_vision
        Text Encoder (frozen) -> MLP Adapter (trainable) -> z_text
        
        Loss: MRL + CLIP contrastive
    """
    
    def __init__(self, cfg: AlignmentConfig):
        super().__init__()
        self.cfg = cfg
        
        # Frozen encoders
        self.vision_encoder = VisionEncoder(
            model_name=cfg.vision_model_name,
            device=cfg.device,
            dtype=cfg.dtype,
        )
        self.text_encoder = TextEncoder(
            model_name=cfg.text_model_name,
            device=cfg.device,
            dtype=cfg.dtype,
            max_length=cfg.max_text_length,
        )
        
        # Update config with actual dimensions
        cfg.d_vision = self.vision_encoder.hidden_size
        cfg.d_text = self.text_encoder.hidden_size
        
        # Trainable adapters
        self.vision_adapter = MLPAdapter(
            d_in=cfg.d_vision,
            d_out=cfg.d_align,
            hidden_factor=cfg.adapter_hidden_factor,
            dropout=cfg.dropout,
        ).to(cfg.device, dtype=cfg.dtype)
        
        self.text_adapter = MLPAdapter(
            d_in=cfg.d_text,
            d_out=cfg.d_align,
            hidden_factor=cfg.adapter_hidden_factor,
            dropout=cfg.dropout,
        ).to(cfg.device, dtype=cfg.dtype)
        
        print(f"[VisionTextAligner] d_vision={cfg.d_vision}, d_text={cfg.d_text}, d_align={cfg.d_align}")
    
    def encode_vision(self, images) -> Tensor:
        """Encode images to aligned embeddings."""
        enc_out = self.vision_encoder.encode(images)
        pooled = enc_out["pooled"]  # (B, d_vision)
        z_vision = self.vision_adapter(pooled)  # (B, d_align)
        return z_vision
    
    def encode_text(self, texts: List[str]) -> Tensor:
        """Encode texts to aligned embeddings."""
        pooled = self.text_encoder.encode(texts)  # (B, d_text)
        z_text = self.text_adapter(pooled)  # (B, d_align)
        return z_text
    
    def forward(
        self,
        images,
        texts: List[str],
    ) -> Dict[str, Any]:
        """
        Forward pass with loss computation.
        
        Returns:
            dict with loss, z_vision, z_text, metrics
        """
        z_vision = self.encode_vision(images)
        z_text = self.encode_text(texts)
        
        # Cast to float32 for stable loss computation
        z_v = z_vision.float()
        z_t = z_text.float()
        
        # Losses
        loss_mrl = matryoshka_loss(z_v, z_t, self.cfg.mrl_dims, self.cfg.mrl_temperature)
        loss_clip = clip_contrastive_loss(z_v, z_t, self.cfg.clip_temperature)
        
        loss = self.cfg.mrl_weight * loss_mrl + self.cfg.clip_weight * loss_clip
        
        return {
            "loss": loss,
            "loss_mrl": loss_mrl.item(),
            "loss_clip": loss_clip.item(),
            "z_vision": z_vision,
            "z_text": z_text,
        }
    
    def get_trainable_params(self):
        """Get parameters that require gradients."""
        params = []
        params.extend(self.vision_adapter.parameters())
        params.extend(self.text_adapter.parameters())
        return [p for p in params if p.requires_grad]


# ============================================================
# Retrieval Evaluation
# ============================================================

@torch.no_grad()
def compute_retrieval_metrics(
    z_a: Tensor,
    z_b: Tensor,
) -> Dict[str, float]:
    """
    Compute retrieval metrics (R@1, R@5, R@10).
    
    Args:
        z_a: (N, D) query embeddings
        z_b: (N, D) gallery embeddings (matched pairs)
    
    Returns:
        dict with recall@k metrics
    """
    z_a = l2_normalize(z_a)
    z_b = l2_normalize(z_b)
    
    sims = z_a @ z_b.T  # (N, N)
    N = sims.size(0)
    
    # For each query, find rank of correct match
    ranks = sims.argsort(dim=-1, descending=True)
    targets = torch.arange(N, device=ranks.device)
    
    # Find position of correct match for each query
    correct_positions = (ranks == targets.unsqueeze(1)).nonzero(as_tuple=True)[1]
    
    r1 = (correct_positions < 1).float().mean().item()
    r5 = (correct_positions < 5).float().mean().item()
    r10 = (correct_positions < 10).float().mean().item()
    
    return {
        "R@1": r1,
        "R@5": r5,
        "R@10": r10,
    }


# ============================================================
# Utility Functions
# ============================================================

def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count trainable and total parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
    }


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
