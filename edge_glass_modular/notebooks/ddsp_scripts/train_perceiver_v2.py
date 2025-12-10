#!/usr/bin/env python3
"""
================================================================================
DDSP PERCEIVER ALIGNMENT TRAINING SCRIPT (V2)
================================================================================

A complete standalone training script for multimodal alignment with Perceiver
Resampler, based on the working 02_alig_multi_perciever.ipynb notebook.

Features:
- Frozen encoders (CLIP, MiniLM) for feature extraction
- Trainable Perceiver Resampler shared across modalities
- Trainable MLP adapters for modality projection
- MRL (Matryoshka Representation Learning) and CLIP-style contrastive loss
- WandB integration for experiment tracking
- DDP support for distributed training
- Checkpoint saving with rotation (best + latest only)
- Resume from checkpoint support
- Retrieval metrics (R@1, R@5, R@10)

Usage:
    python train_perceiver_v2.py                        # Single GPU
    python train_perceiver_v2.py --resume auto          # Resume training
    torchrun --nproc_per_node=4 train_perceiver_v2.py   # Multi-GPU

================================================================================
"""

import os
import sys
import math
import random
import argparse
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

# Add paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
SRC_PATH = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_PATH))
sys.path.insert(0, str(PROJECT_ROOT))

# Try imports with fallback
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    print("Warning: wandb not available. Logging disabled.")

try:
    from transformers import (
        CLIPImageProcessor,
        CLIPVisionModel,
        AutoTokenizer,
        AutoModel,
    )
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("Error: transformers not available.")
    sys.exit(1)


# ============================================================================
# CONFIGURATION (Optimized hyperparameters)
# ============================================================================

@dataclass
class MultimodalAlignmentConfig:
    """Configuration for multimodal alignment with Perceiver."""
    
    # === Model Names ===
    vision_model_name: str = "openai/clip-vit-large-patch14-336"
    audio_model_name: str = "openai/whisper-base"
    text_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # === Encoder Dimensions ===
    d_vision: int = 1024      # CLIP ViT-L/14 hidden size
    d_audio: int = 512        # Whisper-base encoder hidden size
    d_text: int = 384         # MiniLM hidden size
    
    # === Perceiver Architecture ===
    perceiver_dim: int = 512          # Internal dimension
    num_latents: int = 64             # Number of learned queries
    num_perceiver_layers: int = 4     # Depth
    num_attn_heads: int = 8           # Attention heads
    perceiver_mlp_ratio: float = 4.0  # FFN expansion ratio
    perceiver_dropout: float = 0.1    # Dropout rate
    
    # === Alignment ===
    d_align: int = 512                        # Final alignment dimension
    mrl_dims: Tuple[int, ...] = (64, 128, 256, 512)  # Matryoshka radii
    
    # === Training (Optimized) ===
    batch_size: int = 64              # Optimal batch size
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    num_epochs: int = 10
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    
    # === Contrastive Loss (Optimized) ===
    temperature: float = 0.07
    mrl_weight: float = 1.0
    clip_weight: float = 0.25         # Reduced from 0.5 per config
    
    # === Data ===
    image_size: int = 336
    max_text_length: int = 77
    train_parquet: str = ""
    val_parquet: str = ""
    
    # === Misc ===
    seed: int = 42
    device: str = "cuda"
    output_dir: str = "./outputs/perceiver_v2"
    ckpt_dir: str = "./checkpoints/perceiver_v2"
    log_every: int = 50
    eval_every_epoch: int = 1
    num_workers: int = 8
    use_wandb: bool = True
    wandb_project: str = "edge_glass_alignment"
    wandb_run_name: str = "perceiver_v2_alignment"


# ============================================================================
# PERCEIVER RESAMPLER COMPONENTS
# ============================================================================

class FeedForward(nn.Module):
    """Feed-Forward Network with GELU activation."""
    
    def __init__(self, dim: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PerceiverAttention(nn.Module):
    """Multi-Head Attention for Perceiver with Pre-LayerNorm."""
    
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.ln_q = nn.LayerNorm(dim)
        self.ln_kv = nn.LayerNorm(dim)
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        q: torch.Tensor,
        kv: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N_q, D = q.shape
        N_kv = kv.shape[1]
        
        q = self.ln_q(q)
        kv = self.ln_kv(kv)
        
        Q = self.q_proj(q).view(B, N_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(kv).view(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(kv).view(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn = (Q @ K.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(~mask.bool(), float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = (attn @ V).transpose(1, 2).reshape(B, N_q, D)
        out = self.out_proj(out)
        
        return out


class PerceiverLayer(nn.Module):
    """Single Perceiver Layer with cross-attention, self-attention, and FFN."""
    
    def __init__(
        self, 
        dim: int, 
        num_heads: int = 8, 
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.cross_attn = PerceiverAttention(dim, num_heads, dropout)
        self.self_attn = PerceiverAttention(dim, num_heads, dropout)
        self.ffn = FeedForward(dim, mlp_ratio, dropout)
        self.ln_ffn = nn.LayerNorm(dim)
    
    def forward(
        self,
        latents: torch.Tensor,
        tokens: torch.Tensor,
        token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        latents = latents + self.cross_attn(latents, tokens, token_mask)
        latents = latents + self.self_attn(latents, latents)
        latents = latents + self.ffn(self.ln_ffn(latents))
        return latents


class PerceiverResampler(nn.Module):
    """Perceiver Resampler for variable-length sequence compression."""
    
    def __init__(
        self,
        dim: int,
        num_latents: int = 64,
        num_layers: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.num_latents = num_latents
        
        self.latents = nn.Parameter(
            torch.randn(num_latents, dim) * (dim ** -0.5)
        )
        
        self.layers = nn.ModuleList([
            PerceiverLayer(dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        
        self.ln_out = nn.LayerNorm(dim)
    
    def forward(
        self,
        tokens: torch.Tensor,
        token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B = tokens.shape[0]
        latents = self.latents.unsqueeze(0).expand(B, -1, -1)
        
        for layer in self.layers:
            latents = layer(latents, tokens, token_mask)
        
        return self.ln_out(latents)


# ============================================================================
# MODALITY ADAPTERS
# ============================================================================

class MLPAdapter(nn.Module):
    """MLP Adapter with LayerNorm and GELU."""
    
    def __init__(
        self, 
        in_dim: int, 
        out_dim: int, 
        hidden_factor: float = 2.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        hidden_dim = int(in_dim * hidden_factor)
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AlignmentProjector(nn.Module):
    """Projects Perceiver latents to alignment space."""
    
    def __init__(self, input_dim: int, output_dim: int, pooling: str = 'mean'):
        super().__init__()
        self.pooling = pooling
        
        if pooling == 'attention':
            self.attn_pool = nn.Sequential(
                nn.Linear(input_dim, 1),
                nn.Softmax(dim=1),
            )
        
        self.proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, output_dim),
        )
    
    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        if self.pooling == 'mean':
            pooled = latents.mean(dim=1)
        elif self.pooling == 'first':
            pooled = latents[:, 0]
        elif self.pooling == 'attention':
            weights = self.attn_pool(latents)
            pooled = (latents * weights).sum(dim=1)
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")
        
        return self.proj(pooled)


# ============================================================================
# COMPLETE ALIGNMENT MODEL
# ============================================================================

class MultimodalAlignmentModel(nn.Module):
    """Complete Multimodal Alignment Model with Perceiver Resampler."""
    
    def __init__(self, config: MultimodalAlignmentConfig):
        super().__init__()
        self.config = config
        
        self.vision_adapter = MLPAdapter(
            config.d_vision, 
            config.perceiver_dim,
            hidden_factor=2.0,
            dropout=config.perceiver_dropout,
        )
        
        self.audio_adapter = MLPAdapter(
            config.d_audio,
            config.perceiver_dim,
            hidden_factor=2.0,
            dropout=config.perceiver_dropout,
        )
        
        self.text_adapter = MLPAdapter(
            config.d_text,
            config.perceiver_dim,
            hidden_factor=2.0,
            dropout=config.perceiver_dropout,
        )
        
        self.perceiver = PerceiverResampler(
            dim=config.perceiver_dim,
            num_latents=config.num_latents,
            num_layers=config.num_perceiver_layers,
            num_heads=config.num_attn_heads,
            mlp_ratio=config.perceiver_mlp_ratio,
            dropout=config.perceiver_dropout,
        )
        
        self.alignment_projector = AlignmentProjector(
            config.perceiver_dim,
            config.d_align,
            pooling='mean',
        )
    
    def _encode_modality(
        self,
        features: torch.Tensor,
        adapter: nn.Module,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens = adapter(features)
        latents = self.perceiver(tokens, mask)
        z_aligned = self.alignment_projector(latents)
        return latents, z_aligned
    
    def encode_vision(
        self,
        features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        _, z = self._encode_modality(features, self.vision_adapter, mask)
        return z
    
    def encode_audio(
        self,
        features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        _, z = self._encode_modality(features, self.audio_adapter, mask)
        return z
    
    def encode_text(
        self,
        features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        _, z = self._encode_modality(features, self.text_adapter, mask)
        return z
    
    def forward(
        self,
        vision_features: Optional[torch.Tensor] = None,
        audio_features: Optional[torch.Tensor] = None,
        text_features: Optional[torch.Tensor] = None,
        vision_mask: Optional[torch.Tensor] = None,
        audio_mask: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        outputs = {}
        
        if vision_features is not None:
            outputs['z_vision'] = self.encode_vision(vision_features, vision_mask)
        
        if audio_features is not None:
            outputs['z_audio'] = self.encode_audio(audio_features, audio_mask)
        
        if text_features is not None:
            outputs['z_text'] = self.encode_text(text_features, text_mask)
        
        return outputs


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

def contrastive_loss(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Symmetric InfoNCE contrastive loss (CLIP-style)."""
    z_a = F.normalize(z_a, dim=-1)
    z_b = F.normalize(z_b, dim=-1)
    
    logits = z_a @ z_b.T / temperature
    labels = torch.arange(z_a.size(0), device=z_a.device)
    
    loss_a2b = F.cross_entropy(logits, labels)
    loss_b2a = F.cross_entropy(logits.T, labels)
    
    return (loss_a2b + loss_b2a) / 2


def matryoshka_loss(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    dims: Tuple[int, ...] = (64, 128, 256, 512),
    temperature: float = 0.07,
) -> torch.Tensor:
    """Matryoshka Representation Learning (MRL) loss."""
    total_loss = 0.0
    num_dims = 0
    
    for dim in dims:
        if dim > z_a.size(-1):
            continue
        
        z_a_trunc = z_a[:, :dim]
        z_b_trunc = z_b[:, :dim]
        
        loss = contrastive_loss(z_a_trunc, z_b_trunc, temperature)
        total_loss += loss
        num_dims += 1
    
    return total_loss / num_dims if num_dims > 0 else total_loss


# ============================================================================
# DATA LOADING - Using modular dataset builders
# ============================================================================

try:
    from data.dataset_builder import build_image_datasets_from_parquet
    from data.transforms import get_image_transforms
    HAS_MODULAR_DATA = True
except ImportError:
    HAS_MODULAR_DATA = False
    print("Warning: Could not import modular data loaders.")


def collate_fn(batch: List[Dict]) -> Dict[str, Any]:
    """Collate function for image-text batches."""
    images = torch.stack([item['image'] for item in batch])
    texts = [item['text'] for item in batch]
    
    return {
        'images': images,
        'captions': texts,
    }


def build_train_val_loaders(
    config: MultimodalAlignmentConfig,
    is_distributed: bool = False,
):
    """Build train and validation data loaders using modular dataset builders."""
    
    if not HAS_MODULAR_DATA:
        raise ImportError("Modular data loaders required. Ensure src/ is in path.")
    
    print("Using modular data loaders from src/data/")
    
    train_transforms = get_image_transforms(
        image_size=config.image_size, 
        is_training=True
    )
    val_transforms = get_image_transforms(
        image_size=config.image_size, 
        is_training=False
    )
    
    datasets = build_image_datasets_from_parquet(
        cfg=config,
        train_parquet_path=config.train_parquet,
        val_parquet_path=config.val_parquet,
        train_transforms=train_transforms,
        val_transforms=val_transforms,
        max_text_length=config.max_text_length,
        text_dropout_prob=0.1,
    )
    
    train_dataset = datasets['train']
    val_dataset = datasets['val']
    
    train_sampler = None
    shuffle = True
    if is_distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, shuffle=True
        )
        shuffle = False
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=config.num_workers > 0,
        drop_last=True,
    )
    
    val_sampler = None
    if is_distributed:
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset, shuffle=False
        )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Val dataset: {len(val_dataset)} samples")
    
    return train_loader, val_loader


# ============================================================================
# TRAINING UTILITIES
# ============================================================================

def get_rank():
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def is_main_process():
    return get_rank() == 0


def count_parameters(model: nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'total': total, 'trainable': trainable, 'frozen': total - trainable}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================================
# TRAINER
# ============================================================================

class PerceiverTrainer:
    """Trainer for multimodal alignment with Perceiver."""
    
    def __init__(
        self,
        config: MultimodalAlignmentConfig,
        model: MultimodalAlignmentModel,
        clip_processor: CLIPImageProcessor,
        clip_vision: CLIPVisionModel,
        text_tokenizer: AutoTokenizer,
        text_encoder: AutoModel,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        device: torch.device = torch.device("cuda"),
    ):
        self.config = config
        self.model = model
        self.clip_processor = clip_processor
        self.clip_vision = clip_vision
        self.text_tokenizer = text_tokenizer
        self.text_encoder = text_encoder
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        
        total_steps = len(train_loader) * config.num_epochs
        
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config.learning_rate,
            total_steps=total_steps,
            pct_start=config.warmup_ratio,
            anneal_strategy='cos',
        )
        
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        self.wandb_run = None
        if config.use_wandb and HAS_WANDB and is_main_process():
            self.wandb_run = wandb.init(
                project=config.wandb_project,
                name=config.wandb_run_name,
                config=asdict(config),
            )
    
    @torch.no_grad()
    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images to CLIP features."""
        images = images.to(self.device)
        outputs = self.clip_vision(pixel_values=images)
        return outputs.last_hidden_state
    
    @torch.no_grad()
    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        """Encode texts to features."""
        tokens = self.text_tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.config.max_text_length,
            return_tensors="pt",
        ).to(self.device)
        outputs = self.text_encoder(**tokens)
        return outputs.last_hidden_state
    
    def train_step(self, batch: Dict) -> Dict[str, float]:
        """Single training step."""
        self.model.train()
        
        vision_features = self.encode_images(batch['images'])
        text_features = self.encode_texts(batch['captions'])
        
        z_vision = self.model.encode_vision(vision_features)
        z_text = self.model.encode_text(text_features)
        
        loss_clip = contrastive_loss(z_vision, z_text, self.config.temperature)
        loss_mrl = matryoshka_loss(z_vision, z_text, self.config.mrl_dims, self.config.temperature)
        
        total_loss = self.config.clip_weight * loss_clip + self.config.mrl_weight * loss_mrl
        
        self.optimizer.zero_grad()
        total_loss.backward()
        
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), 
            self.config.max_grad_norm
        )
        
        self.optimizer.step()
        self.scheduler.step()
        
        return {
            'loss': total_loss.item(),
            'loss_clip': loss_clip.item(),
            'loss_mrl': loss_mrl.item(),
            'grad_norm': grad_norm.item(),
            'lr': self.scheduler.get_last_lr()[0],
        }
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation with loss and retrieval metrics."""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        
        total_loss = 0.0
        total_loss_clip = 0.0
        total_loss_mrl = 0.0
        num_batches = 0
        
        all_z_vision = []
        all_z_text = []
        
        for batch in tqdm(self.val_loader, desc="Validating", disable=not is_main_process()):
            vision_features = self.encode_images(batch['images'])
            text_features = self.encode_texts(batch['captions'])
            
            z_vision = self.model.encode_vision(vision_features)
            z_text = self.model.encode_text(text_features)
            
            loss_clip = contrastive_loss(z_vision, z_text, self.config.temperature)
            loss_mrl = matryoshka_loss(z_vision, z_text, self.config.mrl_dims, self.config.temperature)
            loss = self.config.clip_weight * loss_clip + self.config.mrl_weight * loss_mrl
            
            total_loss += loss.item()
            total_loss_clip += loss_clip.item()
            total_loss_mrl += loss_mrl.item()
            num_batches += 1
            
            if len(all_z_vision) * z_vision.size(0) < 1000:
                all_z_vision.append(z_vision.cpu())
                all_z_text.append(z_text.cpu())
        
        metrics = {
            'val_loss': total_loss / num_batches,
            'val_loss_clip': total_loss_clip / num_batches,
            'val_loss_mrl': total_loss_mrl / num_batches,
        }
        
        if all_z_vision:
            z_v = torch.cat(all_z_vision, dim=0).to(self.device)
            z_t = torch.cat(all_z_text, dim=0).to(self.device)
            retrieval_metrics = self.compute_retrieval_metrics(z_v, z_t)
            metrics.update({f'val_{k}': v for k, v in retrieval_metrics.items()})
        
        return metrics
    
    @torch.no_grad()
    def compute_retrieval_metrics(self, z_vision: torch.Tensor, z_text: torch.Tensor) -> Dict[str, float]:
        """Compute retrieval recall metrics (R@1, R@5, R@10)."""
        z_vision = F.normalize(z_vision, dim=-1)
        z_text = F.normalize(z_text, dim=-1)
        
        sim_i2t = z_vision @ z_text.T
        
        B = z_vision.size(0)
        targets = torch.arange(B, device=z_vision.device)
        
        rankings_i2t = (sim_i2t.argsort(dim=-1, descending=True) == targets.unsqueeze(1)).nonzero(as_tuple=True)[1]
        rankings_t2i = (sim_i2t.T.argsort(dim=-1, descending=True) == targets.unsqueeze(1)).nonzero(as_tuple=True)[1]
        
        def recall_at_k(rankings, k):
            return (rankings < k).float().mean().item()
        
        metrics = {
            'i2t_r@1': recall_at_k(rankings_i2t, 1),
            'i2t_r@5': recall_at_k(rankings_i2t, 5),
            'i2t_r@10': recall_at_k(rankings_i2t, 10),
            't2i_r@1': recall_at_k(rankings_t2i, 1),
            't2i_r@5': recall_at_k(rankings_t2i, 5),
            't2i_r@10': recall_at_k(rankings_t2i, 10),
        }
        
        metrics['avg_r@1'] = (metrics['i2t_r@1'] + metrics['t2i_r@1']) / 2
        metrics['avg_r@5'] = (metrics['i2t_r@5'] + metrics['t2i_r@5']) / 2
        metrics['avg_r@10'] = (metrics['i2t_r@10'] + metrics['t2i_r@10']) / 2
        
        return metrics
    
    def save_checkpoint(self, tag: str = "latest", epoch: int = 0):
        """Save model checkpoint."""
        if not is_main_process():
            return
        
        ckpt_dir = Path(self.config.ckpt_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        
        model_state = self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict()
        
        checkpoint = {
            'model_state_dict': model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': asdict(self.config),
            'global_step': self.global_step,
            'epoch': epoch,
            'best_val_loss': self.best_val_loss,
            'perceiver_config': {
                'num_latents': self.config.num_latents,
                'latent_dim': self.config.perceiver_dim,
                'num_layers': self.config.num_perceiver_layers,
                'num_heads': self.config.num_attn_heads,
                'dropout': self.config.perceiver_dropout,
            },
            'mrl_dims': self.config.mrl_dims,
            'projection_dim': self.config.d_align,
            'training_date': datetime.now().isoformat(),
        }
        
        torch.save(checkpoint, ckpt_dir / f"{tag}.pt")
        print(f"Saved checkpoint: {ckpt_dir / f'{tag}.pt'}")
    
    def load_checkpoint(self, checkpoint_path: str) -> int:
        """Load model checkpoint and return the epoch to resume from."""
        print(f"\nLoading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        model = self.model.module if hasattr(self.model, 'module') else self.model
        model.load_state_dict(checkpoint['model_state_dict'])
        
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.global_step = checkpoint.get('global_step', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        epoch = checkpoint.get('epoch', 0)
        print(f"Resumed from epoch {epoch}, step {self.global_step}, best_val_loss: {self.best_val_loss:.4f}")
        
        return epoch + 1
    
    def _save_with_rotation(self, epoch: int):
        """Save latest checkpoint with rotation (delete old epoch checkpoints)."""
        ckpt_dir = Path(self.config.ckpt_dir)
        
        self.save_checkpoint("latest", epoch=epoch)
        
        if is_main_process():
            for old_ckpt in ckpt_dir.glob("epoch_*.pt"):
                try:
                    old_ckpt.unlink()
                except Exception as e:
                    print(f"Warning: Could not delete {old_ckpt}: {e}")
            
            state_file = ckpt_dir / "training_state.txt"
            with open(state_file, 'w') as f:
                f.write(f"last_epoch: {epoch}\n")
                f.write(f"total_epochs: {self.config.num_epochs}\n")
                f.write(f"best_val_loss: {self.best_val_loss:.6f}\n")
                f.write(f"global_step: {self.global_step}\n")
    
    def train(self, start_epoch: int = 1):
        """Full training loop."""
        print("\n" + "="*70)
        print("STARTING TRAINING" + (f" (resuming from epoch {start_epoch})" if start_epoch > 1 else ""))
        print("="*70)
        
        params = count_parameters(self.model)
        print(f"Trainable parameters: {params['trainable']:,}")
        print(f"Total parameters: {params['total']:,}")
        
        for epoch in range(start_epoch, self.config.num_epochs + 1):
            print(f"\n=== Epoch {epoch}/{self.config.num_epochs} ===")
            
            epoch_loss = 0.0
            num_batches = 0
            
            pbar = tqdm(
                self.train_loader, 
                desc=f"Epoch {epoch}", 
                disable=not is_main_process()
            )
            
            for batch in pbar:
                metrics = self.train_step(batch)
                self.global_step += 1
                
                epoch_loss += metrics['loss']
                num_batches += 1
                
                pbar.set_postfix({
                    'loss': f"{metrics['loss']:.4f}",
                    'lr': f"{metrics['lr']:.2e}",
                })
                
                if self.wandb_run is not None and self.global_step % self.config.log_every == 0:
                    self.wandb_run.log({
                        'train/loss': metrics['loss'],
                        'train/loss_clip': metrics['loss_clip'],
                        'train/loss_mrl': metrics['loss_mrl'],
                        'train/grad_norm': metrics['grad_norm'],
                        'train/lr': metrics['lr'],
                        'train/epoch': epoch,
                        'train/step': self.global_step,
                    })
            
            avg_epoch_loss = epoch_loss / num_batches
            print(f"Epoch {epoch} average loss: {avg_epoch_loss:.4f}")
            
            if epoch % self.config.eval_every_epoch == 0:
                val_metrics = self.validate()
                
                if val_metrics:
                    print(f"Validation loss: {val_metrics['val_loss']:.4f}")
                    if 'val_avg_r@1' in val_metrics:
                        print(f"R@1: {val_metrics['val_avg_r@1']:.4f}, R@5: {val_metrics['val_avg_r@5']:.4f}, R@10: {val_metrics['val_avg_r@10']:.4f}")
                    
                    if self.wandb_run is not None:
                        self.wandb_run.log({
                            **val_metrics,
                            'val/epoch': epoch,
                        })
                    
                    if val_metrics['val_loss'] < self.best_val_loss:
                        self.best_val_loss = val_metrics['val_loss']
                        self.save_checkpoint("best", epoch=epoch)
            
            self._save_with_rotation(epoch)
        
        self.save_checkpoint("final", epoch=self.config.num_epochs)
        
        if self.wandb_run is not None:
            self.wandb_run.finish()
        
        print("\n" + "="*70)
        print("TRAINING COMPLETE")
        print("="*70)
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Checkpoints saved to: {self.config.ckpt_dir}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train Perceiver Alignment Model")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--train_parquet", type=str, default=None)
    parser.add_argument("--val_parquet", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./outputs/perceiver_v2")
    parser.add_argument("--ckpt_dir", type=str, default="./checkpoints/perceiver_v2")
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint or 'auto'")
    args = parser.parse_args()
    
    is_distributed = "RANK" in os.environ or "LOCAL_RANK" in os.environ
    if is_distributed:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    config = MultimodalAlignmentConfig()
    
    if args.train_parquet:
        config.train_parquet = args.train_parquet
    else:
        config.train_parquet = "/home/hice1/vchopra37/scratch/projects/edge_glass/dataset/final_dataset/pixmo/pixmo_train.parquet"
    
    if args.val_parquet:
        config.val_parquet = args.val_parquet
    else:
        config.val_parquet = "/home/hice1/vchopra37/scratch/projects/edge_glass/dataset/final_dataset/pixmo/pixmo_val.parquet"
    
    config.batch_size = args.batch_size
    config.num_epochs = args.epochs
    config.learning_rate = args.lr
    config.output_dir = args.output_dir
    config.ckpt_dir = args.ckpt_dir
    config.use_wandb = not args.no_wandb
    
    set_seed(config.seed)
    
    if is_main_process():
        print("\n" + "="*70)
        print("PERCEIVER MULTIMODAL ALIGNMENT TRAINING V2")
        print("="*70)
        print(f"Device: {device}")
        print(f"Train parquet: {config.train_parquet}")
        print(f"Val parquet: {config.val_parquet}")
        print(f"Batch size: {config.batch_size}")
        print(f"Learning rate: {config.learning_rate}")
        print(f"Epochs: {config.num_epochs}")
    
    if is_main_process():
        print("\n=== Loading Frozen Encoders ===")
    
    print(f"Loading CLIP: {config.vision_model_name}")
    clip_processor = CLIPImageProcessor.from_pretrained(config.vision_model_name)
    clip_vision = CLIPVisionModel.from_pretrained(config.vision_model_name)
    clip_vision.to(device)
    clip_vision.eval()
    for p in clip_vision.parameters():
        p.requires_grad = False
    
    print(f"Loading text encoder: {config.text_model_name}")
    text_tokenizer = AutoTokenizer.from_pretrained(config.text_model_name)
    text_encoder = AutoModel.from_pretrained(config.text_model_name)
    text_encoder.to(device)
    text_encoder.eval()
    for p in text_encoder.parameters():
        p.requires_grad = False
    
    if is_main_process():
        print("\n=== Creating Alignment Model ===")
    
    model = MultimodalAlignmentModel(config)
    model.to(device)
    
    if is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, 
            device_ids=[local_rank],
            output_device=local_rank,
        )
    
    if is_main_process():
        print("\n=== Loading Datasets ===")
    
    train_loader, val_loader = build_train_val_loaders(
        config=config,
        is_distributed=is_distributed,
    )
    
    trainer = PerceiverTrainer(
        config=config,
        model=model,
        clip_processor=clip_processor,
        clip_vision=clip_vision,
        text_tokenizer=text_tokenizer,
        text_encoder=text_encoder,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
    )
    
    start_epoch = 1
    if args.resume:
        ckpt_dir = Path(config.ckpt_dir)
        if args.resume == 'auto':
            latest_ckpt = ckpt_dir / "latest.pt"
            if latest_ckpt.exists():
                start_epoch = trainer.load_checkpoint(str(latest_ckpt))
            else:
                print("No latest.pt found, starting from scratch.")
        else:
            resume_path = Path(args.resume)
            if resume_path.exists():
                start_epoch = trainer.load_checkpoint(str(resume_path))
            else:
                print(f"Warning: Checkpoint {args.resume} not found, starting from scratch.")
    
    trainer.train(start_epoch=start_epoch)
    
    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
