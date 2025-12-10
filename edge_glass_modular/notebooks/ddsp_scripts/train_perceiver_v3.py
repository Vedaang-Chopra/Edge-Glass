#!/usr/bin/env python3
"""
================================================================================
DDSP PERCEIVER ALIGNMENT TRAINING SCRIPT (V2) - ACCELERATE VERSION
================================================================================

A complete training script for multimodal alignment with Perceiver Resampler,
using Accelerate + DeepSpeed for efficient distributed training.

Features:
- Accelerate for distributed training with DeepSpeed ZeRO
- Frozen encoders (CLIP, MiniLM) for feature extraction
- Trainable Perceiver Resampler shared across modalities
- MRL (Matryoshka Representation Learning) and CLIP-style contrastive loss
- WandB integration for experiment tracking
- Checkpoint saving with rotation (best + latest only)
- Resume from checkpoint support
- Retrieval metrics (R@1, R@5, R@10)

Usage:
    accelerate launch --multi_gpu train_perceiver_v2.py --epochs 10
    accelerate launch --num_processes=4 train_perceiver_v2.py --epochs 10

================================================================================
"""

import os
import sys
import math
import random
import argparse
import time
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# Disable torch compile to avoid bf16/fp32 conflicts with frozen encoders
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCH_DYNAMO_DISABLE"] = "1"

try:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
    torch._dynamo.config.disable = True
except:
    pass

# Accelerate imports
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from accelerate import DeepSpeedPlugin

import logging

# Add paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
SRC_PATH = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_PATH))
sys.path.insert(0, str(PROJECT_ROOT))

# Try imports
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

try:
    from transformers import (
        CLIPImageProcessor,
        CLIPVisionModel,
        AutoTokenizer,
        AutoModel,
    )
except ImportError:
    print("Error: transformers not available.")
    sys.exit(1)

logger = get_logger(__name__)


# ============================================================================
# CONFIGURATION (Optimized hyperparameters)
# ============================================================================

@dataclass
class MultimodalAlignmentConfig:
    """Configuration for multimodal alignment with Perceiver."""
    
    # Model Names
    vision_model_name: str = "openai/clip-vit-large-patch14-336"
    text_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Encoder Dimensions
    d_vision: int = 1024
    d_text: int = 384
    
    # Perceiver Architecture
    perceiver_dim: int = 512
    num_latents: int = 64
    num_perceiver_layers: int = 4
    num_attn_heads: int = 8
    perceiver_mlp_ratio: float = 4.0
    perceiver_dropout: float = 0.1
    
    # Alignment
    d_align: int = 512
    mrl_dims: Tuple[int, ...] = (64, 128, 256, 512)
    
    # Training (Optimized)
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    num_epochs: int = 10
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 1
    
    # Loss (Optimized)
    temperature: float = 0.07
    mrl_weight: float = 1.0
    clip_weight: float = 0.25
    
    # Data
    image_size: int = 336
    max_text_length: int = 77
    train_parquet: str = ""
    val_parquet: str = ""
    num_workers: int = 8
    
    # Misc
    seed: int = 42
    output_dir: str = "./outputs/perceiver_v2"
    ckpt_dir: str = "./checkpoints/perceiver_v2"
    log_every: int = 50
    wandb_project: str = "edge_glass_alignment"
    wandb_run_name: str = "perceiver_v2_accelerate"


# ============================================================================
# PERCEIVER COMPONENTS
# ============================================================================

class FeedForward(nn.Module):
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
    
    def forward(self, x):
        return self.net(x)


class PerceiverAttention(nn.Module):
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
    
    def forward(self, q, kv, mask=None):
        B, N_q, D = q.shape
        N_kv = kv.shape[1]
        
        q = self.ln_q(q)
        kv = self.ln_kv(kv)
        
        Q = self.q_proj(q).view(B, N_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(kv).view(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(kv).view(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn = (Q @ K.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn.masked_fill(~mask.bool(), float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = (attn @ V).transpose(1, 2).reshape(B, N_q, D)
        return self.out_proj(out)


class PerceiverLayer(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = PerceiverAttention(dim, num_heads, dropout)
        self.self_attn = PerceiverAttention(dim, num_heads, dropout)
        self.ffn = FeedForward(dim, mlp_ratio, dropout)
        self.ln_ffn = nn.LayerNorm(dim)
    
    def forward(self, latents, tokens, mask=None):
        latents = latents + self.cross_attn(latents, tokens, mask)
        latents = latents + self.self_attn(latents, latents)
        latents = latents + self.ffn(self.ln_ffn(latents))
        return latents


class PerceiverResampler(nn.Module):
    def __init__(self, dim: int, num_latents: int = 64, num_layers: int = 4,
                 num_heads: int = 8, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, dim) * (dim ** -0.5))
        self.layers = nn.ModuleList([
            PerceiverLayer(dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        self.ln_out = nn.LayerNorm(dim)
    
    def forward(self, tokens, mask=None):
        B = tokens.shape[0]
        latents = self.latents.unsqueeze(0).expand(B, -1, -1)
        for layer in self.layers:
            latents = layer(latents, tokens, mask)
        return self.ln_out(latents)


class MLPAdapter(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_factor: float = 2.0, dropout: float = 0.1):
        super().__init__()
        hidden_dim = int(in_dim * hidden_factor)
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )
    
    def forward(self, x):
        return self.net(x)


class AlignmentProjector(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, output_dim),
        )
    
    def forward(self, latents):
        return self.proj(latents.mean(dim=1))


class MultimodalAlignmentModel(nn.Module):
    def __init__(self, config: MultimodalAlignmentConfig):
        super().__init__()
        self.config = config
        
        self.vision_adapter = MLPAdapter(config.d_vision, config.perceiver_dim, dropout=config.perceiver_dropout)
        self.text_adapter = MLPAdapter(config.d_text, config.perceiver_dim, dropout=config.perceiver_dropout)
        
        self.perceiver = PerceiverResampler(
            dim=config.perceiver_dim,
            num_latents=config.num_latents,
            num_layers=config.num_perceiver_layers,
            num_heads=config.num_attn_heads,
            mlp_ratio=config.perceiver_mlp_ratio,
            dropout=config.perceiver_dropout,
        )
        
        self.alignment_projector = AlignmentProjector(config.perceiver_dim, config.d_align)
    
    def encode_vision(self, features):
        tokens = self.vision_adapter(features)
        latents = self.perceiver(tokens)
        return self.alignment_projector(latents)
    
    def encode_text(self, features):
        tokens = self.text_adapter(features)
        latents = self.perceiver(tokens)
        return self.alignment_projector(latents)


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

def contrastive_loss(z_a, z_b, temperature=0.07):
    z_a = F.normalize(z_a, dim=-1)
    z_b = F.normalize(z_b, dim=-1)
    logits = z_a @ z_b.T / temperature
    labels = torch.arange(z_a.size(0), device=z_a.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2


def matryoshka_loss(z_a, z_b, dims=(64, 128, 256, 512), temperature=0.07):
    total_loss = 0.0
    num_dims = 0
    for dim in dims:
        if dim > z_a.size(-1):
            continue
        total_loss += contrastive_loss(z_a[:, :dim], z_b[:, :dim], temperature)
        num_dims += 1
    return total_loss / num_dims if num_dims > 0 else total_loss


# ============================================================================
# DATA LOADING
# ============================================================================

try:
    from data.dataset_builder import build_image_datasets_from_parquet
    from data.transforms import get_image_transforms
    HAS_MODULAR_DATA = True
except ImportError:
    HAS_MODULAR_DATA = False
    logger.warning("Could not import modular data loaders.")


def collate_fn(batch):
    images = torch.stack([item['image'] for item in batch])
    texts = [item['text'] for item in batch]
    return {'images': images, 'captions': texts}


def build_dataloaders(config, accelerator):
    if not HAS_MODULAR_DATA:
        raise ImportError("Modular data loaders required.")
    
    train_transforms = get_image_transforms(image_size=config.image_size, is_training=True)
    val_transforms = get_image_transforms(image_size=config.image_size, is_training=False)
    
    datasets = build_image_datasets_from_parquet(
        cfg=config,
        train_parquet_path=config.train_parquet,
        val_parquet_path=config.val_parquet,
        train_transforms=train_transforms,
        val_transforms=val_transforms,
        max_text_length=config.max_text_length,
        text_dropout_prob=0.1,
    )
    
    train_loader = DataLoader(
        datasets['train'],
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        datasets['val'],
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    logger.info(f"Train: {len(datasets['train'])} samples, Val: {len(datasets['val'])} samples")
    return train_loader, val_loader


# ============================================================================
# RETRIEVAL METRICS
# ============================================================================

@torch.no_grad()
def compute_retrieval_metrics(z_vision, z_text):
    z_vision = F.normalize(z_vision, dim=-1)
    z_text = F.normalize(z_text, dim=-1)
    sim = z_vision @ z_text.T
    
    B = z_vision.size(0)
    targets = torch.arange(B, device=z_vision.device)
    
    rankings_i2t = (sim.argsort(dim=-1, descending=True) == targets.unsqueeze(1)).nonzero(as_tuple=True)[1]
    rankings_t2i = (sim.T.argsort(dim=-1, descending=True) == targets.unsqueeze(1)).nonzero(as_tuple=True)[1]
    
    def r_at_k(r, k):
        return (r < k).float().mean().item()
    
    return {
        'i2t_r@1': r_at_k(rankings_i2t, 1),
        'i2t_r@5': r_at_k(rankings_i2t, 5),
        'i2t_r@10': r_at_k(rankings_i2t, 10),
        't2i_r@1': r_at_k(rankings_t2i, 1),
        't2i_r@5': r_at_k(rankings_t2i, 5),
        't2i_r@10': r_at_k(rankings_t2i, 10),
        'avg_r@1': (r_at_k(rankings_i2t, 1) + r_at_k(rankings_t2i, 1)) / 2,
        'avg_r@5': (r_at_k(rankings_i2t, 5) + r_at_k(rankings_t2i, 5)) / 2,
        'avg_r@10': (r_at_k(rankings_i2t, 10) + r_at_k(rankings_t2i, 10)) / 2,
    }


# ============================================================================
# MAIN
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Train Perceiver Alignment (Accelerate)")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--train_parquet", type=str, default=None)
    parser.add_argument("--val_parquet", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./outputs/perceiver_v2")
    parser.add_argument("--ckpt_dir", type=str, default="./checkpoints/perceiver_v2")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint or 'auto'")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--deepspeed_stage", type=int, default=2, choices=[0, 1, 2, 3])
    parser.add_argument("--early_stopping_patience", type=int, default=5, 
                        help="Stop if val loss doesn't improve for N epochs (0=disabled)")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Config
    config = MultimodalAlignmentConfig()
    config.batch_size = args.batch_size
    config.num_epochs = args.epochs
    config.learning_rate = args.lr
    config.output_dir = args.output_dir
    config.ckpt_dir = args.ckpt_dir
    config.gradient_accumulation_steps = args.gradient_accumulation_steps
    
    if args.train_parquet:
        config.train_parquet = args.train_parquet
    else:
        config.train_parquet = "/home/hice1/vchopra37/scratch/projects/edge_glass/dataset/final_dataset/pixmo/pixmo_train.parquet"
    
    if args.val_parquet:
        config.val_parquet = args.val_parquet
    else:
        config.val_parquet = "/home/hice1/vchopra37/scratch/projects/edge_glass/dataset/final_dataset/pixmo/pixmo_val.parquet"
    
    if args.run_name:
        config.wandb_run_name = args.run_name
    
    # Setup Accelerator with DeepSpeed
    ds_plugin = DeepSpeedPlugin(
        zero_stage=args.deepspeed_stage,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_clipping=config.max_grad_norm,
    )
    
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="bf16",
        log_with="wandb" if args.use_wandb else None,
        deepspeed_plugin=ds_plugin,
    )
    
    # Logging
    logging.basicConfig(level=logging.INFO)
    logger.info(accelerator.state)
    
    set_seed(config.seed)
    
    # Init WandB
    if accelerator.is_main_process and args.use_wandb:
        accelerator.init_trackers(
            project_name=config.wandb_project,
            config=asdict(config),
            init_kwargs={"wandb": {"name": config.wandb_run_name}}
        )
    
    # Load Frozen Encoders
    logger.info(f"Loading CLIP: {config.vision_model_name}")
    clip_processor = CLIPImageProcessor.from_pretrained(config.vision_model_name)
    clip_vision = CLIPVisionModel.from_pretrained(config.vision_model_name)
    clip_vision.to(accelerator.device)
    clip_vision.eval()
    for p in clip_vision.parameters():
        p.requires_grad = False
    
    logger.info(f"Loading text encoder: {config.text_model_name}")
    text_tokenizer = AutoTokenizer.from_pretrained(config.text_model_name)
    text_encoder = AutoModel.from_pretrained(config.text_model_name)
    text_encoder.to(accelerator.device)
    text_encoder.eval()
    for p in text_encoder.parameters():
        p.requires_grad = False
    
    # Encoding helpers
    @torch.no_grad()
    def encode_images(images):
        images = images.to(accelerator.device)
        with torch.amp.autocast('cuda', enabled=False):  # Keep fp32 for frozen encoder
            features = clip_vision(pixel_values=images).last_hidden_state
        return features.to(torch.bfloat16)  # Convert to bf16 for model
    
    @torch.no_grad()
    def encode_texts(texts):
        tokens = text_tokenizer(texts, padding=True, truncation=True, 
                                max_length=config.max_text_length, return_tensors="pt")
        tokens = {k: v.to(accelerator.device) for k, v in tokens.items()}
        with torch.amp.autocast('cuda', enabled=False):  # Keep fp32 for frozen encoder
            features = text_encoder(**tokens).last_hidden_state
        return features.to(torch.bfloat16)  # Convert to bf16 for model
    
    # Create Model
    logger.info("Creating alignment model...")
    model = MultimodalAlignmentModel(config)
    
    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    
    # Data
    train_loader, val_loader = build_dataloaders(config, accelerator)
    
    # Calculate steps
    num_update_steps_per_epoch = len(train_loader) // args.gradient_accumulation_steps
    total_steps = config.num_epochs * num_update_steps_per_epoch
    warmup_steps = int(total_steps * config.warmup_ratio)
    
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Prepare with Accelerator
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )
    
    # Resume
    start_epoch = 0
    best_val_loss = float('inf')
    global_step = 0
    epochs_without_improvement = 0
    
    if args.resume:
        ckpt_dir = Path(config.ckpt_dir)
        if args.resume == 'auto':
            latest = ckpt_dir / "checkpoint-latest"
            if latest.exists():
                logger.info(f"Resuming from {latest}")
                accelerator.load_state(str(latest))
        elif Path(args.resume).exists():
            logger.info(f"Resuming from {args.resume}")
            accelerator.load_state(args.resume)
        
        # Always try to read training state when resuming
        state_file = ckpt_dir / "training_state.txt"
        if state_file.exists():
            logger.info(f"Reading training state from {state_file}")
            for line in state_file.read_text().split('\n'):
                if line.startswith('last_epoch:'):
                    start_epoch = int(line.split(':')[1].strip())
                if line.startswith('best_val_loss:'):
                    best_val_loss = float(line.split(':')[1].strip())
                if line.startswith('global_step:'):
                    global_step = int(line.split(':')[1].strip())
            logger.info(f"Resuming from epoch {start_epoch}, best_val_loss={best_val_loss:.4f}")
    
    # Training Loop
    logger.info("Starting training...")
    
    for epoch in range(start_epoch, config.num_epochs):
        model.train()
        total_loss = 0.0
        
        progress = tqdm(train_loader, disable=not accelerator.is_local_main_process, desc=f"Epoch {epoch+1}")
        
        for batch in progress:
            with accelerator.accumulate(model):
                # Encode
                vision_features = encode_images(batch['images'])
                text_features = encode_texts(batch['captions'])
                
                # Forward
                z_vision = model.encode_vision(vision_features)
                z_text = model.encode_text(text_features)
                
                # Loss
                loss_clip = contrastive_loss(z_vision, z_text, config.temperature)
                loss_mrl = matryoshka_loss(z_vision, z_text, config.mrl_dims, config.temperature)
                loss = config.clip_weight * loss_clip + config.mrl_weight * loss_mrl
                
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    total_norm = accelerator.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            if accelerator.sync_gradients:
                global_step += 1
                total_loss += loss.item()
                
                if global_step % config.log_every == 0 and accelerator.is_main_process:
                    lr = scheduler.get_last_lr()[0]
                    accelerator.log({
                        "train/loss": loss.item(),
                        "train/loss_clip": loss_clip.item(),
                        "train/loss_mrl": loss_mrl.item(),
                        "train/lr": lr,
                        "train/epoch": epoch + 1,
                    }, step=global_step)
                
                progress.set_postfix({'loss': f"{loss.item():.4f}"})
        
        # Validation
        model.eval()
        val_loss = 0.0
        num_val = 0
        all_z_v, all_z_t = [], []
        
        for batch in tqdm(val_loader, disable=not accelerator.is_local_main_process, desc="Validation"):
            with torch.no_grad():
                vision_features = encode_images(batch['images'])
                text_features = encode_texts(batch['captions'])
                
                z_vision = model.encode_vision(vision_features)
                z_text = model.encode_text(text_features)
                
                loss_clip = contrastive_loss(z_vision, z_text, config.temperature)
                loss_mrl = matryoshka_loss(z_vision, z_text, config.mrl_dims, config.temperature)
                loss = config.clip_weight * loss_clip + config.mrl_weight * loss_mrl
                
                val_loss += loss.item()
                num_val += 1
                
                if len(all_z_v) * z_vision.size(0) < 1000:
                    all_z_v.append(z_vision.cpu())
                    all_z_t.append(z_text.cpu())
        
        # Calculate local average
        avg_val_loss = val_loss / num_val if num_val > 0 else 0.0
        
        # Synchronize validation loss
        avg_val_loss_tensor = torch.tensor([avg_val_loss], device=accelerator.device)
        avg_val_loss = accelerator.gather(avg_val_loss_tensor).mean().item()
        
        # Retrieval metrics
        retrieval_metrics = {}
        if all_z_v:
            z_v = torch.cat(all_z_v, dim=0).to(accelerator.device)
            z_t = torch.cat(all_z_t, dim=0).to(accelerator.device)
            # Gather embeddings from all processes for accurate metrics
            z_v = accelerator.gather(z_v)
            z_t = accelerator.gather(z_t)
            
            # Compute metrics only on main process to save compute
            if accelerator.is_main_process:
                retrieval_metrics = compute_retrieval_metrics(z_v, z_t)
        
        # Sync metrics to other processes for logging consistency (optional but good)
        # For simplicity, we only log on main process, but saving decision depends on loss
        
        if accelerator.is_main_process:
            logger.info(f"Epoch {epoch+1} | Val Loss: {avg_val_loss:.4f} | R@1: {retrieval_metrics.get('avg_r@1', 0):.4f}")
            
            if args.use_wandb:
                # Log metrics
                wandb_metrics = {
                    "val/loss": avg_val_loss,
                    **{f"val/{k}": v for k, v in retrieval_metrics.items()},
                }
                accelerator.log(wandb_metrics, step=global_step)
        
        # Save best model - MUST be called on all processes for DeepSpeed
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0  # Reset counter
            ckpt_dir = Path(config.ckpt_dir) / "checkpoint-best"
            if accelerator.is_main_process:
                logger.info(f"New best! Saving to {ckpt_dir}")
            accelerator.save_state(str(ckpt_dir))
        else:
            epochs_without_improvement += 1
            if accelerator.is_main_process:
                logger.info(f"No improvement for {epochs_without_improvement} epoch(s)")
        
        # Early stopping check
        if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
            if accelerator.is_main_process:
                logger.info(f"Early stopping triggered after {epochs_without_improvement} epochs without improvement")
            break
        
        # Save latest (with rotation)
        ckpt_dir = Path(config.ckpt_dir)
        latest_dir = ckpt_dir / "checkpoint-latest"
        accelerator.save_state(str(latest_dir))
        
        # Write state file
        if accelerator.is_main_process:
            state_file = ckpt_dir / "training_state.txt"
            with open(state_file, 'w') as f:
                f.write(f"last_epoch: {epoch + 1}\n")
                f.write(f"total_epochs: {config.num_epochs}\n")
                f.write(f"best_val_loss: {best_val_loss:.6f}\n")
                f.write(f"global_step: {global_step}\n")
    
    # Final save - MUST be called on all processes for DeepSpeed
    final_dir = Path(config.ckpt_dir) / "checkpoint-final"
    accelerator.save_state(str(final_dir))
    
    if accelerator.is_main_process:
        logger.info(f"Training complete! Best val loss: {best_val_loss:.4f}")
    
    accelerator.end_training()


if __name__ == "__main__":
    main()
