"""
train.py - Training utilities for multimodal alignment

Provides:
- Training loop for Phase 1 (vision-text alignment)
- Evaluation utilities
- Checkpoint saving/loading
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List
import time

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from imports.core import (
    VisionTextAligner,
    AlignmentConfig,
    compute_retrieval_metrics,
    l2_normalize,
)


# ============================================================
# Optimizer and Scheduler
# ============================================================

def build_optimizer(
    model: VisionTextAligner,
    learning_rate: float,
    weight_decay: float,
) -> AdamW:
    """Build AdamW optimizer for trainable parameters."""
    params = model.get_trainable_params()
    
    if not params:
        raise ValueError("No trainable parameters found!")
    
    return AdamW(params, lr=learning_rate, weight_decay=weight_decay)


def build_scheduler(
    optimizer: AdamW,
    num_training_steps: int,
    warmup_ratio: float = 0.1,
):
    """Build learning rate scheduler with warmup."""
    num_warmup_steps = int(num_training_steps * warmup_ratio)
    
    # Linear warmup
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=num_warmup_steps,
    )
    
    # Cosine decay
    decay_scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=num_training_steps - num_warmup_steps,
        T_mult=1,
    )
    
    return SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, decay_scheduler],
        milestones=[num_warmup_steps],
    )


# ============================================================
# Training Loop
# ============================================================

def train_one_epoch(
    model: VisionTextAligner,
    dataloader: DataLoader,
    optimizer: AdamW,
    scheduler: Optional[Any] = None,
    device: Optional[torch.device] = None,
    log_every: int = 50,
    epoch: int = 0,
    use_features: bool = False,
) -> Dict[str, float]:
    """
    Train for one epoch.
    
    Args:
        model: VisionTextAligner model
        dataloader: Training data loader
        optimizer: Optimizer
        scheduler: Optional LR scheduler
        device: Device to use
        log_every: Log every N steps
        epoch: Current epoch number
        use_features: If True, batch contains pre-extracted features
    
    Returns:
        dict with epoch metrics
    """
    model.vision_adapter.train()
    model.text_adapter.train()
    
    if device is None:
        device = model.cfg.device
    
    running_loss = 0.0
    running_mrl = 0.0
    running_clip = 0.0
    n_steps = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=True)
    
    for step, batch in enumerate(pbar, start=1):
        optimizer.zero_grad(set_to_none=True)
        
        if use_features:
            # Pre-extracted features
            features = batch["features"].to(device)
            mask = batch["feature_mask"].to(device)
            texts = batch["texts"]
            
            # Pool features (mean over valid tokens)
            mask_f = mask.float().unsqueeze(-1)
            pooled = (features * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1e-6)
            
            # Forward through adapters only
            z_vision = model.vision_adapter(pooled)
            z_text = model.encode_text(texts)
            
            # Compute loss
            from core import matryoshka_loss, clip_contrastive_loss
            z_v = z_vision.float()
            z_t = z_text.float()
            
            loss_mrl = matryoshka_loss(z_v, z_t, model.cfg.mrl_dims, model.cfg.mrl_temperature)
            loss_clip = clip_contrastive_loss(z_v, z_t, model.cfg.clip_temperature)
            loss = model.cfg.mrl_weight * loss_mrl + model.cfg.clip_weight * loss_clip
            
            metrics = {
                "loss": loss.item(),
                "loss_mrl": loss_mrl.item(),
                "loss_clip": loss_clip.item(),
            }
        else:
            # On-the-fly images
            images = batch["images"]
            texts = batch["texts"]
            
            output = model(images, texts)
            loss = output["loss"]
            metrics = {
                "loss": loss.item(),
                "loss_mrl": output["loss_mrl"],
                "loss_clip": output["loss_clip"],
            }
        
        # Backward
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.get_trainable_params(), model.cfg.max_grad_norm)
        
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        
        # Accumulate metrics
        running_loss += metrics["loss"]
        running_mrl += metrics["loss_mrl"]
        running_clip += metrics["loss_clip"]
        n_steps += 1
        
        # Log
        if step % log_every == 0:
            avg_loss = running_loss / n_steps
            pbar.set_postfix({
                "loss": f"{avg_loss:.4f}",
                "mrl": f"{running_mrl/n_steps:.4f}",
                "clip": f"{running_clip/n_steps:.4f}",
            })
    
    return {
        "train_loss": running_loss / max(1, n_steps),
        "train_mrl": running_mrl / max(1, n_steps),
        "train_clip": running_clip / max(1, n_steps),
    }


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate(
    model: VisionTextAligner,
    dataloader: DataLoader,
    device: Optional[torch.device] = None,
    use_features: bool = False,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    """
    Evaluate model on validation data.
    
    Returns:
        dict with loss and retrieval metrics
    """
    model.vision_adapter.eval()
    model.text_adapter.eval()
    
    if device is None:
        device = model.cfg.device
    
    all_z_vision = []
    all_z_text = []
    running_loss = 0.0
    n_steps = 0
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
        if max_batches and batch_idx >= max_batches:
            break
        
        if use_features:
            features = batch["features"].to(device)
            mask = batch["feature_mask"].to(device)
            texts = batch["texts"]
            
            mask_f = mask.float().unsqueeze(-1)
            pooled = (features * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1e-6)
            
            z_vision = model.vision_adapter(pooled)
            z_text = model.encode_text(texts)
            
            from core import matryoshka_loss, clip_contrastive_loss
            z_v = z_vision.float()
            z_t = z_text.float()
            
            loss_mrl = matryoshka_loss(z_v, z_t, model.cfg.mrl_dims, model.cfg.mrl_temperature)
            loss_clip = clip_contrastive_loss(z_v, z_t, model.cfg.clip_temperature)
            loss = model.cfg.mrl_weight * loss_mrl + model.cfg.clip_weight * loss_clip
        else:
            images = batch["images"]
            texts = batch["texts"]
            
            output = model(images, texts)
            loss = output["loss"]
            z_vision = output["z_vision"]
            z_text = output["z_text"]
        
        all_z_vision.append(z_vision.cpu())
        all_z_text.append(z_text.cpu())
        running_loss += loss.item()
        n_steps += 1
    
    # Concatenate all embeddings
    all_z_vision = torch.cat(all_z_vision, dim=0)
    all_z_text = torch.cat(all_z_text, dim=0)
    
    # Compute retrieval metrics
    retrieval = compute_retrieval_metrics(all_z_vision, all_z_text)
    
    return {
        "val_loss": running_loss / max(1, n_steps),
        **retrieval,
    }


# ============================================================
# Full Training Pipeline
# ============================================================

def train_alignment(
    model: VisionTextAligner,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    num_epochs: int = 5,
    log_every: int = 50,
    save_dir: Optional[str] = None,
    use_features: bool = False,
    log_fn: Optional[Callable] = None,
) -> Dict[str, List[float]]:
    """
    Full training pipeline.
    
    Args:
        model: VisionTextAligner model
        train_loader: Training data loader
        val_loader: Optional validation data loader
        num_epochs: Number of epochs
        log_every: Log every N steps
        save_dir: Directory to save checkpoints
        use_features: Whether using pre-extracted features
        log_fn: Optional logging function (e.g., wandb.log)
    
    Returns:
        dict with training history
    """
    cfg = model.cfg
    device = cfg.device
    
    # Build optimizer and scheduler
    optimizer = build_optimizer(model, cfg.learning_rate, cfg.weight_decay)
    
    num_training_steps = len(train_loader) * num_epochs
    scheduler = build_scheduler(optimizer, num_training_steps, cfg.warmup_ratio)
    
    # History
    history = {
        "train_loss": [],
        "val_loss": [],
        "R@1": [],
        "R@5": [],
        "R@10": [],
    }
    
    best_r1 = 0.0
    
    print(f"\n{'='*60}")
    print(f"Starting Training")
    print(f"  Epochs: {num_epochs}")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  LR: {cfg.learning_rate}")
    print(f"  Device: {device}")
    print(f"{'='*60}\n")
    
    for epoch in range(num_epochs):
        # Train
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            log_every=log_every,
            epoch=epoch,
            use_features=use_features,
        )
        
        history["train_loss"].append(train_metrics["train_loss"])
        
        # Validate
        if val_loader is not None:
            val_metrics = evaluate(
                model=model,
                dataloader=val_loader,
                device=device,
                use_features=use_features,
            )
            
            history["val_loss"].append(val_metrics["val_loss"])
            history["R@1"].append(val_metrics["R@1"])
            history["R@5"].append(val_metrics["R@5"])
            history["R@10"].append(val_metrics["R@10"])
            
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Train Loss: {train_metrics['train_loss']:.4f}")
            print(f"  Val Loss:   {val_metrics['val_loss']:.4f}")
            print(f"  R@1: {val_metrics['R@1']*100:.2f}%")
            print(f"  R@5: {val_metrics['R@5']*100:.2f}%")
            print(f"  R@10: {val_metrics['R@10']*100:.2f}%")
            
            # Save best
            if val_metrics["R@1"] > best_r1:
                best_r1 = val_metrics["R@1"]
                if save_dir:
                    save_checkpoint(model, optimizer, epoch, save_dir, "best")
                    print(f"  → New best R@1: {best_r1*100:.2f}%")
            
            if log_fn:
                log_fn({
                    "epoch": epoch,
                    **train_metrics,
                    **val_metrics,
                })
        else:
            print(f"\nEpoch {epoch} - Train Loss: {train_metrics['train_loss']:.4f}")
            
            if log_fn:
                log_fn({"epoch": epoch, **train_metrics})
    
    # Save final
    if save_dir:
        save_checkpoint(model, optimizer, num_epochs - 1, save_dir, "final")
    
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    if val_loader:
        print(f"  Best R@1: {best_r1*100:.2f}%")
    print(f"{'='*60}\n")
    
    return history


# ============================================================
# Checkpoint Management
# ============================================================

def save_checkpoint(
    model: VisionTextAligner,
    optimizer: AdamW,
    epoch: int,
    save_dir: str,
    name: str = "checkpoint",
):
    """Save model checkpoint."""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        "epoch": epoch,
        "config": model.cfg,
        "vision_adapter": model.vision_adapter.state_dict(),
        "text_adapter": model.text_adapter.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    
    torch.save(checkpoint, save_path / f"{name}.pt")
    print(f"Saved checkpoint: {save_path / f'{name}.pt'}")


def load_checkpoint(
    model: VisionTextAligner,
    checkpoint_path: str,
    load_optimizer: bool = False,
    optimizer: Optional[AdamW] = None,
    cfg: Optional= None,
):
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=cfg.device, weights_only=False)
    
    model.vision_adapter.load_state_dict(checkpoint["vision_adapter"])
    model.text_adapter.load_state_dict(checkpoint["text_adapter"])
    
    if load_optimizer and optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    return checkpoint["epoch"]
