"""Improved DDP-ready training with better checkpointing and logging."""

from __future__ import annotations

import os
import json
import dataclasses
from dataclasses import dataclass
from typing import Callable, Optional, Dict
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
import numpy as np

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from config import ExperimentConfig
from models import MultimodalAlignmentModel
from utils import get_rank, init_distributed, setup_logger


@dataclass
class TrainerState:
    """Training state for checkpointing."""
    global_step: int = 0
    epoch: int = 0
    best_val_loss: float = float('inf')
    best_model_path: Optional[str] = None


class ImprovedMultimodalTrainer:
    """Improved trainer with comprehensive logging and checkpointing.

    Features:
    - Automatic checkpoint saving and loading
    - Best model tracking with crash recovery
    - Warmup + cosine decay LR schedule
    - Comprehensive loss logging (CLIP, MRL, total)
    - WandB integration
    - Gradient clipping and accumulation
    """

    def __init__(
        self,
        cfg: ExperimentConfig,
        model: MultimodalAlignmentModel,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        use_wandb: bool = False,
    ):
        self.cfg = cfg
        self.logger = setup_logger(cfg.name)
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.use_wandb = use_wandb and WANDB_AVAILABLE

        # Device setup
        self.device = torch.device("cuda", get_rank()) if torch.cuda.is_available() else torch.device("cpu")
        self.model.to(self.device)

        # DDP setup
        ddp_requested = cfg.trainer.strategy == "ddp" and torch.cuda.device_count() > 1
        distributed_env_ready = "RANK" in os.environ or dist.is_initialized()
        if ddp_requested and distributed_env_ready:
            init_distributed()
            self.model = DDP(self.model, device_ids=[self.device.index])
            self.world_size = dist.get_world_size()
        else:
            if ddp_requested and not distributed_env_ready:
                self.logger.warning(
                    "DDP requested but no distributed process group found. "
                    "Falling back to single-process training. Launch with torchrun to enable DDP."
                )
            self.world_size = 1

        # Effective batch size with DDP
        self.effective_batch_size = cfg.dataset.batch_size * self.world_size

        # Optimizer
        model_params = self.model.module.parameters() if isinstance(self.model, DDP) else self.model.parameters()
        self.optimizer = AdamW(
            model_params,
            lr=cfg.optimization.lr,
            weight_decay=cfg.optimization.weight_decay,
            betas=(0.9, 0.95),
        )

        # Learning rate scheduler (warmup + cosine decay)
        print(len(train_loader), cfg.trainer.epochs, cfg.optimization.grad_accum_steps)
        num_training_steps = len(train_loader) * cfg.trainer.epochs // cfg.optimization.grad_accum_steps
        warmup_steps = int(num_training_steps * cfg.optimization.warmup_ratio)
        self.scheduler = self._get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
        )

        # Mixed precision
        self.scaler = torch.cuda.amp.GradScaler(enabled=cfg.optimization.fp16 or cfg.optimization.bf16)

        # Training state
        self.state = TrainerState()

        # Checkpoint directory
        self.ckpt_dir = Path(cfg.trainer.ckpt_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Load checkpoint if exists
        self._load_latest_checkpoint()

        # Training history for visualization
        self.history = {
            'train_loss': [],
            'train_loss_clip': [],
            'train_loss_mrl': [],
            'val_loss': [],
            'val_loss_clip': [],
            'val_loss_mrl': [],
            'val_i2t_r1': [],
            'val_i2t_r5': [],
            'val_i2t_r10': [],
            'lr': [],
        }

        # Initialize WandB
        if self.use_wandb and get_rank() == 0:
            wandb.init(
                project=cfg.trainer.wandb_project,
                name=cfg.trainer.wandb_run_name,
                config=dataclasses.asdict(cfg),
                resume="allow",
            )

    def _get_cosine_schedule_with_warmup(
        self,
        optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
    ) -> LambdaLR:
        """Create warmup + cosine decay schedule."""
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                # Linear warmup
                return float(current_step) / float(max(1, num_warmup_steps))
            # Cosine decay
            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))

        return LambdaLR(optimizer, lr_lambda)

    def _save_checkpoint(self, is_best: bool = False, epoch: Optional[int] = None):
        """Save training checkpoint.

        Args:
            is_best: Whether this is the best model so far
            epoch: Current epoch (for periodic checkpoints)
        """
        if get_rank() != 0:  # Only save on rank 0
            return

        model_to_save = self.model.module if isinstance(self.model, DDP) else self.model

        save_optimizer_state = getattr(self.cfg.trainer, "save_optimizer_state", True)
        best_weights_only = getattr(self.cfg.trainer, "best_weights_only", False)

        checkpoint = {
            'epoch': self.state.epoch,
            'global_step': self.state.global_step,
            'best_val_loss': self.state.best_val_loss,
            'model_state_dict': model_to_save.state_dict(),
            'history': self.history,
            'config': self.cfg,
        }
        if save_optimizer_state:
            checkpoint.update(
                {
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'scaler_state_dict': self.scaler.state_dict(),
                }
            )

        # Save latest checkpoint
        latest_path = self.ckpt_dir / "checkpoint_latest.pt"
        try:
            torch.save(checkpoint, latest_path)
            self.logger.info(f"Saved latest checkpoint to {latest_path}")
        except RuntimeError as e:
            self.logger.error(f"Failed to save latest checkpoint to {latest_path}: {e}")
            return

        # Save best model
        if is_best:
            best_path = self.ckpt_dir / "checkpoint_best.pt"
            best_payload = checkpoint
            if best_weights_only:
                best_payload = {
                    'epoch': self.state.epoch,
                    'global_step': self.state.global_step,
                    'best_val_loss': self.state.best_val_loss,
                    'model_state_dict': model_to_save.state_dict(),
                    'config': self.cfg,
                }
            try:
                torch.save(best_payload, best_path)
                self.state.best_model_path = str(best_path)
                self.logger.info(f"✓ Saved best model to {best_path}")
            except RuntimeError as e:
                self.logger.error(f"Failed to save best checkpoint to {best_path}: {e}")

        # Save periodic checkpoint
        if epoch is not None:
            epoch_path = self.ckpt_dir / f"checkpoint_epoch_{epoch}.pt"
            try:
                torch.save(checkpoint, epoch_path)
                self.logger.info(f"Saved epoch {epoch} checkpoint to {epoch_path}")
            except RuntimeError as e:
                self.logger.error(f"Failed to save epoch checkpoint {epoch_path}: {e}")

    def _load_latest_checkpoint(self):
        """Load latest checkpoint for crash recovery."""
        # Try to load best checkpoint first
        best_path = self.ckpt_dir / "checkpoint_best.pt"
        latest_path = self.ckpt_dir / "checkpoint_latest.pt"
        save_optimizer_state = getattr(self.cfg.trainer, "save_optimizer_state", True)

        # Prefer latest for continuing training
        checkpoint_path = latest_path if latest_path.exists() else (best_path if best_path.exists() else None)

        if checkpoint_path is None:
            self.logger.info("No checkpoint found. Starting training from scratch.")
            return

        self.logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Load model state
        model_to_load = self.model.module if isinstance(self.model, DDP) else self.model
        model_to_load.load_state_dict(checkpoint['model_state_dict'])

        # Load optimizer and scheduler
        if save_optimizer_state and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        else:
            self.logger.info("Checkpoint missing optimizer/scheduler/scaler states; starting those fresh.")

        # Load training state
        self.state.epoch = checkpoint['epoch']
        self.state.global_step = checkpoint['global_step']
        self.state.best_val_loss = checkpoint['best_val_loss']

        # Load history
        if 'history' in checkpoint:
            self.history = checkpoint['history']

        self.logger.info(f"Resumed from epoch {self.state.epoch}, step {self.state.global_step}")

    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        epoch_losses = {
            'total': 0,
            'clip': 0,
            'mrl': 0,
        }
        num_batches = 0

        for batch_idx, batch in enumerate(self.train_loader):
            # Move to device
            images = batch['image'].to(self.device)
            texts = batch['text']

            # Forward pass
            with torch.cuda.amp.autocast(enabled=self.cfg.optimization.fp16 or self.cfg.optimization.bf16):
                outputs = self.model(images=images, texts=texts)
                loss = outputs.loss / self.cfg.optimization.grad_accum_steps

            # Backward pass
            self.scaler.scale(loss).backward()

            # Update weights
            if (batch_idx + 1) % self.cfg.optimization.grad_accum_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.cfg.optimization.gradient_clip,
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.scheduler.step()
                self.state.global_step += 1

                # Log LR
                current_lr = self.scheduler.get_last_lr()[0]
                self.history['lr'].append(current_lr)

                # Log to WandB
                if self.use_wandb and self.state.global_step % self.cfg.trainer.log_every == 0 and get_rank() == 0:
                    log_dict = {
                        'train/loss': outputs.loss.item(),
                        'train/lr': current_lr,
                        'train/epoch': self.state.epoch,
                        'train/step': self.state.global_step,
                    }
                    if outputs.losses:
                        if 'loss_clip' in outputs.losses:
                            log_dict['train/loss_clip'] = outputs.losses['loss_clip'].item()
                        if 'loss_mrl' in outputs.losses:
                            log_dict['train/loss_mrl'] = outputs.losses['loss_mrl'].item()
                    wandb.log(log_dict)

            # Track losses
            epoch_losses['total'] += outputs.loss.item()
            if outputs.losses:
                if 'loss_clip' in outputs.losses:
                    epoch_losses['clip'] += outputs.losses['loss_clip'].item()
                if 'loss_mrl' in outputs.losses:
                    epoch_losses['mrl'] += outputs.losses['loss_mrl'].item()
            num_batches += 1

        # Average losses
        avg_losses = {k: v / num_batches for k, v in epoch_losses.items()}
        return avg_losses

    @torch.no_grad()
    def validate(self):
        """Validate model."""
        if self.val_loader is None:
            return {}

        self.model.eval()
        val_losses = {
            'total': 0,
            'clip': 0,
            'mrl': 0,
        }
        num_batches = 0

        # Collect embeddings for retrieval
        vision_embeddings = []
        text_embeddings = []

        for batch in self.val_loader:
            images = batch['image'].to(self.device)
            texts = batch['text']

            outputs = self.model(images=images, texts=texts, return_embeddings=True)

            if outputs.loss is not None:
                val_losses['total'] += outputs.loss.item()
            if outputs.losses:
                if 'loss_clip' in outputs.losses:
                    val_losses['clip'] += outputs.losses['loss_clip'].item()
                if 'loss_mrl' in outputs.losses:
                    val_losses['mrl'] += outputs.losses['loss_mrl'].item()

            # Collect embeddings
            if outputs.vision_emb is not None:
                vision_embeddings.append(outputs.vision_emb.cpu())
            if outputs.text_emb is not None:
                text_embeddings.append(outputs.text_emb.cpu())

            num_batches += 1

        # Average losses
        avg_losses = {k: v / num_batches for k, v in val_losses.items()}

        # Compute retrieval metrics
        metrics = {}
        if vision_embeddings and text_embeddings:
            vision_embs = torch.cat(vision_embeddings, dim=0)
            text_embs = torch.cat(text_embeddings, dim=0)

            similarity = torch.matmul(vision_embs, text_embs.t())

            # Image-to-text retrieval
            ranks = torch.argsort(similarity, dim=1, descending=True)
            correct_indices = torch.arange(len(vision_embs)).unsqueeze(1)

            r1 = (ranks[:, :1] == correct_indices).any(dim=1).float().mean().item()
            r5 = (ranks[:, :5] == correct_indices).any(dim=1).float().mean().item()
            r10 = (ranks[:, :10] == correct_indices).any(dim=1).float().mean().item()

            metrics.update({
                'i2t_r1': r1,
                'i2t_r5': r5,
                'i2t_r10': r10,
            })

        return {**avg_losses, **metrics}

    def train(self):
        """Main training loop."""
        self.logger.info("Starting training...")
        self.logger.info(f"Epochs: {self.cfg.trainer.epochs}")
        self.logger.info(f"Steps per epoch: {len(self.train_loader)}")
        self.logger.info(f"Total training steps: {len(self.train_loader) * self.cfg.trainer.epochs}")
        self.logger.info(f"Effective batch size: {self.effective_batch_size}")

        for epoch in range(self.state.epoch, self.cfg.trainer.epochs):
            self.state.epoch = epoch
            self.logger.info(f"\nEpoch {epoch + 1}/{self.cfg.trainer.epochs}")

            # Train
            train_metrics = self.train_epoch()
            self.logger.info(f"Train - Total: {train_metrics['total']:.4f}, "
                           f"CLIP: {train_metrics['clip']:.4f}, MRL: {train_metrics['mrl']:.4f}")

            # Update history
            self.history['train_loss'].append(train_metrics['total'])
            self.history['train_loss_clip'].append(train_metrics['clip'])
            self.history['train_loss_mrl'].append(train_metrics['mrl'])

            # Validate
            val_metrics = self.validate()
            if val_metrics:
                self.logger.info(f"Val - Total: {val_metrics['total']:.4f}, "
                               f"CLIP: {val_metrics.get('clip', 0):.4f}, "
                               f"MRL: {val_metrics.get('mrl', 0):.4f}")
                if 'i2t_r1' in val_metrics:
                    self.logger.info(f"  R@1: {val_metrics['i2t_r1']*100:.2f}%, "
                                   f"R@5: {val_metrics['i2t_r5']*100:.2f}%, "
                                   f"R@10: {val_metrics['i2t_r10']*100:.2f}%")

                # Update history
                self.history['val_loss'].append(val_metrics['total'])
                self.history['val_loss_clip'].append(val_metrics.get('clip', 0))
                self.history['val_loss_mrl'].append(val_metrics.get('mrl', 0))
                self.history['val_i2t_r1'].append(val_metrics.get('i2t_r1', 0))
                self.history['val_i2t_r5'].append(val_metrics.get('i2t_r5', 0))
                self.history['val_i2t_r10'].append(val_metrics.get('i2t_r10', 0))

                # Log to WandB
                if self.use_wandb and get_rank() == 0:
                    wandb.log({
                        'val/loss': val_metrics['total'],
                        'val/loss_clip': val_metrics.get('clip', 0),
                        'val/loss_mrl': val_metrics.get('mrl', 0),
                        'val/i2t_r1': val_metrics.get('i2t_r1', 0),
                        'val/i2t_r5': val_metrics.get('i2t_r5', 0),
                        'val/i2t_r10': val_metrics.get('i2t_r10', 0),
                        'epoch': epoch,
                    })

                # Save best model
                is_best = val_metrics['total'] < self.state.best_val_loss
                if is_best:
                    self.state.best_val_loss = val_metrics['total']
                    self._save_checkpoint(is_best=True)

            # Save periodic checkpoint
            if (epoch + 1) % self.cfg.trainer.save_every == 0:
                self._save_checkpoint(epoch=epoch + 1)

            # Always save latest
            self._save_checkpoint(is_best=False)

        # Save training history
        if get_rank() == 0:
            history_path = self.ckpt_dir / 'training_history.json'
            with open(history_path, 'w') as f:
                json.dump(self.history, f, indent=2)
            self.logger.info(f"Saved training history to {history_path}")

        self.logger.info("\nTraining completed!")
        self.logger.info(f"Best validation loss: {self.state.best_val_loss:.4f}")

        if self.use_wandb and get_rank() == 0:
            wandb.finish()

        return self.history
