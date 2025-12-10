"""
Train Pixmo Vision-Text Alignment Model using Accelerate.

This script incorporates all good training practices from train_vlm_accelerate.py:
- Accelerate + DeepSpeed integration for distributed training
- Automatic checkpoint resumption and rotation
- Warmup + cosine decay LR scheduling
- Gradient norm logging and clipping
- WandB integration with comprehensive metrics
- Mixed precision training (BF16/FP16)
- Debug mode for quick iteration
"""

import os
import sys
import argparse
import random
import numpy as np

# Disable torch compilation features to prevent backend compiler errors (cudagraphs)
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCH_DYNAMO_DISABLE"] = "1"

import torch
import torch._dynamo
# Attempt to disable dynamo optimization if implicit
try:
    torch._dynamo.config.suppress_errors = True
    torch._dynamo.config.disable = True
except:
    pass

import math
import time
import wandb
import shutil
from pathlib import Path
from tqdm.auto import tqdm
from dataclasses import asdict
from datetime import datetime
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from accelerate import DeepSpeedPlugin
import logging
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings("ignore")

# Add src to path
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))
sys.path.insert(0, str(project_root))

# Local imports
try:
    from config import load_config
    from models.alignment import MultimodalAlignmentModel
    from data.dataset_builder import build_image_datasets_from_parquet
    from data.transforms import get_image_transforms
except ImportError as e:
    print(f"Error importing local modules: {e}")
    print(f"sys.path: {sys.path}")
    raise

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Pixmo Vision-Text Alignment with Accelerate")
    
    # Config arguments
    parser.add_argument("--config", type=str, 
                        default=str(project_root / "configs/pixmo_alignment.yaml"),
                        help="Path to the config file")
    
    # Training overrides
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (overrides config)")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size per device")
    parser.add_argument("--learning_rate", type=float, default=None, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    
    # Checkpointing
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Resume training from checkpoint")
    
    # WandB and logging
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases")
    parser.add_argument("--run_name", type=str, default=None, help="WandB run name (overrides config)")
    
    # Mode options
    parser.add_argument("--max_steps", type=int, default=None, help="Max training steps (for debugging)")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode (fewer steps, more logging)")
    parser.add_argument("--deepspeed_stage", type=int, default=2, choices=[0, 1, 2, 3], help="ZeRO stage for DeepSpeed")
    
    # Model optimization
    parser.add_argument("--disable_decoder", action="store_true", 
                        help="Disable the decoder to save memory (for alignment-only training)")
    
    return parser.parse_args()


def get_lr_scheduler(optimizer, warmup_steps, total_steps):
    """Warmup + Cosine decay schedule (matching VLM training)."""
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def collate_fn(batch):
    """Collate function for alignment dataset (image + text pairs)."""
    # Filter None items if any
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    
    images = torch.stack([b['image'] for b in batch])
    texts = [b['text'] for b in batch]
    
    return {
        'images': images,
        'texts': texts,
    }


def main():
    args = parse_args()
    
    # Load config
    if not Path(args.config).exists():
        print(f"Error: Config file not found at {args.config}")
        sys.exit(1)
        
    config = load_config(args.config)
    
    # Apply CLI overrides
    if args.output_dir:
        config.trainer.output_dir = args.output_dir
        config.trainer.ckpt_dir = os.path.join(args.output_dir, "checkpoints")
    if args.batch_size:
        config.dataset.batch_size = args.batch_size
    if args.learning_rate:
        config.optimization.lr = args.learning_rate
        config.optimization.learning_rate = args.learning_rate
    if args.num_epochs:
        config.trainer.epochs = args.num_epochs
        config.trainer.num_epochs = args.num_epochs
    if args.run_name:
        config.trainer.wandb_run_name = args.run_name
        
    # Debug mode overrides
    if args.debug:
        args.max_steps = args.max_steps or 50
        config.trainer.log_every = 5
        logger.info("Debug mode enabled: max_steps=50, log_every=5")
        
    # Disable decoder for alignment-only training
    if args.disable_decoder and hasattr(config, 'decoder'):
        config.decoder = None
        logger.info("Decoder disabled for alignment-only training")
    
    # Ensure output dir exists on trainer config
    output_dir = getattr(config.trainer, 'output_dir', './outputs/pixmo_alignment')
    ckpt_dir = getattr(config.trainer, 'ckpt_dir', os.path.join(output_dir, 'checkpoints'))
    
    # Setup Accelerator
    ds_plugin = DeepSpeedPlugin(
        zero_stage=args.deepspeed_stage,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_clipping=getattr(config.optimization, 'max_grad_norm', 1.0),
    )
    
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=getattr(config.optimization, 'mixed_precision', 'bf16'),
        log_with="wandb" if (args.use_wandb or getattr(config.trainer, 'use_wandb', False)) else None,
        deepspeed_plugin=ds_plugin,
        project_config=ProjectConfiguration(
            project_dir=output_dir,
            logging_dir=str(Path(output_dir) / "logs")
        )
    )
    
    # Ensure sentence-transformers loads on the accelerator device (GPU if available)
    os.environ["SENTENCE_TRANSFORMERS_DEFAULT_DEVICE"] = accelerator.device.type
    
    # Logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    
    # Set seed
    set_seed(args.seed)
    
    # Create output directories
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(ckpt_dir, exist_ok=True)
        
        # Init WandB
        if args.use_wandb or getattr(config.trainer, 'use_wandb', False):
            accelerator.init_trackers(
                project_name=getattr(config.trainer, 'wandb_project', 'edge_glass_alignment'),
                config=vars(args),
                init_kwargs={"wandb": {"name": getattr(config.trainer, 'wandb_run_name', 'pixmo_alignment')}}
            )
    
    # 1. Initialize Model
    logger.info("Initializing MultimodalAlignmentModel...")
    model = MultimodalAlignmentModel(config)
    
    # Log trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    # 2. Datasets
    logger.info("Loading datasets...")
    train_transforms = get_image_transforms(config.dataset.image_size, is_training=True)
    val_transforms = get_image_transforms(config.dataset.image_size, is_training=False)
    
    datasets = build_image_datasets_from_parquet(
        cfg=config,
        train_parquet_path=config.dataset.train_parquet,
        val_parquet_path=config.dataset.val_parquet,
        test_parquet_path=getattr(config.dataset, 'test_parquet', None),
        train_transforms=train_transforms,
        val_transforms=val_transforms,
        max_text_length=getattr(config.dataset, 'max_text_length', 512),
        text_dropout_prob=getattr(config.dataset, 'text_dropout_prob', 0.1),
    )
    
    train_dataset = datasets['train']
    val_dataset = datasets['val']
    
    if accelerator.is_main_process:
        logger.info(f"Train samples: {len(train_dataset)}")
        logger.info(f"Validation samples: {len(val_dataset)}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.dataset.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=getattr(config.dataset, 'pin_memory', True),
        collate_fn=collate_fn,
        persistent_workers=getattr(config.dataset, 'persistent_workers', False) if args.num_workers > 0 else False,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.dataset.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=getattr(config.dataset, 'pin_memory', True),
        collate_fn=collate_fn,
    )
    
    # 3. Optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=getattr(config.optimization, 'lr', 2e-4),
        weight_decay=getattr(config.optimization, 'weight_decay', 0.01),
        betas=tuple(getattr(config.optimization, 'betas', [0.9, 0.95])),
    )
    
    # 4. Learning Rate Scheduler
    num_epochs = getattr(config.trainer, 'epochs', 10)
    num_update_steps_per_epoch = len(train_loader) // args.gradient_accumulation_steps
    max_train_steps = num_epochs * num_update_steps_per_epoch
    
    if args.max_steps:
        max_train_steps = args.max_steps
        
    warmup_ratio = getattr(config.optimization, 'warmup_ratio', 0.1)
    warmup_steps = int(max_train_steps * warmup_ratio)
    
    scheduler = get_lr_scheduler(optimizer, warmup_steps, max_train_steps)
    
    logger.info(f"Training for {max_train_steps} steps ({num_epochs} epochs)")
    logger.info(f"Warmup steps: {warmup_steps}")
    
    # 5. Prepare with Accelerate
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )
    
    # 6. Load checkpoint if resuming
    global_step = 0
    start_epoch = 0
    best_val_loss = float('inf')
    
    # Auto-detect checkpoint if not specified
    if args.resume_from_checkpoint is None:
        ckpt_path = Path(ckpt_dir)
        if ckpt_path.exists():
            # Get all checkpoints sorted by epoch (newest first)
            all_checkpoints = sorted(ckpt_path.glob("checkpoint-epoch-*"), 
                                     key=lambda p: int(p.name.split('-')[-1]), reverse=True)
            
            for ckpt in all_checkpoints:
                # Validate checkpoint completeness
                if (ckpt / "scheduler.bin").exists():
                    logger.info(f"Auto-detected latest VALID checkpoint: {ckpt}")
                    args.resume_from_checkpoint = str(ckpt)
                    break
                else:
                    logger.warning(f"Skipping incomplete/corrupted checkpoint: {ckpt}")
    
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint == "latest":
            ckpt_path = Path(ckpt_dir)
            all_checkpoints = sorted(ckpt_path.glob("checkpoint-epoch-*"), 
                                     key=lambda p: int(p.name.split('-')[-1]), reverse=True)
            found = False
            for ckpt in all_checkpoints:
                if (ckpt / "scheduler.bin").exists():
                    args.resume_from_checkpoint = str(ckpt)
                    logger.info(f"Auto-detected latest VALID checkpoint: {args.resume_from_checkpoint}")
                    found = True
                    break
            if not found:
                logger.warning("No valid checkpoints found for 'latest' resume. Starting from scratch.")
                args.resume_from_checkpoint = None
    
    if args.resume_from_checkpoint:
        logger.info(f"Resuming from {args.resume_from_checkpoint}")
        load_kwargs = {}
        if accelerator.distributed_type == "DEEPSPEED":
            load_kwargs["load_module_strict"] = False
        else:
            load_kwargs["strict"] = False
            
        accelerator.load_state(args.resume_from_checkpoint, **load_kwargs)
        
        # Parse epoch from checkpoint name
        try:
            ckpt_name = Path(args.resume_from_checkpoint).name
            if "checkpoint-epoch-" in ckpt_name:
                start_epoch = int(ckpt_name.split('-')[-1])
                logger.info(f"Resuming from epoch {start_epoch}")
        except ValueError:
            logger.warning(f"Could not parse epoch from checkpoint name: {args.resume_from_checkpoint}")
    
    # 7. Training Loop
    logger.info("Starting training...")
    
    for epoch in range(start_epoch, num_epochs):
        model.train()
        total_loss = 0.0
        epoch_clip_loss = 0.0
        epoch_mrl_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, disable=not accelerator.is_local_main_process, desc=f"Epoch {epoch}")
        
        for batch in progress_bar:
            if batch is None:
                continue
                
            with accelerator.accumulate(model):
                # Forward pass
                images = batch['images']
                texts = batch['texts']
                
                outputs = model(images=images, texts=texts)
                loss = outputs.loss
                
                if loss is None:
                    logger.warning("Loss is None, skipping batch")
                    continue
                
                accelerator.backward(loss)
                
                # Gradient clipping and norm
                total_norm = 0.0
                if accelerator.sync_gradients:
                    max_grad_norm = getattr(config.optimization, 'max_grad_norm', 1.0)
                    total_norm = accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
                    if isinstance(total_norm, torch.Tensor):
                        total_norm = total_norm.item()
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
            # Logging
            if accelerator.sync_gradients:
                global_step += 1
                total_loss += loss.item()
                num_batches += 1
                
                # Track component losses
                if outputs.losses:
                    if 'loss_clip' in outputs.losses:
                        epoch_clip_loss += outputs.losses['loss_clip'].item()
                    if 'loss_mrl' in outputs.losses:
                        epoch_mrl_loss += outputs.losses['loss_mrl'].item()
                
                log_every = getattr(config.trainer, 'log_every', 20)
                if global_step % log_every == 0 and accelerator.is_main_process:
                    current_loss = loss.item()
                    lr = scheduler.get_last_lr()[0]
                    
                    log_data = {
                        "train/loss": current_loss,
                        "train/grad_norm": total_norm,
                        "train/lr": lr,
                        "train/epoch": epoch,
                        "train/step": global_step,
                    }
                    
                    # Add component losses
                    if outputs.losses:
                        if 'loss_clip' in outputs.losses:
                            log_data["train/loss_clip"] = outputs.losses['loss_clip'].item()
                        if 'loss_mrl' in outputs.losses:
                            log_data["train/loss_mrl"] = outputs.losses['loss_mrl'].item()
                    
                    accelerator.log(log_data, step=global_step)
                    progress_bar.set_postfix({'loss': f"{current_loss:.4f}", 'lr': f"{lr:.2e}"})
                
                # Debug mode early stopping
                if args.max_steps and global_step >= args.max_steps:
                    logger.info(f"Reached max steps {args.max_steps}, stopping.")
                    # Save debug checkpoint
                    debug_ckpt = Path(ckpt_dir) / "checkpoint-debug"
                    accelerator.save_state(debug_ckpt)
                    if accelerator.is_main_process:
                        logger.info(f"Saved debug checkpoint to {debug_ckpt}")
                    break
        
        if args.max_steps and global_step >= args.max_steps:
            break
        
        # 8. Validation
        model.eval()
        val_loss = 0.0
        val_clip_loss = 0.0
        val_mrl_loss = 0.0
        val_batches = 0
        
        # Collect embeddings for retrieval metrics
        vision_embeddings = []
        text_embeddings = []
        
        logger.info("Running validation...")
        
        for batch in tqdm(val_loader, disable=not accelerator.is_local_main_process, desc="Validation"):
            if batch is None:
                continue
                
            with torch.no_grad():
                images = batch['images']
                texts = batch['texts']
                
                outputs = model(images=images, texts=texts, return_embeddings=True)
                
                if outputs.loss is not None:
                    val_loss += outputs.loss.item()
                    val_batches += 1
                    
                    if outputs.losses:
                        if 'loss_clip' in outputs.losses:
                            val_clip_loss += outputs.losses['loss_clip'].item()
                        if 'loss_mrl' in outputs.losses:
                            val_mrl_loss += outputs.losses['loss_mrl'].item()
                
                # Collect embeddings for retrieval
                if outputs.vision_emb is not None:
                    vision_embeddings.append(outputs.vision_emb.detach().cpu())
                if outputs.text_emb is not None:
                    text_embeddings.append(outputs.text_emb.detach().cpu())
        
        # Compute average validation metrics
        if val_batches > 0:
            avg_val_loss = val_loss / val_batches
            avg_val_clip = val_clip_loss / val_batches
            avg_val_mrl = val_mrl_loss / val_batches
        else:
            avg_val_loss = float('inf')
            avg_val_clip = 0.0
            avg_val_mrl = 0.0
        
        # Compute retrieval metrics
        retrieval_metrics = {}
        if vision_embeddings and text_embeddings:
            vision_embs = torch.cat(vision_embeddings, dim=0)
            text_embs = torch.cat(text_embeddings, dim=0)
            
            similarity = torch.matmul(vision_embs, text_embs.t())
            ranks = torch.argsort(similarity, dim=1, descending=True)
            correct_indices = torch.arange(len(vision_embs)).unsqueeze(1)
            num_samples = len(vision_embs)
            
            r1 = (ranks[:, :1] == correct_indices).any(dim=1).float().mean().item()
            r5 = (ranks[:, :5] == correct_indices).any(dim=1).float().mean().item()
            r10 = (ranks[:, :10] == correct_indices).any(dim=1).float().mean().item()
            
            # R@1k and R@5k (capped by actual number of samples)
            k1000 = min(1000, num_samples)
            k5000 = min(5000, num_samples)
            r1k = (ranks[:, :k1000] == correct_indices).any(dim=1).float().mean().item()
            r5k = (ranks[:, :k5000] == correct_indices).any(dim=1).float().mean().item()
            
            retrieval_metrics = {
                'i2t_r1': r1, 'i2t_r5': r5, 'i2t_r10': r10,
                'i2t_r1k': r1k, 'i2t_r5k': r5k
            }
        
        # Log validation metrics
        if accelerator.is_main_process:
            logger.info(f"Epoch {epoch} Validation Loss: {avg_val_loss:.4f} | CLIP: {avg_val_clip:.4f} | MRL: {avg_val_mrl:.4f}")
            if retrieval_metrics:
                logger.info(f"  R@1: {retrieval_metrics['i2t_r1']*100:.2f}% | R@5: {retrieval_metrics['i2t_r5']*100:.2f}% | R@10: {retrieval_metrics['i2t_r10']*100:.2f}%")
                logger.info(f"  R@1k: {retrieval_metrics['i2t_r1k']*100:.2f}% | R@5k: {retrieval_metrics['i2t_r5k']*100:.2f}%")
            
            val_log_data = {
                "val/loss": avg_val_loss,
                "val/loss_clip": avg_val_clip,
                "val/loss_mrl": avg_val_mrl,
                **{f"val/{k}": v for k, v in retrieval_metrics.items()},
            }
            accelerator.log(val_log_data, step=global_step)
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_ckpt = Path(ckpt_dir) / "checkpoint_best.pt"
                
                logger.info(f"New best model! Saving to {best_ckpt}...")
                save_start = time.time()
                
                unwrapped_model = accelerator.unwrap_model(model)
                best_checkpoint = {
                    'model_state_dict': unwrapped_model.state_dict(),
                    'config': asdict(config),
                    'mrl_dims': getattr(config.vision_encoder, 'mrl_dimensions', []),
                    'projection_dim': getattr(config.vision_encoder, 'projection_dim', 4096),
                    'best_val_loss': best_val_loss,
                    'epoch': epoch,
                    'global_step': global_step,
                    'training_date': datetime.now().isoformat(),
                }
                torch.save(best_checkpoint, best_ckpt)
                
                logger.info(f"Saved best checkpoint (Loss: {best_val_loss:.4f}). Time: {time.time() - save_start:.2f}s")
            else:
                logger.info(f"Validation loss {avg_val_loss:.4f} did not improve from {best_val_loss:.4f}")
        
        # Save epoch checkpoint (full state for resume)
        epoch_ckpt = Path(ckpt_dir) / f"checkpoint-epoch-{epoch+1}"
        logger.info(f"Saving epoch checkpoint to {epoch_ckpt}...")
        accelerator.save_state(epoch_ckpt)
        
        # Checkpoint rotation: Keep only last 2 epochs, but preserve every 7th
        all_checkpoints = sorted(Path(ckpt_dir).glob("checkpoint-epoch-*"), 
                                 key=lambda p: int(p.name.split('-')[-1]))
        
        if len(all_checkpoints) > 2:
            candidates_to_delete = all_checkpoints[:-2]
            
            for old_ckpt in candidates_to_delete:
                try:
                    ckpt_epoch = int(old_ckpt.name.split('-')[-1])
                    if ckpt_epoch % 7 == 0:
                        logger.info(f"Preserving checkpoint {old_ckpt} (Epoch {ckpt_epoch} is multiple of 7)")
                        continue
                except ValueError:
                    pass
                
                if accelerator.is_main_process:
                    logger.info(f"Deleting old checkpoint {old_ckpt}...")
                    try:
                        shutil.rmtree(old_ckpt)
                    except Exception as e:
                        logger.warning(f"Failed to delete {old_ckpt}: {e}")
    
    # 9. Final save
    if global_step > 0:
        final_ckpt = Path(ckpt_dir) / "checkpoint-final"
        logger.info(f"Saving final checkpoint to {final_ckpt}...")
        save_start = time.time()
        accelerator.save_state(final_ckpt)
        
        if accelerator.is_main_process:
            # Also save a lightweight final checkpoint
            unwrapped_model = accelerator.unwrap_model(model)
            final_lightweight = {
                'model_state_dict': unwrapped_model.state_dict(),
                'config': asdict(config),
                'mrl_dims': getattr(config.vision_encoder, 'mrl_dimensions', []),
                'projection_dim': getattr(config.vision_encoder, 'projection_dim', 4096),
                'best_val_loss': best_val_loss,
                'num_epochs': num_epochs,
                'total_steps': global_step,
                'training_date': datetime.now().isoformat(),
            }
            torch.save(final_lightweight, Path(ckpt_dir) / "pixmo_alignment_final.pt")
            
            logger.info(f"Saved final checkpoint. Time: {time.time() - save_start:.2f}s")
            logger.info("=" * 60)
            logger.info("Training Completed!")
            logger.info(f"Best Validation Loss: {best_val_loss:.4f}")
            logger.info(f"Final checkpoint: {final_ckpt}")
            logger.info(f"Best checkpoint: {Path(ckpt_dir) / 'checkpoint_best.pt'}")
            logger.info("=" * 60)
    
    accelerator.end_training()


if __name__ == "__main__":
    main()
