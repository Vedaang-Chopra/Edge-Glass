
import sys
import os
from pathlib import Path
import torch
import torch.distributed as dist
import torch.nn.functional as F
import numpy as np
from datetime import datetime
from dataclasses import asdict

# Add src to path
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from config import load_config
from models.alignment import MultimodalAlignmentModel
from data.dataset_builder import build_image_datasets_from_parquet
from data.transforms import get_image_transforms
from training.improved_trainer import ImprovedMultimodalTrainer
from utils import init_distributed, get_rank

def main():
    # 1. Configuration
    config_path = project_root / "configs/perceiver_mrl_alignment.yaml"
    print(f"Loading config from {config_path}")
    config = load_config(str(config_path))
    
    # 1.5 Config Overrides for Stability
    # Disable decoder for alignment training (saves massive memory/compute)
    if hasattr(config, 'decoder'):
        print("Disabling decoder for alignment-only training...")
        config.decoder = None
        
    # Lower learning rate if it's too high
    if config.optimization.lr > 0.0005:
        print(f"Lowering learning rate from {config.optimization.lr} to 2e-4 for stability")
        config.optimization.lr = 0.0002
        config.optimization.learning_rate = 0.0002
        
    # 2. Distributed Setup
    # Check if we are running in a distributed environment (e.g. torchrun or accelerate)
    is_distributed = "RANK" in os.environ or "LOCAL_RANK" in os.environ
    if is_distributed:
        print("Distributed environment detected. Initializing DDP...")
        init_distributed()
        config.trainer.strategy = "ddp"
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        print("Single process execution.")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 3. Model
    if get_rank() == 0:
        print("Initializing model...")
    model = MultimodalAlignmentModel(config)
    model.to(device)
    
    # Parameter diagnostic
    if get_rank() == 0:
        print("\n=== Parameter Diagnostic ===")
        total_trainable = 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                total_trainable += param.numel()
                # Print first 10 trainable params for debugging
        print(f"Total trainable parameters: {total_trainable:,}")
        
        # Check per-component
        if model.vision_encoder is not None:
            ve_trainable = sum(p.numel() for p in model.vision_encoder.parameters() if p.requires_grad)
            print(f"  Vision Encoder trainable: {ve_trainable:,}")
        if model.text_encoder is not None:
            te_trainable = sum(p.numel() for p in model.text_encoder.parameters() if p.requires_grad)
            print(f"  Text Encoder trainable: {te_trainable:,}")
        print("===========================\n")
    
    # 4. Dataset
    if get_rank() == 0:
        print("Loading datasets...")
    train_transforms = get_image_transforms(image_size=config.dataset.image_size, is_training=True)
    val_transforms = get_image_transforms(image_size=config.dataset.image_size, is_training=False)
    
    datasets = build_image_datasets_from_parquet(
        cfg=config,
        train_parquet_path=config.dataset.train_parquet,
        val_parquet_path=config.dataset.val_parquet,
        test_parquet_path=config.dataset.test_parquet,
        train_transforms=train_transforms,
        val_transforms=val_transforms,
        max_text_length=config.dataset.max_text_length,
        text_dropout_prob=config.dataset.text_dropout_prob,
    )
    
    train_dataset = datasets['train']
    val_dataset = datasets['val']
    
    # Samplers for DDP
    train_sampler = None
    shuffle = True
    if is_distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
        shuffle = False # Sampler handles shuffle
        
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.dataset.batch_size,
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=config.dataset.num_workers,
        pin_memory=config.dataset.pin_memory,
        persistent_workers=config.dataset.persistent_workers if config.dataset.num_workers > 0 else False,
    )
    
    val_sampler = None
    if is_distributed:
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
        
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.trainer.eval_batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=config.dataset.num_workers,
        pin_memory=config.dataset.pin_memory,
    )

    
    # 5. Training
    if get_rank() == 0:
        print("Initializing trainer...")
        
    trainer = ImprovedMultimodalTrainer(
        cfg=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        use_wandb=True
    )
    
    if get_rank() == 0:
        print("Starting training...")
        
    # Verify gradients before starting
    print("Running pre-flight gradient check...")
    try:
        model.train()
        batch = next(iter(train_loader))
        images = batch['image'].to(device)
        texts = batch['text']
        
        # Determine mixed precision context
        use_amp = config.optimization.fp16 or config.optimization.bf16
        with torch.cuda.amp.autocast(enabled=use_amp):
             outputs = model(images=images, texts=texts)
             loss = outputs.loss
             
        if loss is not None:
            loss.backward()
            gradients_present = False
            for name, p in model.named_parameters():
                if p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0:
                    if "perceiver" in name:
                        gradients_present = True
                        break
            
            if gradients_present:
                print("✓ Pre-flight check passed: Perceiver gradients detected.")
            else:
                print("⚠️ WARNING: No gradients detected in Perceiver during pre-flight!")
                print("DEBUG: Checking trainable parameters:")
                for name, p in model.named_parameters():
                    if p.requires_grad:
                        grad_status = "None" if p.grad is None else f"Zero: {p.grad.abs().sum() == 0}"
                        # Check vision encoder specific parts
                        if "vision_encoder" in name:
                            print(f"  - {name}: requires_grad=True, grad={grad_status}")
                        elif "perceiver" in name: # specific check if using direct perceiver module
                            print(f"  - {name}: requires_grad=True, grad={grad_status}")
            
            model.zero_grad()
        else:
            print("⚠️ WARNING: Pre-flight check returned None loss.")
            
    except Exception as e:
        print(f"⚠️ Pre-flight check failed (non-fatal): {e}")

    history = trainer.train()
    
    # 6. Save Final Checkpoint (Only Rank 0)
    if get_rank() == 0:
        print("Saving final checkpoint for next stage...")
        
        # We need the best val loss from the trainer state
        best_val_loss = trainer.state.best_val_loss
        global_step = trainer.state.global_step
        
        # Prepare final checkpoint dict as per notebook
        # We need to access model.module if DDP
        raw_model = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        
        final_checkpoint = {
            # Model state
            'model_state_dict': raw_model.state_dict(),
            
            # Configuration
            'config': asdict(config),
            
            # Perceiver configuration for reconstruction
            'perceiver_config': {
                'num_latents': config.vision_encoder.perceiver_num_latents,
                'latent_dim': config.vision_encoder.perceiver_latent_dim,
                'num_layers': config.vision_encoder.perceiver_num_layers,
                'num_heads': config.vision_encoder.perceiver_num_heads,
                'dropout': config.vision_encoder.perceiver_dropout,
            },
            
            # MRL configuration
            'mrl_dims': config.vision_encoder.mrl_dimensions,
            'projection_dim': config.vision_encoder.projection_dim,
            
            # Vision encoder info
            'vision_encoder_name': config.vision_encoder.model_name,
            'text_encoder_name': config.text_encoder.model_name if hasattr(config, 'text_encoder') else 'unknown',

            'best_val_loss': best_val_loss,
            'final_metrics': history,
            
            # Metadata
            'training_date': datetime.now().isoformat(),
            'num_epochs': config.trainer.epochs,
            'total_steps': global_step,
        }
        
        # Try to add extra encoder names if they exist in config or hardcode as per notebook context
        # In notebook: cfg.vision_encoder_name was used. 
        # But cell 7 prints: Experiment: perceiver_mrl_alignment
        # Let's hope asdict(config) covers most.
        # But for 'vision_encoder_name', let's check config structure? 
        # I'll stick to what I extracted from config object.
        if hasattr(config.vision_encoder, 'model_name'):
             final_checkpoint['vision_encoder_name'] = config.vision_encoder.model_name
             
        # Save
        ckpt_dir = Path(config.trainer.ckpt_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        final_ckpt_path = ckpt_dir / "perceiver_mrl_alignment_final.pt"
        
        torch.save(final_checkpoint, final_ckpt_path)
        
        print("\n" + "="*60)
        print("FINAL MODEL SAVED")
        print("="*60)
        print(f"Checkpoint: {final_ckpt_path}")

if __name__ == "__main__":
    main()
