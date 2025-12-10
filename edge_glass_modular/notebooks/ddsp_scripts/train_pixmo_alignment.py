import sys
import os
import argparse
from pathlib import Path
import torch
import torch.distributed as dist
import numpy as np
from datetime import datetime
from dataclasses import asdict

# Add src to path
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from config import load_config
from models import MultimodalAlignmentModel
from data.dataset_builder import build_image_datasets_from_parquet
from data.transforms import get_image_transforms
from training.improved_trainer import ImprovedMultimodalTrainer
from utils import init_distributed, get_rank

def parse_args():
    parser = argparse.ArgumentParser(description="Train Pixmo Vision-Text Alignment Model")
    
    # Config arguments
    parser.add_argument("--config", type=str, default="configs/pixmo_alignment.yaml",
                        help="Path to the config file (relative to project root)")
    
    # Training overrides
    parser.add_argument("--lr", type=float, help="Override learning rate")
    parser.add_argument("--batch-size", type=int, help="Override batch size per GPU")
    parser.add_argument("--epochs", type=int, help="Override number of epochs")
    parser.add_argument("--output-dir", type=str, help="Override output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Model optimization
    parser.add_argument("--disable-decoder", action="store_true", 
                        help="Disable the decoder to save memory (if only training alignment)")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 1. Configuration
    config_path = project_root / args.config
    if get_rank() == 0:
        print(f"Loading config from {config_path}")
    
    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)
        
    config = load_config(str(config_path))
    
    # Apply overrides
    if args.lr:
        config.optimization.lr = args.lr
        config.optimization.learning_rate = args.lr
    if args.batch_size:
        config.dataset.batch_size = args.batch_size
    if args.epochs:
        config.trainer.epochs = args.epochs
    if args.output_dir:
        config.trainer.output_dir = args.output_dir
        config.trainer.ckpt_dir = os.path.join(args.output_dir, "checkpoints")

    # 1.5 Config Overrides for Stability / Optimization
    if args.disable_decoder:
        if hasattr(config, 'decoder'):
            if get_rank() == 0:
                print("Disabling decoder for alignment-only training...")
            config.decoder = None
        
    # 2. Distributed Setup
    is_distributed = "RANK" in os.environ or "LOCAL_RANK" in os.environ
    if is_distributed:
        if get_rank() == 0:
            print("Distributed environment detected. Initializing DDP...")
        init_distributed()
        config.trainer.strategy = "ddp"
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        if get_rank() == 0:
            print("Single process execution.")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 3. Model
    if get_rank() == 0:
        print("Initializing model...")
    model = MultimodalAlignmentModel(config)
    model.to(device)
    
    if get_rank() == 0:
        print("="*60)
        # model.print_parameter_counts() # Optional: if the model has this method
        print("="*60)

    # 4. Dataset
    if get_rank() == 0:
        print("Loading datasets...")
        
    train_transforms = get_image_transforms(image_size=config.dataset.image_size, is_training=True)
    val_transforms = get_image_transforms(image_size=config.dataset.image_size, is_training=False)
    
    # Note: dataset paths in config are usually absolute or relative to some understood root.
    # The loading function handles reading.
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
    
    if get_rank() == 0:
        print(f"Train samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
    
    # Samplers for DDP
    train_sampler = None
    shuffle = True
    if is_distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
        shuffle = False 
        
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
        batch_size=config.dataset.batch_size, # Using same batch size for simplicity, or add eval_batch_size if needed
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
        
    # Pre-flight check (optional, but good practice from reference script)
    if get_rank() == 0:
        print("Running pre-flight check...")
        
    # Run training
    history = trainer.train()
    
    # 6. Save Final Checkpoint (Only Rank 0)
    if get_rank() == 0:
        print("Saving final checkpoint...")
        
        best_val_loss = getattr(trainer.state, 'best_val_loss', 0.0)
        global_step = getattr(trainer.state, 'global_step', 0)
        
        # Unwrap model
        raw_model = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        
        final_checkpoint = {
            'model_state_dict': raw_model.state_dict(),
            'config': asdict(config),
            'mrl_dims': getattr(config.vision_encoder, 'mrl_dimensions', []),
            'projection_dim': getattr(config.vision_encoder, 'projection_dim', 4096),
            'best_val_loss': best_val_loss,
            'final_metrics': history,
            'training_date': datetime.now().isoformat(),
            'num_epochs': config.trainer.epochs,
            'total_steps': global_step,
        }
        
        # Save
        ckpt_dir = Path(config.trainer.ckpt_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        final_ckpt_path = ckpt_dir / "pixmo_alignment_final.pt"
        
        torch.save(final_checkpoint, final_ckpt_path)
        
        print("\n" + "="*60)
        print("Training Completed and Model Saved")
        print(f"Checkpoint: {final_ckpt_path}")
        print("="*60)

if __name__ == "__main__":
    main()
