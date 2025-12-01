"""
train_with_in_memory_datasets.py - Training script using in-memory datasets

This script demonstrates how to use the InMemoryImageTextDataset and
InMemoryAudioTextDataset for faster training by pre-loading all data into memory.
"""

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset

# Import your existing modules
from imports.core import (
    AlignmentConfig,
    VisionTextAligner,
    set_seed,
    count_parameters,
)

# Import the new in-memory datasets
from imports.in_memory_datasets import (
    InMemoryImageTextDataset,
    InMemoryAudioTextDataset,
    collate_in_memory_images,
    collate_in_memory_audio,
)


def create_in_memory_image_dataloader(
    max_samples: int = 10000,
    batch_size: int = 32,
    num_workers: int = 4,
):
    """
    Create DataLoader with pre-loaded images.

    Args:
        max_samples: Number of samples to load (None for all)
        batch_size: Batch size for training
        num_workers: Number of DataLoader workers

    Returns:
        DataLoader with images in memory
    """
    print("\n" + "=" * 60)
    print("Loading PixMo-Cap Dataset")
    print("=" * 60)

    # Load the HuggingFace dataset
    hf_dataset = load_dataset("allenai/pixmo-cap", split="train")

    # Create in-memory dataset (this will pre-load all images)
    dataset = InMemoryImageTextDataset(
        hf_dataset=hf_dataset,
        img_col="image_url",
        txt_col="caption",
        max_samples=max_samples,
        image_size=(224, 224),
    )

    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_in_memory_images,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    print(f"\n✅ DataLoader ready with {len(dataset)} samples")
    print(f"   Batches per epoch: {len(dataloader)}")

    return dataloader


def create_in_memory_audio_dataloader(
    dataset_name: str = "google/MusicCaps",
    max_samples: int = 5000,
    batch_size: int = 32,
    num_workers: int = 4,
):
    """
    Create DataLoader with pre-loaded audio.

    Args:
        dataset_name: HuggingFace dataset name
        max_samples: Number of samples to load (None for all)
        batch_size: Batch size for training
        num_workers: Number of DataLoader workers

    Returns:
        DataLoader with audio in memory
    """
    print("\n" + "=" * 60)
    print(f"Loading Audio Dataset: {dataset_name}")
    print("=" * 60)

    # Load the HuggingFace dataset
    # Note: You may need to adjust the split and columns based on the dataset
    try:
        hf_dataset = load_dataset(dataset_name, split="train")
    except Exception as e:
        print(f"Error loading {dataset_name}: {e}")
        print("Trying alternative datasets...")
        # Try alternative
        hf_dataset = load_dataset("google/MusicCaps", split="train")

    # Determine audio and text columns
    cols = hf_dataset.column_names
    audio_col = "audio" if "audio" in cols else "audio"
    txt_col = "caption" if "caption" in cols else ("text" if "text" in cols else "caption_writing")

    print(f"   Using columns: audio={audio_col}, text={txt_col}")

    # Create in-memory dataset (this will pre-load all audio)
    dataset = InMemoryAudioTextDataset(
        hf_dataset=hf_dataset,
        audio_col=audio_col,
        txt_col=txt_col,
        max_samples=max_samples,
        target_sr=16000,
        max_duration=30.0,
    )

    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_in_memory_audio,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    print(f"\n✅ DataLoader ready with {len(dataset)} samples")
    print(f"   Batches per epoch: {len(dataloader)}")

    return dataloader


def train_with_in_memory_data():
    """
    Main training function using in-memory datasets.
    """
    print("\n" + "=" * 60)
    print("TRAINING WITH IN-MEMORY DATASETS")
    print("=" * 60)

    # Set seed for reproducibility
    set_seed(42)

    # Check available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"\n🖥️  Available GPUs: {num_gpus}")
    if num_gpus > 0:
        for i in range(num_gpus):
            print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")

    # Create configuration
    cfg = AlignmentConfig(
        batch_size=32 * max(1, num_gpus),  # Scale batch size with GPU count
        learning_rate=1e-4,
        num_epochs=5,
        max_grad_norm=1.0,
    )

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg.device = device
    print(f"\n🖥️  Primary device: {device}")
    print(f"   Effective batch size: {cfg.batch_size}")

    # ============================================================
    # 1. Create In-Memory DataLoaders
    # ============================================================

    print("\n" + "=" * 60)
    print("STEP 1: Loading Data into Memory")
    print("=" * 60)

    # Create image dataloader (this will pre-load all images)
    # Increase num_workers to leverage more CPU cores
    num_dataloader_workers = min(8, torch.get_num_threads() // 2) if num_gpus > 0 else 4
    image_dataloader = create_in_memory_image_dataloader(
        max_samples=10000,  # Adjust based on available memory
        batch_size=cfg.batch_size,
        num_workers=num_dataloader_workers,
    )

    # Optional: Create audio dataloader
    # audio_dataloader = create_in_memory_audio_dataloader(
    #     dataset_name="google/MusicCaps",
    #     max_samples=5000,
    #     batch_size=cfg.batch_size,
    #     num_workers=4,
    # )

    # ============================================================
    # 2. Create Model
    # ============================================================

    print("\n" + "=" * 60)
    print("STEP 2: Creating Model")
    print("=" * 60)

    model = VisionTextAligner(cfg).to(device)

    # Wrap model with DataParallel for multi-GPU training
    if num_gpus > 1:
        print(f"\n🔗 Wrapping model with DataParallel for {num_gpus} GPUs")
        model = torch.nn.DataParallel(model)
        print(f"   Model will use GPUs: {list(range(num_gpus))}")

    params = count_parameters(model)
    print(f"\n📊 Model Parameters:")
    print(f"   Total: {params['total']:,}")
    print(f"   Trainable: {params['trainable']:,}")
    print(f"   Frozen: {params['frozen']:,}")

    # ============================================================
    # 3. Setup Optimizer
    # ============================================================

    # Handle DataParallel wrapper for getting trainable params
    model_for_params = model.module if isinstance(model, torch.nn.DataParallel) else model
    optimizer = torch.optim.AdamW(
        model_for_params.get_trainable_params(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    # ============================================================
    # 4. Training Loop
    # ============================================================

    print("\n" + "=" * 60)
    print("STEP 3: Training")
    print("=" * 60)

    model.train()
    global_step = 0

    for epoch in range(cfg.num_epochs):
        print(f"\n📍 Epoch {epoch + 1}/{cfg.num_epochs}")
        print("-" * 60)

        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(image_dataloader):
            images = batch["images"]  # List of PIL images
            texts = batch["captions"]  # List of strings

            # Forward pass
            outputs = model(images=images, texts=texts)
            loss = outputs["loss"]

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping (handle DataParallel wrapper)
            torch.nn.utils.clip_grad_norm_(
                model_for_params.get_trainable_params(),
                cfg.max_grad_norm,
            )

            optimizer.step()

            # Track metrics
            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1

            # Log progress
            if (batch_idx + 1) % cfg.log_every == 0:
                avg_loss = epoch_loss / num_batches
                print(
                    f"   Step {global_step:5d} | "
                    f"Batch {batch_idx + 1:4d}/{len(image_dataloader)} | "
                    f"Loss: {loss.item():.4f} | "
                    f"Avg Loss: {avg_loss:.4f}"
                )

        # Epoch summary
        avg_epoch_loss = epoch_loss / num_batches
        print(f"\n✅ Epoch {epoch + 1} completed | Avg Loss: {avg_epoch_loss:.4f}")

        # Save checkpoint
        checkpoint_path = f"checkpoints/checkpoint_epoch_{epoch + 1}.pt"
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_epoch_loss,
                "config": cfg,
            },
            checkpoint_path,
        )
        print(f"💾 Checkpoint saved: {checkpoint_path}")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    train_with_in_memory_data()
