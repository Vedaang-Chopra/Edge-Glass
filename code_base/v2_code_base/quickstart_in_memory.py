"""
quickstart_in_memory.py - Quick start example for in-memory datasets

This is a minimal example showing how to use the in-memory datasets.
Run this to get started quickly!
"""

from datasets import load_dataset
from torch.utils.data import DataLoader
from in_memory_datasets import (
    InMemoryImageTextDataset,
    InMemoryAudioTextDataset,
    collate_in_memory_images,
    collate_in_memory_audio,
)


def quickstart_images():
    """Quick example for image-text dataset."""
    print("\n" + "=" * 60)
    print("QUICKSTART: Image-Text Dataset")
    print("=" * 60)

    # Step 1: Load HuggingFace dataset
    print("\n[Step 1] Loading PixMo-Cap from HuggingFace...")
    hf_dataset = load_dataset("allenai/pixmo-cap", split="train")
    print(f"   Total samples: {len(hf_dataset)}")

    # Step 2: Create in-memory dataset (pre-loads images)
    print("\n[Step 2] Pre-loading images into memory...")
    dataset = InMemoryImageTextDataset(
        hf_dataset=hf_dataset,
        img_col="image_url",
        txt_col="caption",
        max_samples=1000,  # Start with 1K images (~150 MB)
        image_size=(224, 224),
    )

    # Step 3: Create DataLoader
    print("\n[Step 3] Creating DataLoader...")
    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_in_memory_images,
        pin_memory=True,
    )
    print(f"   Batches per epoch: {len(dataloader)}")

    # Step 4: Test iteration
    print("\n[Step 4] Testing batch iteration...")
    batch = next(iter(dataloader))
    print(f"   ✅ Batch loaded successfully!")
    print(f"   - Images: {len(batch['images'])} PIL Images")
    print(f"   - Captions: {len(batch['captions'])} strings")
    print(f"   - Example caption: {batch['captions'][0][:80]}...")

    print("\n" + "=" * 60)
    print("✅ Image dataset ready for training!")
    print("=" * 60)

    return dataloader


def quickstart_audio():
    """Quick example for audio-text dataset."""
    print("\n" + "=" * 60)
    print("QUICKSTART: Audio-Text Dataset")
    print("=" * 60)

    # Step 1: Load HuggingFace dataset
    print("\n[Step 1] Loading MusicCaps from HuggingFace...")
    try:
        hf_dataset = load_dataset("google/MusicCaps", split="train")
        print(f"   Total samples: {len(hf_dataset)}")
    except Exception as e:
        print(f"   ⚠️  Could not load MusicCaps: {e}")
        print("   Skipping audio example...")
        return None

    # Step 2: Create in-memory dataset (pre-loads audio)
    print("\n[Step 2] Pre-loading audio into memory...")
    dataset = InMemoryAudioTextDataset(
        hf_dataset=hf_dataset,
        audio_col="audio",
        txt_col="caption",
        max_samples=500,  # Start with 500 audio clips
        target_sr=16000,
        max_duration=30.0,
    )

    # Step 3: Create DataLoader
    print("\n[Step 3] Creating DataLoader...")
    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_in_memory_audio,
        pin_memory=True,
    )
    print(f"   Batches per epoch: {len(dataloader)}")

    # Step 4: Test iteration
    print("\n[Step 4] Testing batch iteration...")
    batch = next(iter(dataloader))
    print(f"   ✅ Batch loaded successfully!")
    print(f"   - Audio: {len(batch['audio'])} waveforms")
    print(f"   - Captions: {len(batch['captions'])} strings")
    print(f"   - Example caption: {batch['captions'][0][:80]}...")

    print("\n" + "=" * 60)
    print("✅ Audio dataset ready for training!")
    print("=" * 60)

    return dataloader


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print(" " * 15 + "IN-MEMORY DATASETS QUICKSTART")
    print("=" * 70)

    # Test image dataset
    image_loader = quickstart_images()

    # Test audio dataset (optional)
    try:
        audio_loader = quickstart_audio()
    except Exception as e:
        print(f"\n⚠️  Audio dataset test skipped: {e}")

    # Summary
    print("\n" + "=" * 70)
    print("QUICKSTART COMPLETE!")
    print("=" * 70)
    print("\n📚 Next steps:")
    print("   1. Increase max_samples based on your RAM")
    print("   2. Integrate with your training loop")
    print("   3. See train_with_in_memory_datasets.py for complete example")
    print("   4. Read IN_MEMORY_DATASETS_GUIDE.md for details")
    print("\n💡 Tips:")
    print("   - Start small (1K-5K samples) then increase")
    print("   - Monitor RAM usage during loading")
    print("   - Use max_samples to control memory")
    print("=" * 70)
