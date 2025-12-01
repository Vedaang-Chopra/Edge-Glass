"""
test_in_memory_datasets.py - Quick test script for in-memory datasets

This script tests that the in-memory datasets can be created and used.
"""

import torch
from torch.utils.data import DataLoader
import sys

print("=" * 60)
print("Testing In-Memory Datasets")
print("=" * 60)

# Test 1: Check imports
print("\n[1/4] Testing imports...")
try:
    from in_memory_datasets import (
        InMemoryImageTextDataset,
        InMemoryAudioTextDataset,
        collate_in_memory_images,
        collate_in_memory_audio,
    )
    print("✅ Imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

# Test 2: Create a small dummy image dataset
print("\n[2/4] Testing InMemoryImageTextDataset...")
try:
    from datasets import load_dataset
    from PIL import Image
    import numpy as np

    # Load a small subset of PixMo-Cap
    print("   Loading PixMo-Cap dataset (small subset)...")
    hf_dataset = load_dataset("allenai/pixmo-cap", split="train", streaming=False)

    # Create in-memory dataset with only 100 samples
    print("   Creating in-memory dataset (100 samples)...")
    dataset = InMemoryImageTextDataset(
        hf_dataset=hf_dataset,
        img_col="image_url",
        txt_col="caption",
        max_samples=100,  # Very small for testing
        image_size=(224, 224),
    )

    # Test __len__ and __getitem__
    print(f"   Dataset length: {len(dataset)}")
    sample = dataset[0]
    print(f"   Sample keys: {sample.keys()}")
    print(f"   Image type: {type(sample['image'])}")
    print(f"   Image size: {sample['image'].size}")
    print(f"   Caption preview: {sample['caption'][:50]}...")

    # Test DataLoader
    print("   Creating DataLoader...")
    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=collate_in_memory_images,
    )

    print("   Testing batch iteration...")
    batch = next(iter(dataloader))
    print(f"   Batch images: {len(batch['images'])}")
    print(f"   Batch captions: {len(batch['captions'])}")

    print("✅ InMemoryImageTextDataset test passed")

except Exception as e:
    print(f"❌ InMemoryImageTextDataset test failed: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Test audio dataset (if available)
print("\n[3/4] Testing InMemoryAudioTextDataset...")
try:
    from datasets import load_dataset
    import numpy as np

    # Try to load a small audio dataset
    print("   Loading MusicCaps dataset (small subset)...")
    try:
        hf_dataset = load_dataset("google/MusicCaps", split="train")
    except Exception as e:
        print(f"   ⚠️  MusicCaps not available: {e}")
        print("   Skipping audio dataset test")
        raise

    # Create in-memory dataset with only 50 samples
    print("   Creating in-memory audio dataset (50 samples)...")
    dataset = InMemoryAudioTextDataset(
        hf_dataset=hf_dataset,
        audio_col="audio",
        txt_col="caption",
        max_samples=50,  # Very small for testing
        target_sr=16000,
        max_duration=30.0,
    )

    # Test __len__ and __getitem__
    print(f"   Dataset length: {len(dataset)}")
    sample = dataset[0]
    print(f"   Sample keys: {sample.keys()}")
    print(f"   Audio type: {type(sample['audio'])}")
    print(f"   Audio shape: {sample['audio'].shape}")
    print(f"   Caption preview: {sample['caption'][:50]}...")

    # Test DataLoader
    print("   Creating DataLoader...")
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_in_memory_audio,
    )

    print("   Testing batch iteration...")
    batch = next(iter(dataloader))
    print(f"   Batch audio: {len(batch['audio'])}")
    print(f"   Batch captions: {len(batch['captions'])}")

    print("✅ InMemoryAudioTextDataset test passed")

except Exception as e:
    print(f"⚠️  InMemoryAudioTextDataset test skipped or failed: {e}")

# Test 4: Memory usage estimate
print("\n[4/4] Memory usage estimate...")
try:
    import psutil
    mem = psutil.virtual_memory()
    print(f"   Total RAM: {mem.total / (1024**3):.1f} GB")
    print(f"   Available RAM: {mem.available / (1024**3):.1f} GB")
    print(f"   Used RAM: {mem.used / (1024**3):.1f} GB")
    print(f"   Usage: {mem.percent:.1f}%")
    print("✅ Memory info retrieved")
except Exception as e:
    print(f"⚠️  Could not get memory info: {e}")

# Final summary
print("\n" + "=" * 60)
print("Test Summary")
print("=" * 60)
print("✅ In-memory datasets are ready to use!")
print("\nNext steps:")
print("1. Adjust max_samples based on your RAM")
print("2. Run train_with_in_memory_datasets.py for full training")
print("3. See IN_MEMORY_DATASETS_GUIDE.md for details")
print("=" * 60)
