# In-Memory Datasets Guide

This guide explains how to use the in-memory dataset implementations for faster training.

## Overview

The in-memory datasets (`InMemoryImageTextDataset` and `InMemoryAudioTextDataset`) pre-load all data into RAM at initialization. This eliminates repeated network requests and file I/O during training, resulting in:

- **Faster training**: No download/loading overhead per epoch
- **Consistent iteration times**: Predictable batch timing
- **Better GPU utilization**: No I/O bottlenecks

## Files

- `in_memory_datasets.py`: Contains the dataset implementations
- `train_with_in_memory_datasets.py`: Complete training example
- `data.py`: Original dataset implementations (still available)

## Quick Start

### 1. Image-Text Dataset (PixMo-Cap)

```python
from datasets import load_dataset
from torch.utils.data import DataLoader
from in_memory_datasets import InMemoryImageTextDataset, collate_in_memory_images

# Load HuggingFace dataset
hf_dataset = load_dataset("allenai/pixmo-cap", split="train")

# Create in-memory dataset (this pre-loads all images)
dataset = InMemoryImageTextDataset(
    hf_dataset=hf_dataset,
    img_col="image_url",
    txt_col="caption",
    max_samples=10000,  # Limit to 10K samples (adjust based on RAM)
    image_size=(224, 224),
)

# Create DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    collate_fn=collate_in_memory_images,
    pin_memory=True,
    drop_last=True,
)

# Use in training
for batch in dataloader:
    images = batch["images"]  # List of PIL Images
    captions = batch["captions"]  # List of strings
    # ... training code ...
```

### 2. Audio-Text Dataset (MusicCaps)

```python
from datasets import load_dataset
from torch.utils.data import DataLoader
from in_memory_datasets import InMemoryAudioTextDataset, collate_in_memory_audio

# Load HuggingFace dataset
hf_dataset = load_dataset("google/MusicCaps", split="train")

# Create in-memory dataset (this pre-loads all audio)
dataset = InMemoryAudioTextDataset(
    hf_dataset=hf_dataset,
    audio_col="audio",
    txt_col="caption",
    max_samples=5000,  # Limit to 5K samples
    target_sr=16000,  # Resample to 16kHz
    max_duration=30.0,  # Max 30 seconds per clip
)

# Create DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    collate_fn=collate_in_memory_audio,
    pin_memory=True,
    drop_last=True,
)

# Use in training
for batch in dataloader:
    audio_waveforms = batch["audio"]  # List of np.ndarray
    captions = batch["captions"]  # List of strings
    # ... training code ...
```

## Complete Training Example

See `train_with_in_memory_datasets.py` for a complete training script that shows:

1. Loading data into memory
2. Creating model and optimizer
3. Training loop with in-memory data
4. Checkpoint saving

Run it with:

```bash
python train_with_in_memory_datasets.py
```

## Memory Considerations

### Estimating Memory Usage

**Images:**
- Each 224x224 RGB image: ~150 KB (uncompressed PIL Image)
- 10,000 images ≈ 1.5 GB
- 50,000 images ≈ 7.5 GB

**Audio:**
- 30 seconds at 16kHz: ~1.9 MB (float32 array)
- 5,000 audio clips ≈ 9.5 GB

### Tips for Managing Memory

1. **Use `max_samples` parameter** to limit dataset size:
   ```python
   dataset = InMemoryImageTextDataset(
       hf_dataset=hf_dataset,
       max_samples=10000,  # Only load 10K samples
   )
   ```

2. **Reduce image size** for images:
   ```python
   dataset = InMemoryImageTextDataset(
       hf_dataset=hf_dataset,
       image_size=(128, 128),  # Smaller images
   )
   ```

3. **Reduce audio duration** for audio:
   ```python
   dataset = InMemoryAudioTextDataset(
       hf_dataset=hf_dataset,
       max_duration=10.0,  # Only 10 seconds
   )
   ```

4. **Check available RAM** before loading:
   ```python
   import psutil
   available_gb = psutil.virtual_memory().available / (1024**3)
   print(f"Available RAM: {available_gb:.1f} GB")
   ```

## Comparison: In-Memory vs On-the-Fly

| Aspect | In-Memory | On-the-Fly |
|--------|-----------|------------|
| Initial loading time | Slow (minutes) | Fast (seconds) |
| Training speed per epoch | Fast | Slow |
| Memory usage | High | Low |
| Best for | Multi-epoch training | Single-pass training |
| Best for | Small-medium datasets | Large datasets |

## When to Use In-Memory Datasets

✅ **Use in-memory datasets when:**
- Training for multiple epochs (3+)
- Dataset fits in RAM (with `max_samples`)
- Iterating over the same data repeatedly
- Network I/O is a bottleneck

❌ **Use on-the-fly datasets when:**
- Single-pass training
- Dataset too large for RAM
- RAM is limited
- Exploring data quickly

## Supported Datasets

### Image Datasets

| Dataset | HuggingFace ID | Size | Notes |
|---------|---------------|------|-------|
| PixMo-Cap | `allenai/pixmo-cap` | ~18M | High-quality captions |
| COCO | `HuggingFaceM4/COCO` | ~118K | Standard benchmark |
| Flickr30k | `nlphuji/flickr30k` | ~30K | Multiple captions/image |

### Audio Datasets

| Dataset | HuggingFace ID | Size | Notes |
|---------|---------------|------|-------|
| MusicCaps | `google/MusicCaps` | ~5.5K | Music with expert captions |
| LAION-Audio | `laion/audio-dataset` | ~630K | Large-scale audio-text |
| Clotho | `ChristophSchuhmann/Clotho` | ~5K | General audio with captions |

## Troubleshooting

### Out of Memory (OOM) during loading

**Solution:** Reduce `max_samples`:
```python
dataset = InMemoryImageTextDataset(
    hf_dataset=hf_dataset,
    max_samples=5000,  # Start with smaller size
)
```

### Slow loading times

**Solution:** This is expected! Loading takes time upfront but pays off during training. You can:
- Use a smaller subset initially for testing
- Load once and save to disk (see "Caching" below)

### Failed to load some images/audio

**Solution:** The dataset automatically uses fallback data (blank image or silence) and reports which indices failed. This is normal for datasets with URLs.

## Advanced: Caching Loaded Data

To avoid re-loading data every time you run training:

```python
import pickle
from pathlib import Path

# Load and cache
cache_file = Path("cached_dataset.pkl")

if cache_file.exists():
    print("Loading from cache...")
    with open(cache_file, "rb") as f:
        dataset = pickle.load(f)
else:
    print("Creating dataset...")
    hf_dataset = load_dataset("allenai/pixmo-cap", split="train")
    dataset = InMemoryImageTextDataset(hf_dataset, max_samples=10000)

    print("Saving to cache...")
    with open(cache_file, "wb") as f:
        pickle.dump(dataset, f)

# Use dataset...
```

## Questions?

For issues or questions, please check:
1. Memory usage with `max_samples`
2. Column names match your dataset
3. RAM availability on your system

## Next Steps

1. Start with a small `max_samples` (e.g., 1000) to test
2. Gradually increase based on available RAM
3. Monitor training speed improvements
4. Adjust batch size and num_workers as needed

Happy training! 🚀
