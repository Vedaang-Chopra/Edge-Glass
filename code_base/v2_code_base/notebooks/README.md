# Multimodal Alignment Training

A clean, modular implementation for vision-text alignment and LLM integration.

## Overview

This package implements a two-phase training approach:

### Phase 1: Vision-Text Alignment
- Freeze vision encoder (CLIP)
- Freeze text encoder (sentence-transformers)
- Train MLP adapters to align embeddings
- Use MRL + CLIP contrastive loss

### Phase 2: LLM Integration
- Freeze Phase 1 alignment model
- Add Vision-to-LLM projector
- Connect to LLM decoder (Qwen)
- Train for caption generation

## Architecture

```
Phase 1: Alignment
==================
Vision Encoder (CLIP, frozen) → MLP Adapter (trainable) → z_vision
Text Encoder (frozen)         → MLP Adapter (trainable) → z_text
                                      ↓
                              Contrastive Loss (MRL + CLIP)

Phase 2: LLM Integration
========================
z_vision → Vision-to-LLM Projector (trainable) → prefix tokens
                                                      ↓
                                               LLM Decoder → Generated text
```

## Files

- `core.py` - Core components (encoders, adapters, losses)
- `data.py` - Dataset and DataLoader utilities
- `train.py` - Training and evaluation functions
- `llm_integration.py` - Phase 2 LLM connection
- `01_multimodal_alignment.ipynb` - Main notebook with complete workflow

## Quick Start

```python
from core import AlignmentConfig, VisionTextAligner, get_device, set_seed

# Setup
cfg = AlignmentConfig(
    vision_model_name="openai/clip-vit-base-patch32",
    text_model_name="sentence-transformers/all-MiniLM-L6-v2",
    d_align=512,
    batch_size=32,
    num_epochs=3,
)
cfg.device = get_device()
set_seed(cfg.seed)

# Create model
model = VisionTextAligner(cfg)

# Train
from train import train_alignment
from data import create_dataloader, ImageTextDataset

train_loader = create_dataloader(your_dataset, batch_size=32)
history = train_alignment(model, train_loader, num_epochs=3)

# Connect to LLM (Phase 2)
from llm_integration import LLMConfig, MultimodalLLM

llm_cfg = LLMConfig(model_name="Qwen/Qwen2.5-1.5B-Instruct")
mm_model = MultimodalLLM(model, llm_cfg)

# Generate
output = mm_model.generate([image], prompt="Describe this image:")
```

## Key Design Decisions

1. **Simple MLP adapters first** - No Perceiver initially, easier to debug
2. **Frozen encoders** - Only train lightweight adapters
3. **Matryoshka loss** - Enables embedding size flexibility
4. **Modular design** - Each component is independent and testable
5. **Clean interfaces** - Clear input/output contracts

## Adding Audio (Future)

The architecture supports audio by adding:
1. Audio encoder (Whisper) in `core.py`
2. Audio adapter in `VisionTextAligner`
3. Audio dataset in `data.py`

## Requirements

```
torch>=2.0
transformers>=4.30
datasets
Pillow
numpy
tqdm
```

## License

MIT
