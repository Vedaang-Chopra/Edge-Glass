# Pixmo Dataset Integration & Training Improvements

This guide documents the comprehensive updates made to integrate the Pixmo dataset and implement all requested training improvements.

## Table of Contents

1. [Dataset Integration](#dataset-integration)
2. [Model Architecture Updates](#model-architecture-updates)
3. [Training Improvements](#training-improvements)
4. [Visualization & Logging](#visualization--logging)
5. [Usage Guide](#usage-guide)
6. [File Structure](#file-structure)

---

## Dataset Integration

### 1. PixmoParquetImageTextDataset

**Location:** `src/data/dataset_builder.py`

Created a new dataset class that loads data from parquet files with embedded image bytes:

```python
dataset = PixmoParquetImageTextDataset(
    parquet_path="path/to/pixmo_train.parquet",
    image_transforms=transforms,
    max_text_length=512,
    text_dropout_prob=0.1,  # NEW: Text dropout feature
)
```

**Features:**
- Reads image bytes directly from parquet (no external image files needed)
- Decodes images on-the-fly using PIL
- Applies transforms automatically
- **Text dropout:** Randomly drops text during training to force the model to rely on visual features

### 2. Dataset Builder Function

**Location:** `src/data/dataset_builder.py`

```python
datasets = build_image_datasets_from_parquet(
    cfg=config,
    train_parquet_path="/path/to/pixmo_train.parquet",
    val_parquet_path="/path/to/pixmo_val.parquet",
    test_parquet_path="/path/to/pixmo_test.parquet",  # Optional
    train_transforms=train_transforms,
    val_transforms=val_transforms,
    max_text_length=512,
    text_dropout_prob=0.1,
)
```

Returns a dictionary with `'train'`, `'val'`, and optionally `'test'` datasets.

---

## Model Architecture Updates

### 1. Embedding Dimension: 4096

**Updated files:**
- `src/encoders/vision.py`
- `src/encoders/text.py`

**Changes:**
- Projection dimension increased from 1024/1536 to **4096** (top MRL dimension)
- Both vision and text encoders now project to 4096-dim space
- MRL dimensions: `[4096, 2048, 1024, 512, 256, 128]`

### 2. Learnable Attention Pooling

**New file:** `src/encoders/pooling.py`

Replaced simple CLS token or mean pooling with learnable attention mechanisms:

```python
# Two implementations available:
# 1. SimpleAttentionPooling - lightweight, single-layer attention
# 2. AttentionPooling - multi-head attention with multiple queries
```

**Vision Encoder Update:**
```python
vision_encoder = VisionEncoder(
    projection_dim=4096,
    use_attention_pooling=True,
    pooling_type="simple",  # or "multihead"
)
```

**Benefits:**
- Learns to attend to most relevant tokens instead of fixed pooling
- More expressive than CLS token
- Differentiable and trainable

### 3. Updated MRL Dimensions

**MRL hierarchy:**
```
4096 (full, top dimension - use for Qwen decoding)
 ├─ 2048 (high quality retrieval)
 ├─ 1024 (good quality, faster)
 ├─ 512  (medium quality)
 ├─ 256  (lower quality, fast)
 └─ 128  (minimal quality, fastest)
```

**MRL Sampling Strategy:**
- Sample **one dimension per batch** during training (for efficiency)
- Use all 4096 dimensions during inference/decoding with Qwen
- Use smaller dims for retrieval tasks and ablations

### 4. Normalization Before MRL

**Updated files:**
- `src/encoders/vision.py`
- `src/encoders/text.py`

**Critical fix:**
```python
# Normalize embeddings BEFORE MRL projection
pooled = F.normalize(pooled, p=2, dim=-1)

# Then apply MRL (which expects normalized input)
mrl_embeddings = self.mrl(pooled)
```

This ensures all embeddings (vision, text, audio) are properly normalized before MRL and contrastive losses.

---

## Training Improvements

### 1. Updated Loss Weights

**File:** `src/models/losses.py`

```python
AlignmentLoss(
    contrastive_weight=0.25,  # CLIP weight = 0.25
    mrl_weight=1.0,           # MRL weight = 1.0
    sample_single_mrl_dim=True,  # Sample one dim per batch
)
```

**Benefits:**
- Higher weight on MRL encourages better multi-scale representations
- Lower CLIP weight prevents over-fitting to top-level alignment
- Sampling strategy reduces memory and computation

### 2. Optimizer Configuration

**Config file:** `configs/pixmo_alignment.yaml`

```yaml
optimization:
  lr: 0.0002              # 2e-4 for aligner
  weight_decay: 0.01
  betas: [0.9, 0.95]
  max_grad_norm: 1.0
  warmup_ratio: 0.1       # 10% warmup
```

### 3. Learning Rate Schedule

**File:** `src/training/improved_trainer.py`

**Warmup + Cosine Decay:**
```python
def _get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    # Linear warmup for first 10% of steps
    # Then cosine decay to 0
```

**Benefits:**
- Smooth learning rate warmup prevents instability
- Cosine decay ensures convergence
- Automatic calculation based on `warmup_ratio`

### 4. Checkpointing & Crash Recovery

**File:** `src/training/improved_trainer.py`

**Features:**
- **Automatic checkpoint saving:**
  - Latest checkpoint every epoch
  - Best model when validation loss improves
  - Periodic checkpoints at specified intervals

- **Crash recovery:**
  - Automatically loads latest checkpoint on restart
  - Resumes from exact step/epoch
  - Preserves optimizer and scheduler state
  - Restores training history

**Checkpoint structure:**
```python
checkpoint = {
    'epoch': epoch,
    'global_step': step,
    'best_val_loss': best_loss,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'scaler_state_dict': scaler.state_dict(),
    'history': training_history,
    'config': config,
}
```

### 5. Comprehensive Logging

**Logged metrics:**

**Training:**
- `train/loss` - Total loss
- `train/loss_clip` - CLIP/contrastive loss
- `train/loss_mrl` - MRL loss (sampled dimension)
- `train/lr` - Learning rate
- `train/step` - Global step
- `train/epoch` - Current epoch

**Validation:**
- `val/loss` - Total loss
- `val/loss_clip` - CLIP loss
- `val/loss_mrl` - MRL loss
- `val/i2t_r1` - Image-to-text Recall@1
- `val/i2t_r5` - Image-to-text Recall@5
- `val/i2t_r10` - Image-to-text Recall@10

**WandB Integration:**
```python
trainer = ImprovedMultimodalTrainer(
    cfg=config,
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    use_wandb=True,  # Enable WandB logging
)
```

---

## Visualization & Logging

### 1. TrainingVisualizer

**File:** `src/utils/visualization.py`

**Capabilities:**

1. **Training Curves:**
   ```python
   visualizer.plot_training_curves(history)
   # Plots train/val losses over epochs
   ```

2. **Loss Components:**
   ```python
   visualizer.plot_loss_components(history)
   # Separate plots for CLIP, MRL, and total losses
   ```

3. **Learning Rate Schedule:**
   ```python
   visualizer.plot_lr_schedule(lr_history)
   # Shows warmup and cosine decay
   ```

4. **Embedding Space Visualization:**
   ```python
   visualizer.plot_embedding_space(
       vision_embs, text_embs,
       method="pca",  # or "tsne"
   )
   # 2D projection showing vision-text alignment
   ```

5. **Similarity Matrix:**
   ```python
   visualizer.plot_similarity_matrix(vision_embs, text_embs)
   # Heatmap of cosine similarities
   ```

6. **MRL Performance:**
   ```python
   visualizer.plot_mrl_performance(mrl_dims, mrl_scores)
   # Shows retrieval performance at different MRL dimensions
   ```

### 2. Automatic Saving

All visualizations are automatically saved to the output directory:
- `training_curves.png`
- `loss_components.png`
- `lr_schedule.png`
- `embedding_space.png`
- `similarity_matrix.png`
- `metrics.csv`

---

## Usage Guide

### 1. Quick Start

```python
# 1. Load config
from config import load_config
config = load_config("configs/pixmo_alignment.yaml")

# 2. Create datasets
from data.dataset_builder import build_image_datasets_from_parquet
from data.transforms import get_image_transforms

train_transforms = get_image_transforms(image_size=224, is_training=True)
val_transforms = get_image_transforms(image_size=224, is_training=False)

datasets = build_image_datasets_from_parquet(
    cfg=config,
    train_parquet_path=config.dataset.train_parquet,
    val_parquet_path=config.dataset.val_parquet,
    train_transforms=train_transforms,
    val_transforms=val_transforms,
    max_text_length=512,
    text_dropout_prob=0.1,
)

# 3. Create dataloaders
from torch.utils.data import DataLoader
train_loader = DataLoader(datasets['train'], batch_size=64, shuffle=True)
val_loader = DataLoader(datasets['val'], batch_size=64, shuffle=False)

# 4. Create model
from models import MultimodalAlignmentModel
model = MultimodalAlignmentModel(config)

# 5. Train
from training.improved_trainer import ImprovedMultimodalTrainer
trainer = ImprovedMultimodalTrainer(
    cfg=config,
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    use_wandb=True,
)
history = trainer.train()

# 6. Visualize
from utils.visualization import TrainingVisualizer
visualizer = TrainingVisualizer(save_dir="./outputs")
visualizer.plot_training_curves(history)
visualizer.plot_embedding_space(vision_embs, text_embs)
```

### 2. Using the Notebook

```bash
cd edge_glass_modular/notebooks
jupyter notebook 02_pixmo_vision_text_alignment.ipynb
```

The notebook provides:
- Interactive visualizations
- Step-by-step training
- Embedding analysis
- Retrieval testing

### 3. Configuration

**Key config file:** `configs/pixmo_alignment.yaml`

**Important settings:**
```yaml
dataset:
  train_parquet: /path/to/pixmo_train.parquet
  val_parquet: /path/to/pixmo_val.parquet
  batch_size: 64
  text_dropout_prob: 0.1

vision_encoder:
  projection_dim: 4096
  mrl_dimensions: [2048, 1024, 512, 256, 128]
  use_attention_pooling: true
  pooling_type: simple

losses:
  contrastive: 0.25  # CLIP weight
  mrl: 1.0           # MRL weight

optimization:
  lr: 0.0002
  weight_decay: 0.01
  max_grad_norm: 1.0
```

---

## File Structure

```
edge_glass_modular/
├── configs/
│   └── pixmo_alignment.yaml          # NEW: Pixmo training config
├── src/
│   ├── data/
│   │   ├── dataset_builder.py         # UPDATED: Added Pixmo dataset classes
│   │   └── transforms.py
│   ├── encoders/
│   │   ├── vision.py                  # UPDATED: 4096 dim + attention pooling
│   │   ├── text.py                    # UPDATED: 4096 dim
│   │   ├── mrl.py                     # UPDATED: Sampling strategy
│   │   └── pooling.py                 # NEW: Learnable attention pooling
│   ├── models/
│   │   ├── alignment.py
│   │   ├── losses.py                  # UPDATED: New weights + sampling
│   │   └── projector.py
│   ├── training/
│   │   ├── trainer.py
│   │   ├── improved_trainer.py        # NEW: Full-featured trainer
│   │   └── callbacks.py
│   └── utils/
│       └── visualization.py           # NEW: Comprehensive visualization
├── notebooks/
│   └── 02_pixmo_vision_text_alignment.ipynb  # NEW: Complete training notebook
└── PIXMO_INTEGRATION_GUIDE.md        # This file
```

---

## Key Changes Summary

### ✅ Dataset
- [x] PixmoParquetImageTextDataset for loading parquet + image bytes
- [x] build_image_datasets_from_parquet function
- [x] Text dropout during training (configurable)

### ✅ Model Architecture
- [x] 4096-dim embeddings (top MRL dimension)
- [x] MRL dimensions: [2048, 1024, 512, 256, 128]
- [x] Learnable attention pooling (replaces CLS/mean)
- [x] Proper normalization before MRL

### ✅ Training
- [x] Loss weights: mrl_weight=1.0, clip_weight=0.25
- [x] LR: 2e-4, weight_decay: 0.01, max_grad_norm: 1.0
- [x] Warmup + cosine decay LR schedule
- [x] Sample one MRL dimension per batch
- [x] Checkpointing with best model tracking
- [x] Automatic crash recovery

### ✅ Logging & Visualization
- [x] Comprehensive loss logging (CLIP, MRL, total)
- [x] Validation metrics (loss + retrieval)
- [x] WandB integration
- [x] Training curves visualization
- [x] Embedding space visualization
- [x] Similarity matrix visualization
- [x] LR schedule visualization

### ✅ Documentation
- [x] Complete notebook with all features
- [x] Configuration file
- [x] This integration guide

---

## Next Steps

1. **Run training:**
   ```bash
   cd edge_glass_modular/notebooks
   jupyter notebook 02_pixmo_vision_text_alignment.ipynb
   ```

2. **Monitor training:**
   - Check WandB dashboard for real-time metrics
   - Review saved visualizations in output directory
   - Monitor checkpoint directory for best model

3. **Evaluate:**
   - Test retrieval performance at different MRL dimensions
   - Analyze embedding space alignment
   - Compare with baseline models

4. **Fine-tune:**
   - Adjust text dropout probability
   - Experiment with pooling types
   - Try different MRL dimension sets

5. **Deploy:**
   - Use full 4096-dim for Qwen decoding
   - Use smaller dims for fast retrieval
   - Export best model for production

---

## Troubleshooting

### Issue: Out of memory
**Solution:** Reduce batch size or enable gradient checkpointing

### Issue: Training not resuming from checkpoint
**Solution:** Check that `checkpoint_latest.pt` exists in ckpt_dir

### Issue: Text dropout too aggressive
**Solution:** Reduce `text_dropout_prob` in config (try 0.05 instead of 0.1)

### Issue: Poor convergence
**Solution:**
- Increase warmup_ratio (try 0.15 instead of 0.1)
- Reduce learning rate (try 1e-4 instead of 2e-4)
- Check that embeddings are normalized before MRL

---

## Contact & Support

For issues or questions about this integration:
- Check the training logs in `./outputs/pixmo_alignment/`
- Review WandB runs for metric trends
- Examine checkpoint files for state information

Happy training! 🚀
