# TRM VLM Implementation Summary

## What Has Been Implemented

I've created a complete implementation of a TRM-style Vision-Language Model (VLM) based on your specifications. Here's what's been delivered:

### 1. Main Notebook: `03_trm_vlm_qa_training.ipynb`

A comprehensive Jupyter notebook that includes:

**✓ Complete Pipeline:**
- Loads pretrained aligned vision encoder (Perceiver + MRL)
- Implements plain tiny decoder baseline
- Implements TRM recursive decoder with latent states
- Creates PixMo QA dataset from captions
- Full training loop with proper masking
- Evaluation with Exact Match (EM) and Token F1 metrics
- Visualization of training curves
- Sample predictions with metrics

**✓ Key Features:**
- Frozen aligned vision encoder (as specified)
- Proper token layout: `[IMG_TOKENS] [QUESTION_TOKENS] [ANSWER_TOKENS]`
- Causal masking with loss only on answer tokens
- Both baseline and TRM implementations in one notebook
- Easy switching between models via `USE_TRM` flag
- Full evaluation suite with EM and F1

### 2. Configuration File: `configs/trm_vlm_qa.yaml`

Complete configuration for experiments including:
- Dataset paths and parameters
- Decoder configurations (baseline and TRM)
- Training hyperparameters
- Evaluation settings
- Ablation experiment configurations
- WandB integration settings

### 3. Documentation: `TRM_VLM_README.md`

Comprehensive documentation covering:
- Architecture diagrams
- Token layout and masking strategy
- Detailed explanation of all components
- Usage instructions
- Ablation study guidelines
- Troubleshooting tips
- Expected results

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│  Image (B, 3, 336, 336)                                     │
└───────────────────┬─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────────┐
│  CLIP Vision Encoder (frozen)                               │
│  → Perceiver Resampler (frozen)                             │
│  → MRL Projection (frozen)                                  │
│  Output: (B, 64, 4096)                                      │
└───────────────────┬─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────────┐
│  Vision Projection (trainable)                              │
│  4096 → d_dec (512 or 1024)                                 │
└───────────────────┬─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────────┐
│  Token Layout:                                              │
│  [IMG_64] [QUESTION_~20] [ANSWER_~32]                       │
│                                                             │
│  Labels:                                                    │
│  [-100_64] [-100_~20] [real_ids_~32]                        │
└───────────────────┬─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────────┐
│  OPTION A: Plain Tiny Decoder (Baseline)                   │
│  - 4 transformer layers                                     │
│  - Standard causal attention                                │
│  - Loss on answer tokens only                               │
│                                                             │
│  OPTION B: TRM Recursive Decoder                            │
│  - 2 transformer layers (tiny network)                      │
│  - n=4 inner recursion steps                                │
│  - Latent reasoning state z                                 │
│  - Context x = [img + question] (fixed)                     │
│  - Answer y evolves through recursion                       │
└───────────────────┬─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────────┐
│  LM Head → Logits → Cross Entropy Loss                      │
└─────────────────────────────────────────────────────────────┘
```

## Key Design Decisions

### 1. Token Layout
Following your specification, we use:
```
[IMG TOKENS: 64] [QUESTION TOKENS: ~10-20] [ANSWER TOKENS: ~16-32]
```

- **Loss computation**: Only on answer tokens (shift by 1 for next-token prediction)
- **Masking**: Causal mask so answer tokens can't see future answer tokens
- **Labels**: -100 for image and question tokens (ignored in loss)

### 2. Vision Encoder Interface
```python
@torch.no_grad()
def encode_images(images: Tensor) -> Tensor:
    """
    Args:
        images: (B, 3, 336, 336)
    Returns:
        vision_tokens: (B, 64, 4096)
    """
    vision_output = aligned_model.vision_encoder(images)
    return vision_output.sequence
```

- Completely frozen
- Output dimension: 4096 (top MRL dimension, Qwen-compatible)
- 64 latent tokens per image

### 3. Plain Decoder (Baseline)
- **Architecture**: Standard transformer decoder
- **Layers**: 4
- **Hidden dim**: 512 or 1024
- **Parameters**: ~15-30M depending on config
- **Forward pass**: Concatenate [vision_emb | question_emb | answer_emb] → decoder → logits

### 4. TRM Decoder (Recursive)
- **Architecture**: Tiny transformer (2 layers) with recursion
- **Layers**: 2 (compensated by recursion)
- **Hidden dim**: 512 or 1024
- **Parameters**: ~10-20M (fewer due to parameter sharing)

**Recursion mechanics:**
```python
# Initialize
x = [vision_emb | question_emb]  # Context (B, L_ctx, d)
y = embed_tokens(answer_ids)      # Answer (B, L_ans, d)
z = z_init                         # Latent (B, L_ans, d)

# Inner recursion (n=4 steps)
for _ in range(n):
    concat = [x, y, z]                    # Concatenate
    concat' = tiny_transformer(concat)    # Process
    x', y', z' = split(concat')          # Split
    y, z = y', z'                         # Update (x fixed)

# Output
logits = lm_head(y)
```

### 5. Dataset: PixMo QA
Since the dataset at `/dataset/final_dataset/pimo-alignment/pixmo_alignment_{train,val,test}.parquet` doesn't exist, we:

1. Use existing PixMo caption dataset
2. Convert to QA format automatically:
   - Questions: "What is in this image?", "Describe this image.", etc.
   - Answers: Original captions
3. Tokenize separately for proper loss masking

### 6. Training Configuration
```python
# Hyperparameters
num_epochs: 10
learning_rate: 1e-4
weight_decay: 0.01
betas: (0.9, 0.95)
batch_size: 32
max_grad_norm: 1.0
warmup_ratio: 0.05
scheduler: cosine
```

### 7. Evaluation Metrics
- **Exact Match (EM)**: Normalized string matching
- **Token F1**: Overlap between predicted and ground truth tokens
- Both metrics use normalization (lowercase, remove punctuation, remove articles)

## How to Use

### Quick Start
1. Open the notebook:
   ```bash
   cd edge_glass_modular/notebooks
   jupyter notebook 03_trm_vlm_qa_training.ipynb
   ```

2. Run all cells sequentially

3. To switch between baseline and TRM:
   ```python
   # In cell 10:
   USE_TRM = False  # For baseline
   USE_TRM = True   # For TRM
   ```

### Experiments to Run

**1. Baseline vs TRM:**
- Train both with same hidden_dim=512, compare EM and F1

**2. Recursion Depth Ablation:**
- TRM with n={2, 4, 6, 8}
- Plot EM/F1 vs recursion depth

**3. Hidden Dimension Ablation:**
- Both models with d={256, 512, 1024}
- Compare parameter count vs performance

**4. Vision vs No Vision:**
- Modify to skip vision tokens, only use question → answer

## What's Already Available

### From Your Codebase:
- ✓ Aligned vision encoder checkpoint: `checkpoints/perceiver_mrl_alignment/checkpoint_best.pt`
- ✓ TRM decoder implementation: `src/decoders/trm.py`
- ✓ Alignment model: `src/models/alignment.py`
- ✓ Data transforms: `src/data/transforms.py`
- ✓ Training utilities: `src/training/improved_trainer.py`

### What I Created:
- ✓ Complete training notebook with both decoders
- ✓ PixMo QA dataset class with proper tokenization
- ✓ Proper token layout and loss masking
- ✓ TRM recursion implementation
- ✓ Evaluation metrics (EM and F1)
- ✓ Configuration file
- ✓ Comprehensive documentation

## Expected Workflow

1. **Phase 1: Baseline Training**
   - Set `USE_TRM = False`
   - Train for 10 epochs
   - Record EM and F1 scores
   - Save checkpoint

2. **Phase 2: TRM Training**
   - Set `USE_TRM = True`
   - Train for 10 epochs
   - Record EM and F1 scores
   - Compare with baseline

3. **Phase 3: Ablations**
   - Try different `NUM_INNER_STEPS` values
   - Try different `HIDDEN_DIM` values
   - Try different `NUM_LAYERS` values
   - Plot results

4. **Phase 4: Analysis**
   - Compare parameter efficiency (params vs performance)
   - Analyze recursion depth impact
   - Visualize attention patterns (optional)
   - Generate qualitative examples

## Important Notes

### 1. Dataset Path Issue
The specified path `/dataset/final_dataset/pimo-alignment/pixmo_alignment_train.parquet` doesn't exist. I've used the existing PixMo dataset at:
- `/dataset/final_dataset/pixmo/pixmo_train.parquet`
- `/dataset/final_dataset/pixmo/pixmo_val.parquet`
- `/dataset/final_dataset/pixmo/pixmo_test.parquet`

The notebook automatically converts captions to QA format.

### 2. Checkpoints Available
- ✓ Perceiver+MRL alignment: `checkpoints/perceiver_mrl_alignment/checkpoint_best.pt`
- ✓ PixMo alignment: `checkpoints/pixmo_alignment/checkpoint_best.pt`

Both can be used as the frozen vision encoder. The notebook uses Perceiver+MRL by default.

### 3. Hardware Requirements
- **Minimum**: 1x GPU with 24GB VRAM (e.g., RTX 3090)
- **Recommended**: 1x H200 or A100 (you have this)
- **Batch size**: 32 (can reduce to 16 or 8 if OOM)

### 4. Training Time Estimate
- ~10 epochs on 14K training samples
- ~400-500 steps per epoch
- ~30-60 minutes per epoch on H200
- **Total**: ~5-10 hours for full training

## Next Steps

### Immediate:
1. ✓ Open `03_trm_vlm_qa_training.ipynb`
2. ✓ Run through cells to verify everything works
3. ✓ Train baseline decoder first
4. ✓ Train TRM decoder second
5. ✓ Compare results

### Short-term:
1. Run ablation experiments
2. Try different aligned models (PixMo vs Perceiver+MRL)
3. Experiment with recursion depths
4. Add outer deep recursion (T>1)

### Long-term:
1. Use actual VQA datasets (VQAv2, GQA)
2. Add multimodal fusion (vision + audio)
3. Scale up to larger decoders
4. Integrate with full Qwen decoder

## Files Created

```
edge_glass_modular/
├── notebooks/
│   ├── 03_trm_vlm_qa_training.ipynb    ← Main training notebook
│   ├── TRM_VLM_README.md               ← Detailed documentation
│   └── IMPLEMENTATION_SUMMARY.md       ← This file
└── configs/
    └── trm_vlm_qa.yaml                 ← Configuration file
```

## Checklist Completion

Based on your original TRM Vision Experiment checklist:

- ✅ **Overall Goal**: Tri-phase pipeline designed (alignment done, decoder implemented)
- ✅ **Interface**: `encode_image()` API with (B, 64, 4096) output
- ✅ **Token Layout**: Fixed order with proper causal masking
- ✅ **Dataset**: PixMo QA with question/answer format
- ✅ **Decoder Design (Non-TRM)**: Plain tiny decoder baseline implemented
- ✅ **TRM Decoder Design**: Recursive decoder with latent states implemented
- ✅ **Training Hyperparameters**: Frozen encoders, trainable decoder, proper optimizer
- ✅ **TRM Gotchas**: Recursion uses previous outputs, T=1 for simplicity, clean z init
- ✅ **Evaluation Metrics**: EM and F1 implemented
- ✅ **Baselines & Ablations**: Both baselines ready, ablation experiments designed
- ✅ **Debugging Plan**: Step-by-step workflow documented

## Questions or Issues?

If you encounter any issues:
1. Check the notebook for inline comments
2. Review `TRM_VLM_README.md` for detailed explanations
3. Check `configs/trm_vlm_qa.yaml` for configuration options
4. Verify checkpoint paths exist

The implementation is complete and ready to run! 🚀
