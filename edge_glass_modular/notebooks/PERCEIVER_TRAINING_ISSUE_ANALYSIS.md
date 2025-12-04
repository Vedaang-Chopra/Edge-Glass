# Perceiver MRL Alignment Training Issue - Root Cause Analysis

## Problem Summary

The **Perceiver-based alignment model is NOT learning** - loss is stuck at ~6.0 and not decreasing over 10 epochs.

**Key Evidence:**
- Perceiver Model: Loss stuck at **6.06 (train)** and **6.00 (val)**
- Direct Model: Loss decreases from **~1.24 → 1.12** and continues improving
- Retrieval accuracy: **R@1: 0.03%, R@5: 0.17%** (essentially random)

---

## Root Cause: Perceiver Latent Initialization Bug

### Issue Location: `encoders/perceiver.py:155-156`

```python
# Learned latent queries
self.latents = nn.Parameter(torch.randn(num_latents, dim))
nn.init.normal_(self.latents, std=0.02)  # ❌ INCORRECT - reinitializes after randn
```

**Problem:** The latents are initialized with `torch.randn()` (std=1.0) and then **immediately reinitialized** with `nn.init.normal_(std=0.02)`. This creates extremely small initial values that prevent learning.

### Why This Breaks Training

1. **Too Small Initial Values**: `std=0.02` means initial latent values are ~±0.04
2. **Vanishing Gradients**: Small latents → small attention scores → flat softmax → weak gradients
3. **No Learning Signal**: The perceiver never gets strong enough signals to learn meaningful compressions
4. **Loss Plateaus**: Without gradient flow through perceiver, only the final projector trains minimally

---

## Configuration Comparison

### Perceiver Model (NOT working)
```yaml
# perceiver_mrl_alignment.yaml
vision_encoder:
  use_perceiver: true
  perceiver_num_latents: 64
  perceiver_latent_dim: 1024
  perceiver_num_layers: 4
  use_attention_pooling: false  # Disabled because perceiver handles pooling

losses:
  contrastive: 0.25  # CLIP weight
  mrl: 1.0           # MRL weight

optimization:
  lr: 0.002          # ❌ 10x HIGHER than working model
  batch_size: 64     # ❌ HALF the working model
```

### Direct Model (WORKING)
```yaml
# pixmo_alignment.yaml
vision_encoder:
  use_perceiver: false
  use_attention_pooling: true  # Uses learnable attention pooling
  pooling_type: simple

losses:
  contrastive: 0.25
  mrl: 1.0

optimization:
  lr: 0.0002         # ✓ 2e-4
  batch_size: 128    # ✓ Larger batch size
```

---

## Secondary Issues Found

### 1. **Learning Rate Too High** (`perceiver_mrl_alignment.yaml:70`)

**Current:** `lr: 0.002` (2e-3)
**Working Model:** `lr: 0.0002` (2e-4)
**Issue:** 10x higher LR can cause instability with the perceiver's sensitive attention mechanism

### 2. **Batch Size Too Small** (`perceiver_mrl_alignment.yaml:14`)

**Current:** `batch_size: 64`
**Working Model:** `batch_size: 128`
**Issue:** Contrastive learning requires larger batches for negative pairs. With only 64 samples, the model has insufficient negatives.

### 3. **Loss Computation Issue** (`models/losses.py:170-173`)

```python
losses["loss_clip"] = contrastive  # For logging
if mrl_losses:
    # Aggregate MRL loss for logging
    losses["loss_mrl"] = sum(mrl_losses.values()) if mrl_losses else torch.tensor(0.0, device=contrastive.device)
```

**Problem:** When `sample_single_mrl_dim=True`, only ONE MRL dimension is sampled per batch, but the logged `loss_mrl` is reported as if it's the full loss. This makes loss values look artificially similar between CLIP and MRL.

### 4. **Mean Pooling After Perceiver** (`encoders/vision.py:168`)

```python
# Pool latents (mean pooling for perceiver)
pooled = latents.mean(dim=1)  # (B, projection_dim)
```

**Issue:** After compressing to 64 latents, taking a simple mean throws away the structured information. The working model uses **learnable attention pooling** which is much more effective.

---

## Training Results Comparison

### Perceiver Model (Broken)
```
Epoch 10/10:
  Train Loss: 6.0602 (CLIP: 4.8481, MRL: 4.8481)
  Val Loss: 6.0045 (CLIP: 4.8036, MRL: 4.8036)
  R@1: 0.03%, R@5: 0.17%, R@10: 0.34%

Status: NO LEARNING - Loss flat across all 10 epochs
```

### Direct Model (Working)
```
Epoch 10/10:
  Train Loss: 1.1239 (CLIP: 0.9070, MRL: 0.8971)
  Val Loss: ~0.0 (best models saved)
  Retrieval: Improving over time

Status: LEARNING - Loss decreasing consistently
```

**Loss Difference:** The perceiver model's loss is **5.4x higher** and not decreasing!

---

## Why Direct Model Works

The direct model works because:
1. **No Perceiver bottleneck**: Direct projection from CLIP patches → 4096d
2. **Learnable attention pooling**: Better than mean pooling
3. **Proper learning rate**: 2e-4 is stable for fine-tuning projectors
4. **Larger batch size**: 128 samples provide more negative pairs

---

## Impact Analysis

### Architecture Flow Comparison

**Perceiver Path (Broken):**
```
CLIP patches (B, 577, 1024)
    ↓ Linear projector
(B, 577, 1024) latent_dim
    ↓ Perceiver [❌ BROKEN INIT]
(B, 64, 1024) compressed latents
    ↓ Linear final_projector
(B, 64, 4096)
    ↓ Mean pooling [❌ THROWS AWAY INFO]
(B, 4096)
    ↓ L2 normalize
    ↓ MRL projection
```

**Direct Path (Working):**
```
CLIP patches (B, 577, 1024)
    ↓ Linear projector
(B, 577, 4096)
    ↓ Learnable attention pooling [✓ WORKS]
(B, 4096)
    ↓ L2 normalize
    ↓ MRL projection
```

---

## Recommended Fixes

### CRITICAL FIX 1: Fix Perceiver Initialization

**File:** `encoders/perceiver.py:155-156`

```python
# Current (BROKEN):
self.latents = nn.Parameter(torch.randn(num_latents, dim))
nn.init.normal_(self.latents, std=0.02)

# Fix Option A - Remove redundant init:
self.latents = nn.Parameter(torch.randn(num_latents, dim) * 0.02)

# Fix Option B - Use proper init (RECOMMENDED):
self.latents = nn.Parameter(torch.empty(num_latents, dim))
nn.init.normal_(self.latents, mean=0.0, std=0.02)
```

### FIX 2: Lower Learning Rate

**File:** `configs/perceiver_mrl_alignment.yaml:70`

```yaml
optimization:
  lr: 0.0002  # Change from 0.002 to 0.0002
```

### FIX 3: Increase Batch Size

**File:** `configs/perceiver_mrl_alignment.yaml:14`

```yaml
dataset:
  batch_size: 128  # Change from 64 to 128
```

### FIX 4: Add Learnable Pooling After Perceiver

**File:** `encoders/vision.py:168`

```python
# Instead of mean pooling:
if self.use_perceiver:
    latents = self.perceiver(sequence)
    latents = self.final_projector(latents)

    # Add learnable attention pooling
    if self.perceiver_attention_pool is None:
        self.perceiver_attention_pool = SimpleAttentionPooling(
            input_dim=projection_dim, dropout=0.1
        )
    pooled = self.perceiver_attention_pool(latents)
```

---

## Verification Steps

After applying fixes:

1. **Check gradient flow:**
   ```python
   for name, param in model.named_parameters():
       if "perceiver" in name and param.grad is not None:
           print(f"{name}: grad_norm={param.grad.norm().item():.6f}")
   ```

2. **Monitor loss components separately:**
   - CLIP loss should start ~4-5 and decrease
   - MRL loss should track similarly
   - Total loss should be decreasing by epoch 2

3. **Check retrieval metrics:**
   - R@1 should be >1% by epoch 3
   - R@5 should be >5% by epoch 5
   - R@10 should be >10% by epoch 10

---

## Additional Observations

### Loss Scales Are Suspicious

Both CLIP and MRL losses are **~4.8**, which is unusually high for contrastive loss:
- Normal CLIP loss starts around **2-3** for untrained
- Loss of **4.8** suggests:
  - Embeddings are not normalized properly, OR
  - Loss temperature is incorrect, OR
  - Gradients are not flowing

### Text Encoder Uses Different Model

**Perceiver config:** Uses `openai/clip-vit-large-patch14-336` text encoder
**Direct config:** Uses `sentence-transformers/all-mpnet-base-v2`

This shouldn't break training, but worth noting for reproducibility.

---

## Conclusion

The primary issue is the **Perceiver latent initialization bug** combined with:
1. Learning rate 10x too high
2. Batch size 2x too small
3. Mean pooling after compression loses information

The direct model works because it avoids the perceiver bottleneck entirely and uses proper attention pooling.

**Estimated fix impact:** After fixing initialization + hyperparameters, expect to see:
- Loss decreasing by epoch 2
- R@1 >5% by epoch 5
- Comparable performance to direct model by epoch 10

---

## Next Steps

1. Apply CRITICAL FIX 1 immediately
2. Update hyperparameters (fixes 2-3)
3. Restart training from scratch
4. Monitor first 3 epochs closely
5. If still not working, add learnable pooling (fix 4)
