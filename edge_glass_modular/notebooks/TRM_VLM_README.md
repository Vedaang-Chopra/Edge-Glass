# TRM-Style Vision-Language Model (VLM) Implementation

## Overview

This implementation builds a **TRM-style VLM** that combines:
1. **Pretrained aligned vision encoder** (Perceiver + MRL, frozen)
2. **Plain tiny decoder baseline** (no recursion)
3. **TRM recursive decoder** (with latent reasoning states)
4. Training on **PixMo QA dataset**
5. Evaluation with **Exact Match (EM)** and **Token F1** metrics

## Architecture

```
Image (B, 3, 336, 336)
  ↓
CLIP Vision Encoder (frozen)
  ↓
Perceiver Resampler (frozen)
  ↓
MRL Projection (frozen)
  ↓ (B, 64, 4096) - vision tokens
Projection to decoder dim (trainable)
  ↓ (B, 64, d_dec)
Token Layout: [IMG_TOKENS] [QUESTION_TOKENS] [ANSWER_TOKENS]
  ↓
TRM Decoder / Plain Decoder (trainable)
  ↓
LM Head → Logits
  ↓
Loss (only on answer tokens)
```

## Key Components

### 1. Token Layout

Following the TRM design specification, we use a fixed token order:

```
[IMG TOKENS] [QUESTION TOKENS] [ANSWER TOKENS]
├─ 64 tokens ─┤├─ ~10-20 ─┤├─ ~16-32 ─┤
```

**Masking Strategy:**
- **Causal masking**: Each token can only attend to previous tokens
- Answer tokens can see:
  - All image tokens (prefix)
  - All question tokens
  - Previous answer tokens only (not future)
- **Loss computation**: Only on answer tokens
  - Image tokens: label = -100 (ignored)
  - Question tokens: label = -100 (ignored)
  - Answer tokens: label = actual token IDs

### 2. Aligned Vision Encoder

**Architecture:**
- CLIP ViT-L/14 (frozen)
- Perceiver Resampler: 64 latents × 1024d × 4 layers (frozen)
- MRL Projection: 1024 → 4096 (frozen)

**Output:**
- Shape: `(B, 64, 4096)`
- 64 vision tokens per image
- 4096-dimensional embeddings (matches Qwen hidden size)

**Loading:**
```python
# Load from checkpoint
aligned_model = MultimodalAlignmentModel(config)
checkpoint = torch.load("checkpoints/perceiver_mrl_alignment/checkpoint_best.pt")
aligned_model.load_state_dict(checkpoint['model_state_dict'])
aligned_model.eval()

# Freeze all parameters
for param in aligned_model.parameters():
    param.requires_grad = False

# Extract vision encoder
@torch.no_grad()
def encode_images(images):
    vision_output = aligned_model.vision_encoder(images)
    return vision_output.sequence  # (B, 64, 4096)
```

### 3. Plain Tiny Decoder (Baseline)

**Architecture:**
- Hidden dim: 512 or 1024
- Layers: 4 transformer decoder layers
- Heads: 8
- Token embedding + positional encoding (RoPE)
- LM head tied with embeddings

**Forward Pass:**
```python
# Project vision tokens: (B, 64, 4096) → (B, 64, d_dec)
vision_emb = self.vision_proj(vision_tokens)

# Embed text tokens
question_emb = self.embed_tokens(question_ids)  # (B, L_q, d_dec)
answer_emb = self.embed_tokens(answer_ids)      # (B, L_a, d_dec)

# Concatenate: [vision | question | answer]
full_sequence = torch.cat([vision_emb, question_emb, answer_emb], dim=1)

# Pass through decoder
for layer in self.layers:
    full_sequence = layer(full_sequence)

# Get logits
logits = self.lm_head(full_sequence)

# Compute loss (only on answer tokens)
# Labels: [-100, -100, ..., -100, ans_id_1, ans_id_2, ...]
```

### 4. TRM Recursive Decoder

**Core Idea:**
Instead of a deep transformer, use a **tiny transformer** (2 layers) with **recursive reasoning**.

**Variables:**
- `x`: Context tokens = `[IMG_TOKENS] + [QUESTION_TOKENS]` (fixed)
- `y`: Answer embeddings (updated each recursion)
- `z`: Latent reasoning state (updated each recursion)

**Inner Recursion (n steps):**
```python
# Initialize
x = [vision_emb | question_emb]  # Context (fixed)
y = embed_tokens(answer_ids)      # Answer (teacher-forced)
z = z_init.expand(B, L_ans, d)    # Latent (learned initialization)

# Repeat n times (e.g., n=4)
for _ in range(n):
    # Concatenate along sequence
    concat = torch.cat([x, y, z], dim=1)  # (B, L_ctx + 2*L_ans, d)

    # Pass through tiny transformer (2 layers)
    concat' = TinyTransformer(concat)

    # Split back
    x', y', z' = split(concat')

    # Update y and z (x stays fixed)
    y = y'
    z = z'

# Final output from y
logits = lm_head(y)
```

**Key Differences from Plain Decoder:**
1. **Smaller network**: 2 layers instead of 4
2. **Recursion**: Network runs `n` times on same weights
3. **Latent state `z`**: Stores intermediate reasoning
4. **Context-answer separation**: `x` is fixed, only `y` and `z` evolve

### 5. Dataset: PixMo QA

**Source:** PixMo captioning dataset converted to QA format

**Conversion:**
```python
# Original: {image, caption}
# Converted: {image, question, answer}

QUESTIONS = [
    "What is in this image?",
    "Describe this image.",
    "What do you see?",
    "What is shown in the image?",
    "Describe what you see.",
]

# Randomly sample question per image
question = random.choice(QUESTIONS)
answer = caption
```

**Tokenization:**
```python
# Question tokens (with prompt format)
question_tokens = tokenizer(f"Question: {question} Answer:")

# Answer tokens (ground truth)
answer_tokens = tokenizer(answer)

# Max lengths
max_question_length = 32
max_answer_length = 32
```

### 6. Training

**Hyperparameters:**
```yaml
num_epochs: 10
learning_rate: 1e-4
weight_decay: 0.01
betas: [0.9, 0.95]
max_grad_norm: 1.0
warmup_ratio: 0.05
batch_size: 32
```

**Optimizer:** AdamW with cosine learning rate schedule

**Training Loop:**
1. Encode images (frozen): `vision_tokens = encode_images(images)`
2. Forward pass: `outputs = model(vision_tokens, question_ids, answer_ids)`
3. Backward pass with gradient clipping
4. Update only decoder parameters (vision encoder frozen)

**Checkpointing:**
- Save best model based on validation loss
- Save every epoch for ablations

### 7. Evaluation Metrics

**Exact Match (EM):**
```python
def normalize_answer(s):
    # Remove punctuation, lowercase, remove articles
    s = ''.join(ch for ch in s if ch not in string.punctuation)
    s = s.lower().strip()
    s = ' '.join([w for w in s.split() if w not in {'a', 'an', 'the'}])
    return s

def compute_exact_match(pred, target):
    return float(normalize_answer(pred) == normalize_answer(target))
```

**Token F1:**
```python
def compute_f1(pred, target):
    pred_tokens = normalize_answer(pred).split()
    target_tokens = normalize_answer(target).split()

    common = Counter(pred_tokens) & Counter(target_tokens)
    num_common = sum(common.values())

    precision = num_common / len(pred_tokens)
    recall = num_common / len(target_tokens)
    f1 = 2 * precision * recall / (precision + recall)

    return f1
```

## Usage

### 1. Training

**Option A: Use Jupyter Notebook**
```bash
cd edge_glass_modular/notebooks
jupyter notebook 03_trm_vlm_qa_training.ipynb
```

**Option B: Convert to Python Script**
```bash
jupyter nbconvert --to script 03_trm_vlm_qa_training.ipynb
python 03_trm_vlm_qa_training.py
```

### 2. Configuration

Edit `configs/trm_vlm_qa.yaml` to change:
- Decoder type: `trm` or `baseline`
- Hidden dimension: 256, 512, 1024
- Number of layers: 2, 4, 6
- Recursion depth: `num_inner_steps` = 2, 4, 6, 8

### 3. Experiments

**Baseline vs TRM:**
```python
# In notebook, change:
USE_TRM = False  # Baseline
USE_TRM = True   # TRM
```

**Different Recursion Depths:**
```python
NUM_INNER_STEPS = 2  # Shallow recursion
NUM_INNER_STEPS = 4  # Default
NUM_INNER_STEPS = 6  # Deep recursion
NUM_INNER_STEPS = 8  # Very deep recursion
```

**Different Hidden Dimensions:**
```python
HIDDEN_DIM = 256   # Small
HIDDEN_DIM = 512   # Default
HIDDEN_DIM = 1024  # Large
```

## Ablation Studies

Following the checklist, run these experiments:

### 1. Vision vs No Vision
- **With vision**: `[IMG_TOKENS] + [Q_TOKENS] → [A_TOKENS]`
- **Without vision**: `[Q_TOKENS] → [A_TOKENS]`

### 2. TRM vs Plain Decoder
- Same parameter budget
- Compare EM and F1 scores

### 3. Recursion Depth
- n = {2, 4, 6, 8} inner steps
- Plot EM/F1 vs recursion depth

### 4. Hidden Dimension
- d = {256, 512, 1024}
- Compare parameter count vs performance

### 5. Alignment Variants
- Use PixMo alignment checkpoint vs Perceiver+MRL checkpoint
- Same decoder, different vision encoder

## Expected Results

**Baseline (Plain Decoder):**
- Parameters: ~15-30M (depending on hidden_dim)
- EM: ~5-15% (image captioning is hard)
- F1: ~20-40%

**TRM (Recursive Decoder):**
- Parameters: ~10-20M (smaller due to fewer layers)
- EM: ~8-20% (should improve with recursion)
- F1: ~25-45%

**Hypothesis:**
- TRM should match or exceed baseline performance with fewer parameters
- Deeper recursion (higher n) should improve performance
- Vision tokens should significantly help (compared to text-only)

## File Structure

```
edge_glass_modular/
├── notebooks/
│   ├── 02_perceiver_mrl_vision_text_alignment.ipynb  # Alignment training
│   ├── 03_trm_vlm_qa_training.ipynb                  # TRM VLM training (NEW)
│   ├── TRM_VLM_README.md                             # This file
│   └── checkpoints/
│       ├── perceiver_mrl_alignment/
│       │   └── checkpoint_best.pt                     # Pretrained vision encoder
│       └── trm_vlm_qa/
│           └── checkpoint_best.pt                     # Trained TRM VLM
├── configs/
│   ├── perceiver_mrl_alignment.yaml                  # Alignment config
│   └── trm_vlm_qa.yaml                               # TRM VLM config (NEW)
└── src/
    ├── decoders/
    │   ├── trm.py                                    # TRM decoder (existing)
    │   └── qwen.py                                   # Qwen decoder (existing)
    └── models/
        └── alignment.py                              # Alignment model (existing)
```

## Troubleshooting

### Issue: CUDA out of memory
**Solution:**
- Reduce batch size: `batch_size = 16` or `batch_size = 8`
- Reduce hidden_dim: `hidden_dim = 256`
- Enable gradient checkpointing (add to TRM decoder)

### Issue: Training loss not decreasing
**Solution:**
- Check that vision encoder is frozen
- Verify token layout is correct
- Ensure loss mask is properly applied
- Try lower learning rate: `lr = 5e-5`

### Issue: Poor EM/F1 scores
**Expected:** Image captioning QA is challenging
- EM < 20% is normal for this task
- F1 is more meaningful metric
- Compare relative improvements (TRM vs baseline)

### Issue: Generation produces gibberish
**Solution:**
- Check tokenizer decoding
- Verify temperature is reasonable (0.7-1.0)
- Try greedy decoding first (`temperature=1.0, do_sample=False`)

## Future Improvements

1. **Outer Deep Recursion (T > 1):**
   - Run T-1 recursions without gradients
   - Final recursion with gradients + supervision

2. **Better QA Dataset:**
   - Use actual VQA datasets (VQAv2, GQA, etc.)
   - More diverse question types

3. **Longer Contexts:**
   - Increase max_seq_len for longer answers
   - Add document-grounded QA

4. **Multimodal Decoder:**
   - Combine vision + audio + text
   - Trimodal TRM

5. **Efficient Training:**
   - Gradient checkpointing
   - Mixed precision (FP16/BF16)
   - Distributed training (DDP)

## References

1. **TRM Paper**: "Tiny Recursive Models" (check for official paper)
2. **Perceiver**: "Perceiver: General Perception with Iterative Attention"
3. **MRL**: "Matryoshka Representation Learning"
4. **PixMo**: Allenai PixMo dataset
5. **Qwen**: Qwen2 technical report

## Questions?

For issues or questions:
1. Check the notebook cells for inline comments
2. Review this README
3. Inspect the model code in `src/decoders/trm.py`
4. Check WandB logs for training curves

Good luck with your experiments! 🚀
