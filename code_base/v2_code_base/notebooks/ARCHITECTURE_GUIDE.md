# 🧠 Multimodal Alignment Architecture: Complete Guide

## Overview

This document explains the complete multimodal alignment system with Perceiver Resampler, covering:
1. **What** each component does
2. **Why** it's designed this way
3. **How** the data flows through the system

---

## 🏗️ High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                 INPUT LAYER                                      │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   🖼️ IMAGE              🔊 AUDIO                📝 TEXT                         │
│   (224×224×3)           (waveform)              (string)                        │
│                                                                                  │
└────────┬───────────────────────┬────────────────────────┬───────────────────────┘
         │                       │                        │
         ▼                       ▼                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           FROZEN ENCODERS                                        │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   CLIP ViT-B/32         Whisper-base            Sentence-BERT                   │
│   ─────────────         ────────────            ─────────────                   │
│   • 12 layers           • 6 layers              • 6 layers                      │
│   • 768 hidden dim      • 512 hidden dim        • 384 hidden dim                │
│   • 86M params          • 74M params            • 22M params                    │
│                                                                                  │
│   Output: (B,50,768)    Output: (B,1500,512)    Output: (B,L,384)               │
│   50 patch tokens       1500 audio frames       L text tokens                   │
│                                                                                  │
└────────┬───────────────────────┬────────────────────────┬───────────────────────┘
         │                       │                        │
         ▼                       ▼                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         TRAINABLE ADAPTERS                                       │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   Vision Adapter        Audio Adapter           Text Adapter                    │
│   768 → 512             512 → 512               384 → 512                       │
│                                                                                  │
│   LayerNorm             LayerNorm               LayerNorm                       │
│   Linear(768,1536)      Linear(512,1024)        Linear(384,768)                 │
│   GELU                  GELU                    GELU                            │
│   Linear(1536,512)      Linear(1024,512)        Linear(768,512)                 │
│                                                                                  │
│   Output: (B,50,512)    Output: (B,1500,512)    Output: (B,L,512)               │
│                                                                                  │
└────────┬───────────────────────┬────────────────────────┬───────────────────────┘
         │                       │                        │
         └───────────────────────┼────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        PERCEIVER RESAMPLER                                       │
│                    (Shared across all modalities)                                │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   ┌─────────────────────────────────────────────────────────────────────┐       │
│   │                     LEARNED LATENT QUERIES                          │       │
│   │                        (64 × 512)                                   │       │
│   │                                                                     │       │
│   │   These are trainable parameters that learn to "ask questions"     │       │
│   │   about the input. Think of them as 64 specialized "experts"       │       │
│   │   that each focus on extracting different types of information.    │       │
│   └─────────────────────────────────────────────────────────────────────┘       │
│                                                                                  │
│   FOR EACH LAYER (×4):                                                          │
│   ┌─────────────────────────────────────────────────────────────────────┐       │
│   │ 1. CROSS-ATTENTION                                                  │       │
│   │    ─────────────────                                                │       │
│   │    Q = Latent Queries (64 tokens)                                   │       │
│   │    K = V = Input Tokens (50/1500/L tokens)                          │       │
│   │                                                                     │       │
│   │    Latents "read" from input, extracting relevant information       │       │
│   │    Complexity: O(64 × T) - efficient for long sequences!           │       │
│   ├─────────────────────────────────────────────────────────────────────┤       │
│   │ 2. SELF-ATTENTION                                                   │       │
│   │    ───────────────                                                  │       │
│   │    Q = K = V = Latents (64 tokens)                                  │       │
│   │                                                                     │       │
│   │    Latents communicate with each other to share information         │       │
│   │    Complexity: O(64²) - constant regardless of input length        │       │
│   ├─────────────────────────────────────────────────────────────────────┤       │
│   │ 3. FEED-FORWARD NETWORK                                             │       │
│   │    ─────────────────────                                            │       │
│   │    FFN(x) = Linear(GELU(Linear(x)))                                 │       │
│   │                                                                     │       │
│   │    Non-linear transformation applied to each latent independently   │       │
│   └─────────────────────────────────────────────────────────────────────┘       │
│                                                                                  │
│   Output: (B, 64, 512) - FIXED SIZE regardless of input!                        │
│                                                                                  │
└────────────────────────────────────┬────────────────────────────────────────────┘
                                     │
                    ┌────────────────┼────────────────┐
                    │                │                │
                    ▼                ▼                ▼
┌───────────────────────┐ ┌───────────────────┐ ┌───────────────────────┐
│  ALIGNMENT PROJECTOR  │ │   LLM PROJECTOR   │ │   RAW LATENTS         │
│  (For Retrieval)      │ │   (For Generation)│ │   (For Analysis)      │
├───────────────────────┤ ├───────────────────┤ ├───────────────────────┤
│                       │ │                   │ │                       │
│  Mean Pool + Linear   │ │  Linear           │ │  Direct output        │
│  (B,64,512)→(B,512)   │ │  512 → 1536       │ │  (B,64,512)           │
│                       │ │  (B,64,1536)      │ │                       │
│  L2 Normalize         │ │                   │ │  Use for:             │
│                       │ │  Becomes LLM      │ │  • Visualization      │
│  Use for:             │ │  prefix tokens    │ │  • Probing            │
│  • Retrieval          │ │                   │ │  • Debugging          │
│  • Classification     │ │  Use for:         │ │                       │
│  • Similarity         │ │  • Captioning     │ │                       │
│                       │ │  • VQA            │ │                       │
│                       │ │  • ASR            │ │                       │
└───────────────────────┘ └───────────────────┘ └───────────────────────┘
```

---

## 🔄 Why Perceiver Resampler?

### Problem: Variable-Length Sequences

Different modalities produce different sequence lengths:
- **Image**: 50 patches (7×7 grid + CLS)
- **Audio**: ~1500 frames (30 seconds at 50Hz)
- **Text**: Variable (depends on sentence length)

This creates problems:
1. Can't directly compare embeddings
2. Self-attention is O(T²) - expensive for long audio
3. LLMs expect fixed-size inputs

### Solution: Perceiver Resampler

The Perceiver uses **learned queries** to compress any input to fixed size:

```
Input: Variable length T          Output: Fixed length K (e.g., 64)

┌─────────────────────────┐       ┌─────────────────────────┐
│ Image:  50 patches      │       │                         │
│ Audio:  1500 frames     │  ───► │   64 latent vectors     │
│ Text:   32 tokens       │       │                         │
└─────────────────────────┘       └─────────────────────────┘
```

### How Cross-Attention Achieves This

```
Latent Queries (64 × 512):
┌────┬────┬────┬────┬─────┬────┐
│ q₁ │ q₂ │ q₃ │ q₄ │ ... │q₆₄│  ← Learned "questions"
└────┴────┴────┴────┴─────┴────┘
  │    │    │    │         │
  ▼    ▼    ▼    ▼         ▼
  
Input Tokens (T × 512):
┌────┬────┬────┬────┬─────┬────┐
│ t₁ │ t₂ │ t₃ │ t₄ │ ... │ tₜ│  ← Encoder output
└────┴────┴────┴────┴─────┴────┘

Cross-Attention:
  • Each query attends to ALL input tokens
  • Attention weights determine "how much" to read from each token
  • Output: 64 latent vectors, each a weighted combination of inputs
```

### Benefits

| Aspect | Without Perceiver | With Perceiver |
|--------|-------------------|----------------|
| Memory | O(T²) for self-attn | O(K×T) cross-attn |
| Output Size | Variable | Fixed (64) |
| Long Audio | Very expensive | Efficient |
| Modality Comparison | Difficult | Easy |
| LLM Integration | Need pooling | Natural prefix |

---

## 📊 Loss Functions Explained

### 1. Contrastive Loss (InfoNCE / CLIP Loss)

**Goal**: Bring matching pairs together, push non-matching apart.

```
Similarity Matrix (B × B):

              Text Embeddings
           t₁    t₂    t₃    t₄
         ┌─────┬─────┬─────┬─────┐
    i₁   │ 0.9 │ 0.1 │ 0.2 │ 0.1 │  ← Should be highest on diagonal
Image    ├─────┼─────┼─────┼─────┤
Embs i₂  │ 0.2 │ 0.8 │ 0.1 │ 0.3 │
         ├─────┼─────┼─────┼─────┤
    i₃   │ 0.1 │ 0.2 │ 0.85│ 0.1 │
         ├─────┼─────┼─────┼─────┤
    i₄   │ 0.3 │ 0.1 │ 0.2 │ 0.7 │
         └─────┴─────┴─────┴─────┘

Loss = CrossEntropy(logits, diagonal_labels)
     = -log(exp(sim[i,i]) / Σⱼ exp(sim[i,j]))
```

**Temperature** (τ = 0.07): Controls sharpness of softmax
- Lower τ → Sharper distribution → Harder negatives
- Higher τ → Softer distribution → Easier training

### 2. Matryoshka Representation Learning (MRL)

**Goal**: Pack important information in early dimensions.

```
Full embedding: [d₁, d₂, d₃, d₄, ..., d₅₁₂]
                 ├────────────────────────────┤
                         512 dimensions

MRL trains at multiple truncations:
  • dims[:64]  → Loss₁   (hardest, most compressed)
  • dims[:128] → Loss₂
  • dims[:256] → Loss₃
  • dims[:512] → Loss₄   (full, easiest)

Total Loss = (Loss₁ + Loss₂ + Loss₃ + Loss₄) / 4
```

**Benefits**:
- **Flexible deployment**: Use 64 dims for fast search, 512 for accuracy
- **Better gradients**: Multiple objectives = richer training signal
- **Implicit curriculum**: Small dims are harder, provide challenge

---

## 🎯 Training Strategy

### Phase 1: Multimodal Alignment

**Objective**: Create unified embedding space for all modalities.

```python
# What's trained
trainable = [
    vision_adapter,    # 768 → 512
    audio_adapter,     # 512 → 512
    text_adapter,      # 384 → 512
    perceiver,         # Shared resampler
    alignment_proj,    # Latents → aligned embeddings
]

# What's frozen
frozen = [
    clip_encoder,      # ~86M params
    whisper_encoder,   # ~74M params
    text_encoder,      # ~22M params
]

# Loss
loss = mrl_weight * matryoshka_loss(z_vision, z_text) + 
       clip_weight * contrastive_loss(z_vision, z_text)
```

**Data**: Image-text pairs (COCO, Conceptual Captions) + Audio-text pairs (AudioCaps)

### Phase 2: LLM Integration

**Objective**: Enable LLM to understand multimodal inputs.

```python
# What's trained
trainable = [
    llm_projector,     # 512 → D_llm
    # Optional: LoRA adapters on LLM
]

# What's frozen
frozen = [
    everything_from_phase1,  # Keep alignment intact
    llm_base_weights,        # Keep language abilities
]

# Loss
loss = language_modeling_loss(
    input=concat(multimodal_prefix, text_tokens),
    target=text_tokens
)
```

---

## 🔍 Component Deep Dive

### Modality Adapters

**Purpose**: Bridge dimension gap between encoders and Perceiver.

```python
class MLPAdapter:
    """
    Why MLP instead of Linear?
    
    Linear: y = Wx + b
      • Fast, but limited expressivity
      • Okay for small dimension changes
    
    MLP: y = W₂(GELU(W₁(LN(x))))
      • More expressive transformation
      • LayerNorm stabilizes training
      • GELU adds non-linearity
      • Better for cross-domain mapping
    """
    
    def __init__(self, in_dim, out_dim, hidden_factor=2.0):
        hidden = int(in_dim * hidden_factor)
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),      # Normalize input
            nn.Linear(in_dim, hidden),  # Expand
            nn.GELU(),                  # Non-linearity
            nn.Dropout(0.1),            # Regularization
            nn.Linear(hidden, out_dim), # Project to target
        )
```

### Perceiver Layer

**Purpose**: Extract and refine information from input.

```python
class PerceiverLayer:
    """
    Three-step process per layer:
    
    1. Cross-Attention: READ from input
       - Latents query the input tokens
       - Each latent decides what to pay attention to
       - Information flows: Input → Latents
    
    2. Self-Attention: COMMUNICATE among latents
       - Latents share information with each other
       - Helps coordinate what each latent represents
       - Information flows: Latents ↔ Latents
    
    3. FFN: TRANSFORM latent representations
       - Non-linear transformation
       - Each latent processed independently
       - Adds model capacity
    """
    
    def forward(self, latents, tokens, mask):
        # Step 1: Cross-attention
        latents = latents + self.cross_attn(
            q=latents,      # 64 queries
            kv=tokens,      # T keys/values
            mask=mask
        )
        
        # Step 2: Self-attention
        latents = latents + self.self_attn(
            q=latents,
            kv=latents      # Same source
        )
        
        # Step 3: FFN
        latents = latents + self.ffn(self.ln(latents))
        
        return latents
```

### Number of Layers

**How many Perceiver layers?**

| Layers | Quality | Speed | Use Case |
|--------|---------|-------|----------|
| 1-2 | Lower | Fast | Quick prototyping |
| 4 | Good | Balanced | **Recommended** |
| 6-8 | Best | Slow | Maximum quality |

More layers = more refinement of latents, but diminishing returns.

---

## 📈 Data Flow Example

Let's trace an image through the system:

```
INPUT: Photo of a golden retriever playing fetch in a park

Step 1: CLIP Encoding
────────────────────
Image (224×224×3) 
  → Patch embed (196 patches + CLS = 197 tokens)
  → 12 Transformer layers
  → Output: (1, 50, 768)  # Taking 50 patches
  
  Patches encode: [grass, dog_head, dog_body, ball, sky, trees, ...]

Step 2: Vision Adapter
─────────────────────
(1, 50, 768) → MLP → (1, 50, 512)

  Transforms CLIP features to Perceiver dimension
  Learns task-relevant transformations

Step 3: Perceiver Resampler (4 layers)
─────────────────────────────────────
Initialize: 64 learned latent queries

Layer 1:
  Cross-Attn: Latents attend to patches
    - Query 1 might focus on "animal" patches
    - Query 2 might focus on "action" patches
    - Query 3 might focus on "environment" patches
  Self-Attn: Latents share findings
  FFN: Refine representations

Layer 2-4: Continue refining...

Output: (1, 64, 512)
  - 64 latent vectors, each capturing different aspects
  - Some latents: "golden retriever"
  - Some latents: "playing/running"
  - Some latents: "park setting"
  - Some latents: "ball/toy"

Step 4a: For Retrieval
─────────────────────
(1, 64, 512) → MeanPool → (1, 512) → L2 Norm

Final embedding captures: "golden retriever playing fetch in park"
Can compare with text embedding: "A dog playing in the grass"

Step 4b: For Generation
──────────────────────
(1, 64, 512) → Linear → (1, 64, 1536)

64 prefix tokens for LLM, each carrying different visual information
LLM generates: "A golden retriever is playing fetch with a ball 
                in a sunny park with green grass."
```

---

## 🛠️ Implementation Tips

### 1. Start Simple, Then Add Complexity

```python
# Week 1: Linear adapters, 2 Perceiver layers
config = Config(
    num_perceiver_layers=2,
    adapter_type='linear',
)

# Week 2: MLP adapters, 4 Perceiver layers
config = Config(
    num_perceiver_layers=4,
    adapter_type='mlp',
)

# Week 3: Add MRL loss
config = Config(
    mrl_dims=(64, 128, 256, 512),
    mrl_weight=1.0,
)
```

### 2. Monitor These Metrics

```python
# During training
metrics = {
    'loss': total_loss,
    'loss_mrl': mrl_loss,
    'loss_clip': clip_loss,
    'alignment': compute_alignment(z_v, z_t),  # Should decrease
    'uniformity': compute_uniformity(z_v),      # Should decrease
}

# During evaluation
eval_metrics = {
    'R@1': recall_at_1,     # Primary metric
    'R@5': recall_at_5,
    'R@10': recall_at_10,
    'MRR': mean_reciprocal_rank,
}
```

### 3. Debugging Checklist

```
□ Check embedding norms (should be ~1 after L2 norm)
□ Check similarity distribution (diagonal should be highest)
□ Check gradient norms (should be stable, not exploding/vanishing)
□ Check attention weights (should be meaningful, not uniform)
□ Check latent diversity (latents should specialize)
```

---

## 📚 References

1. **Perceiver**: Jaegle et al., "Perceiver: General Perception with Iterative Attention" (2021)
2. **CLIP**: Radford et al., "Learning Transferable Visual Models From Natural Language Supervision" (2021)
3. **Flamingo**: Alayrac et al., "Flamingo: a Visual Language Model for Few-Shot Learning" (2022)
4. **ImageBind**: Girdhar et al., "ImageBind: One Embedding Space To Bind Them All" (2023)
5. **Matryoshka**: Kusupati et al., "Matryoshka Representation Learning" (2022)

---

## 🎓 Summary

| Component | Purpose | Input → Output |
|-----------|---------|----------------|
| **Frozen Encoders** | Extract rich features | Raw → (B, T, D_enc) |
| **Adapters** | Bridge to Perceiver | (B, T, D_enc) → (B, T, 512) |
| **Perceiver** | Compress to fixed size | (B, T, 512) → (B, 64, 512) |
| **Alignment Proj** | For retrieval | (B, 64, 512) → (B, 512) |
| **LLM Proj** | For generation | (B, 64, 512) → (B, 64, D_llm) |

**Key Insight**: The Perceiver is the bottleneck that enables efficient, unified multimodal representations regardless of input modality or sequence length.
