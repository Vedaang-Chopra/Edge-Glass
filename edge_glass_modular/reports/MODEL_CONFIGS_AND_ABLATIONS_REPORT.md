# Model Configurations & Ablations Report

## 1. Configuration System Overview

The project utilizes a strictly typed hierarchical configuration system defined in `src/config.py`. This ensures type safety and clear documentation of all hyperparameters using usage of Python `dataclasses`.

### 1.1 Config Structure
*   **`ExperimentConfig`** (Root)
    *   `dataset`: `DatasetConfig` (paths, batch_sizes, image_sizes)
    *   `vision_encoder`: `EncoderConfig` (backbone, projection, perceiver settings)
    *   `text_encoder`: `EncoderConfig`
    *   `audio_encoder`: `EncoderConfig`
    *   `decoder`: `DecoderConfig` (LLM type, LoRA params, quantization)
    *   `fusion`: `FusionConfig` (strategy, layers)
    *   `losses`: `LossConfig` (weights for Clip/MRL)
    *   `optimization`: `OptimizationConfig` (LR, scheduler, precision)
    *   `trainer`: `TrainerConfig` (epochs, logging, checkpointing)

### 1.2 Loading Mechanism
YAML files (e.g., `configs/pixmo_alignment.yaml`) are loaded and mapped to these dataclasses. The system supports overriding config values via CLI arguments (e.g., `--optimization.lr 1e-4`).

---

## 2. Model Ablations & Switches

The architecture is highly modular, with behavior controlled by specific boolean flags and string enums in the configuration.

### 2.1 Vision Encoder Ablations (`src/encoders/vision.py`)

| Feature | Config Flag | Description |
| :--- | :--- | :--- |
| **Perceiver Resampler** | `vision_encoder.use_perceiver` | **True**: Uses Perceiver Resampler to compress patch tokens into `perceiver_num_latents`.<br>**False**: Uses MLP projection + Pooling. |
| **Perceiver Depth** | `perceiver_num_layers` | Depth of the Perceiver cross-attention stack (usually 3 or 4). |
| **Pooling Strategy** | `use_attention_pooling` | **True**: Uses learnable Attention Pooling (features $\to$ 1 vector).<br>**False**: Uses standard CLS token or Mean pooling. (Ignored if Perceiver is ON). |
| **Matryoshka (MRL)** | `use_mrl` | **True**: Enables MRL auxiliary heads. Computes loss at multiple granularities (e.g., 128, 256). |

### 2.2 Decoder Ablations (`src/decoders/`)

| Feature | Config Flag | Description |
| :--- | :--- | :--- |
| **Decoder Type** | `decoder.type` | **"qwen"**: Uses pre-trained Qwen LLM (`src/decoders/qwen.py`).<br>**"trm"**: Uses custom Tiny Recursive Model (`src/decoders/trm.py`). |
| **Quantization** | `decoder.load_in_4bit` | **True**: Loads LLM in NF4 precision via bitsandbytes to save memory. |
| **LoRA Tuning** | `decoder.use_lora` | **True**: Freezes LLM weights and trains low-rank adapters (`peft`). Defined by `r`, `alpha`, `target_modules`. |
| **Recursion** | `use_trm_recursion` | **True**: (VLM only) Enables latent recursive refinement loop on top of Qwen hidden states. |

### 2.3 Loss Ablations (`src/models/losses.py`)

*   **`sample_single_mrl_dim`**:
    *   **True**: Randomly selects *one* dimension from `mrl_dimensions` per batch step. Reduces VRAM usage and speeds up backward pass.
    *   **False**: Computes gradients for ALL dimensions every step.

---

## 3. Decoder API & Interface

The system implements a unified interface pattern for Decoders to handle the unique requirement of **visual prefix injection**.

### 3.1 The `QwenDecoder` Interface (`src/decoders/qwen.py`)

Unlike standard HuggingFace models, this wrapper class exposes specific arguments for Multimodality:

```python
def forward(
    self,
    input_ids: torch.Tensor,           # (B, Seq_Len)
    attention_mask: torch.Tensor,      # (B, Seq_Len)
    prefix_embeds: torch.Tensor,       # (B, Prefix_Len, Hidden_Dim) -> VISUAL TOKENS
    labels: torch.Tensor,
    ...
)
```

**Key Implementation Details**:
1.  **Embedding Concatenation**:
    *   Gets text embeddings from `input_ids`.
    *   Concatenates `[prefix_embeds, text_embeddings]` along sequence dimension.
2.  **Mask Extension**:
    *   Automatically extends `attention_mask` with 1s for the prefix length.
3.  **Label Alignment**:
    *   Prepends `-100` (ignore index) to `labels` for the duration of the prefix, ensuring the model is not penalized for predicting the image tokens.

### 3.2 TRM (Transformer-based Recurrent Memory)

An experimental decoder designed for "thinking" or refinement steps.
*   **Logic**: Instead of a deep stack of N layers, uses a shallow stack (e.g., 2 layers) that is applied recursively $K$ times on the hidden state $H$.
*   **State**: Maintains a reasoning state vector $Z$.
*   **Update**: $H_{t+1}, Z_{t+1} = Layer(H_t, Z_t)$.

---

## 4. Configuration Compatibility Matrix

| Config Profile | Encoder Output | Decoder | Key Use Case |
| :--- | :--- | :--- | :--- |
| **`pixmo_alignment.yaml`** | 4096 dim (Pooled) | N/A | High-fidelity retrieval, no generation. |
| **`perceiver_mrl_alignment.yaml`** | 512 dim (Pooled) + 64 Latents | N/A | Efficient retrieval, compressed visual rep. |
| **`trm_vlm_qa_3b_reg.yaml`** | 64 Latents (Perceiver) | Qwen-3B (4-bit) | VQA, Captioning on consumer hardware. |

### Config Difference Highlights
*   **Optimization**:
    *   Alignment LR: `5e-5` (Conservative for contrastive).
    *   VLM LR: `2e-5` (Very conservative for fine-tuning).
    *   VLM uses high `weight_decay` (0.2) and `dropout` (0.2) to fight overfitting on the small PixMo QA dataset.
*   **Dimensions**:
    *   Alignment models target `4096` or `512` based on whether they need to match text embeddings (CLIP-L is 768, usually projected up or down).
    *   MRL schedules differ: `[2048, 1024...]` for high-dim, `[256, 128...]` for low-dim.
