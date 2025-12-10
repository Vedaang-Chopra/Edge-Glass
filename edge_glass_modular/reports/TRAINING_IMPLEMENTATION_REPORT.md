# Training Pipeline & Implementation Report

## 1. End-to-End Pipeline Overview

The training workflow consists of two distinct phases:
1.  **Alignment Pre-training**: Aligns vision/audio encoders to text using contrastive learning.
2.  **VLM Fine-tuning**: Freezes the aligned encoder and trains a language decoder (Qwen) on VQA/Captioning tasks.

Both phases utilize **HuggingFace Accelerate** + **DeepSpeed** for distributed training, handling mixed precision (BF16), gradient accumulation, and multi-GPU orchestration.

---

## 2. Data Pipeline Implementation

### 2.1 Dataset Classes (`src/data/dataset_builder.py`)

*   **`PixmoParquetImageTextDataset`** (Alignment Phase)
    *   **Source**: Parquet files containing `image_bytes`, `caption`, `sample_id`.
    *   **Loading**:
        *   Reads parquet via pandas.
        *   Decodes `image_bytes` on-the-fly using `PIL.Image.open`.
        *   **Key Feature**: Robust error handling (`try/except`) around image loading to skip corrupted bytes without crashing the dataloader.
    *   **Augmentation**: Applies `text_dropout_prob` (probabilistically replaces caption with empty string) to force visual reliance.
    *   **Output**: Dict with `image` (Tensor), `text` (str), `sample_id`.

*   **`PixmoQADataset`** (VLM Phase)
    *   **Source**: Parquet files with `question`, `answer`, `image_bytes`.
    *   **Tokenization**:
        *   Pre-tokenizes `question` and `answer` using the LLM tokenizer.
        *   Returns `input_ids` and `attention_mask` for both components separately.
    *   **Output**: Dict including raw text and tokenized IDs.

### 2.2 Transforms (`src/data/transforms.py`)

*   **Vision**:
    *   **Training**: `Resize(size+32)` -> `RandomCrop(size)` -> `RandomHorizontalFlip` -> `ColorJitter` -> `Normalize` (CLIP stats).
    *   **Eval**: `Resize(size)` -> `Normalize`.
*   **Audio**:
    *   Loads waveform via `torchaudio`.
    *   Resamples to 16kHz.
    *   Converts to Mel Spectrogram (128 bins).

### 2.3 Collate Functions

*   **Alignment Collate**:
    *   Stacks image tensors.
    *   Collects text strings into a list (for `SentenceTransformer` which expects lists) or tokenizes them (for CLIP text encoder).
*   **VLM Collate**:
    *   Pads `question_ids` and `answer_ids` to max batch length.
    *   Creates comprehensive attention masks.
    *   Constructs labels: Sets labels for padding and query tokens to `-100` (ignored by loss), ensuring the model only learns to predict the answer.

---

## 3. Training Logic & Optimization

### 3.1 Loss Functions (`src/models/losses.py`)

*   **Contrastive Loss (InfoNCE)**:
    *   $Sim(A, B) = \frac{A \cdot B^T}{\tau}$
    *   Symmetric Cross Entropy: $L = \frac{1}{2} (CE(Sim, I) + CE(Sim^T, I))$
    *   Implemented manually for full control over temperature and reduction.

*   **Matryoshka (MRL) Loss**:
    *   Re-uses Contrastive Loss at multiple granularities.
    *   **Efficiency Trick**: `sample_single_mrl_dim=True` flag (in `AlignmentLoss`) randomly selects *one* MRL dimension (e.g., 128) per batch to compute loss on, instead of summing all gradients. This drastically reduces computation graph size during training while still supervising all dimensions over time.

### 3.2 Key Training Hyperparameters

| Parameter | Alignment Phase | VLM Phase | Notes |
| :--- | :--- | :--- | :--- |
| **Optimizer** | AdamW | AdamW | Standard choice |
| **Scheduler** | Cosine with Warmup | Cosine with Warmup | Uses `LambdaLR` |
| **Learning Rate** | `5e-5` to `1e-4` | `2e-5` | Lower for VLM fine-tuning |
| **Precision** | BF16 (preferred) | BF16 (4-bit quant) | VLM uses QLoRA 4-bit |
| **Grad Accum** | 1 or 2 | 2 | Higher in VLM to offset small batch size |
| **Gradient Clip** | 1.0 | 1.0 | Prevents exploding gradients |
| **Weight Decay** | 0.01 | 0.2 | Higher regularization in VLM config |

---

## 4. Implementation Details

### 4.1 Training Loop Pseudocode

```python
# Simplified View of the Training Loop
for epoch in range(epochs):
    model.train()
    for batch in dataloader:
        with accelerator.accumulate(model):
            # 1. Forward Pass
            # Alignment Phase:
            outputs = model(images=batch['image'], texts=batch['text'])
            
            # VLM Phase:
            # outputs = model(vision_tokens=..., question_ids=..., labels=...)
            
            # 2. Extract Loss
            loss = outputs.loss
            
            # 3. Backward & Step
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
        # 4. Logging & Eval
        if step % log_every == 0:
            wandb.log({"loss": loss.item(), "lr": lr})
            
    # 5. Checkpointing logic (see below)
```

### 4.2 Checkpointing Strategy API

The scripts implement a robust "Rotate & Best" strategy to manage disk space:

1.  **Selection**: Saves two types of checkpoints:
    *   `checkpoint_best`: Updated whenever validation metric improves.
    *   `checkpoint-epoch-X`: Periodic snapshots.
2.  **Rotation Logic**:
    *   Keeps only the **latest** epoch checkpoint (deletes `epoch-X-1` after saving `epoch-X`).
    *   Exception: Keeps checkpoints at `epoch % 7 == 0` (weekly snapshots) in `train_pixmo_alignment.py`.
3.  **Format**:
    *   **Lightweight**: `torch.save(model.state_dict())` for simple alignment models.
    *   **Full State**: `accelerator.save_state()` for VLM/DeepSpeed to preserve optimizer/scheduler states for resumption.
4.  **Resumption**:
    *   Automatically detects existing checkpoints in `output_dir`.
    *   Resumes epoch count, global step, and optimizer state.

### 4.3 Error Handling & Distributed Utilities

*   **DeepSpeed/Accelerate Integration**: Code wraps model/optimizer prep in `accelerator.prepare()`.
*   **Safe Module Unwrapping**: Uses `accelerator.unwrap_model(model)` before saving to ensure weights are clean of distributed wrappers.
*   **Quantization Safety**: `MultimodalAlignmentModel.to()` method specifically overrides standard PyTorch behavior. It *skips* moving the Decoder if 4-bit/8-bit quantization is detected, as bitsandbytes modules cannot be moved between devices trivially.
*   **Text Encoder Context**: In `src/encoders/text.py`, `SentenceTransformer` inference is wrapped in `torch.no_grad()` by the library. The code explicitly clones the resulting tensor to re-enable autograd for the projector layers: `pooled_base = pooled_base.clone()`.

---

## 5. Metrics & Evaluation
*   **Retrieval (Alignment)**:
    *   Computes cosine similarity matrix between all validation pairs.
    *   Calculates **Recall@1, 5, 10** for both Image-to-Text and Text-to-Image.
*   **Generation (VLM)**:
    *   **Perplexity**: `exp(loss)`.
    *   **BLEU/ROUGE**: Implemented in `qa_metrics.py` (dependency-free versions).
    *   **Exact Match (EM)**: For strict QA evaluation.
