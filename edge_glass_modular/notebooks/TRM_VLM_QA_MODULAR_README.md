# TRM VLM QA Training - Modular Version

## Overview

This document describes the modular refactoring of the TRM VLM QA training notebook. The new version (`03_trm_vlm_qa_training_MODULAR.ipynb`) uses proper abstractions and imports from the `edge_glass_modular/src` directory.

## Key Changes

### 1. **Modular Dataset Class**
- **File**: `edge_glass_modular/src/data/dataset_builder.py`
- **Class**: `PixmoQADataset`
- Loads from parquet files with real QA pairs (question, answer, image_bytes)
- Properly tokenizes questions and answers separately
- Returns attention masks for proper padding

### 2. **Modular Vision Encoder**
- **File**: `edge_glass_modular/src/encoders/vision.py`
- **Class**: `VisionEncoder`
- Supports CLIP + Perceiver + MRL
- Can load from pretrained aligned model checkpoints
- Configurable freeze/unfreeze

### 3. **Modular Decoder**
- **File**: `edge_glass_modular/src/decoders/qwen.py`
- **Class**: `QwenDecoder`
- Qwen2.5 model with LoRA support
- 8-bit/4-bit quantization options
- Supports multimodal prefix embeddings

### 4. **Configuration-Driven**
- **File**: `edge_glass_modular/configs/trm_vlm_qa.yaml`
- All hyperparameters in YAML config
- Easy to modify without changing code
- Supports different decoder types (qwen, trm)

## Dataset Structure

The PixMo QA dataset has the following structure:

```
pixmo_qa_mixed_with_bytes.parquet
├── image_url: str
├── image_sha256: str
├── image_bytes: bytes  # Raw image bytes
├── question: str       # Question text (includes [USER]/[ASSISTANT] tags)
├── answer: str         # Answer text
└── source: str         # Source identifier (e.g., "pixmo-cap-qa")
```

**Example**:
- **Question**: `[USER]Why is John Travolta wearing such an obviously fake mustache in this photo?...`
- **Answer**: `This image is likely from a movie or promotional event. The exaggerated fake mustache...`

## Architecture

```
Input:
  Image (B, 3, 336, 336)
  Question tokens (B, L_q)
  Answer tokens (B, L_a)

Pipeline:
  1. Vision Encoder (frozen)
     └─> Vision tokens (B, num_latents, vision_dim)

  2. Vision Projection (trainable)
     └─> Projected vision (B, num_latents, qwen_dim)

  3. Qwen Decoder with LoRA (trainable)
     Input layout: [VISION_PREFIX] [QUESTION] [ANSWER]
     Labels: [-100 for vision] [-100 for question] [answer_ids]
     └─> Loss computed only on answer tokens
```

## Key Files

1. **Notebook**: `03_trm_vlm_qa_training_MODULAR.ipynb`
   - Clean, modular training pipeline
   - Uses imports from `src/`
   - Configuration-driven

2. **Config**: `configs/trm_vlm_qa.yaml`
   - Dataset paths
   - Model hyperparameters
   - Training settings

3. **Dataset Class**: `src/data/dataset_builder.py::PixmoQADataset`
   - Handles QA parquet loading
   - Tokenization with proper masks
   - Image preprocessing

4. **Vision Encoder**: `src/encoders/vision.py::VisionEncoder`
   - CLIP + Perceiver + MRL
   - Frozen or trainable
   - Returns sequence embeddings

5. **Decoder**: `src/decoders/qwen.py::QwenDecoder`
   - Qwen2.5 with LoRA
   - Supports vision prefix
   - Efficient 8-bit quantization

## Configuration

### Dataset Paths

Update the config file with your dataset paths:

```yaml
# configs/trm_vlm_qa.yaml
dataset:
  train_parquet: /path/to/pixmo_qa_mixed_with_bytes.parquet
  val_parquet: /path/to/val.parquet  # TODO: Add validation set
  test_parquet: /path/to/test.parquet  # TODO: Add test set
```

### Model Configuration

```yaml
decoder:
  type: "qwen"  # or "trm" for recursive decoder
  model_name: "Qwen/Qwen2.5-7B-Instruct"
  load_in_8bit: true
  use_lora: true
  lora_r: 32
  lora_alpha: 64
```

### Training Configuration

```yaml
training:
  num_epochs: 3
  learning_rate: 2.0e-4
  warmup_ratio: 0.1
  max_grad_norm: 1.0
```

## Usage

### 1. Update Config

Edit `configs/trm_vlm_qa.yaml` with your dataset paths:

```yaml
dataset:
  train_parquet: /home/hice1/vchopra37/scratch/projects/edge_glass/dataset/final_dataset/pimo-alignment/pixmo_qa_mixed_with_bytes.parquet
  val_parquet: null  # TODO: Add validation parquet
  test_parquet: null  # TODO: Add test parquet
```

### 2. Run Notebook

Open `03_trm_vlm_qa_training_MODULAR.ipynb` and run all cells.

The notebook will:
1. Load configuration from YAML
2. Initialize vision encoder (from checkpoint or fresh)
3. Initialize Qwen decoder with LoRA
4. Create projection layer
5. Load dataset
6. Train the model
7. Save checkpoints

### 3. Monitor Training

If W&B is enabled (default), training metrics will be logged to:
- Project: `edge_glass_trm_vlm`
- Run name: `trm_vlm_qa_{model_name}`

## Differences from Original Notebook

| Aspect | Original | Modular Version |
|--------|----------|-----------------|
| Dataset | Custom inline class | `PixmoQADataset` from `src/` |
| Vision Encoder | Inline model loading | `VisionEncoder` class |
| Decoder | Custom TRM decoder | `QwenDecoder` with LoRA |
| Config | Hardcoded values | YAML config file |
| Imports | Mixed inline/imports | All from `src/` modules |
| Dataset source | Synthetic QA | Real PixMo QA pairs |

## TODO / Placeholders

The notebook has placeholders for validation and test datasets:

```yaml
val_parquet: null  # TODO: Add validation parquet path
test_parquet: null  # TODO: Add test parquet path
```

Once you have validation/test files, update the config and uncomment the validation loop in the notebook.

## Advantages of Modular Design

1. **Reusability**: Components can be used in other notebooks/scripts
2. **Maintainability**: Changes to encoder/decoder logic in one place
3. **Testability**: Each component can be tested independently
4. **Configurability**: Easy to swap models via config
5. **Best Practices**: Follows software engineering patterns

## Example Output

```
Loaded config: trm_vlm_qa

Dataset:
  Train parquet: .../pixmo_qa_mixed_with_bytes.parquet
  Image size: 336
  Max question length: 128
  Max answer length: 256
  Batch size: 16

Decoder:
  Type: qwen
  Model: Qwen/Qwen2.5-7B-Instruct
  Use LoRA: True
  LoRA rank: 32

Training Setup:
  Num epochs: 3
  Learning rate: 2e-4
  Total steps: 2250
  Warmup steps: 225
  Trainable parameters: 42,893,312

Training...
Epoch 1/3: 100%|████████| 750/750 [12:34<00:00, loss=2.1234, lr=1.5e-4]
✓ Saved best checkpoint (loss: 2.1234)
```

## Next Steps

1. **Add Validation Set**: Create validation parquet and update config
2. **Add Test Set**: Create test parquet and update config
3. **Evaluation Metrics**: Add BLEU, ROUGE, METEOR for QA evaluation
4. **Generation Strategies**: Experiment with beam search, top-k sampling
5. **Vision Encoder Experiments**: Try different pretrained models
6. **Decoder Experiments**: Try Llama, Mistral, etc.

## Support

For issues or questions:
1. Check the configuration in `configs/trm_vlm_qa.yaml`
2. Verify dataset paths exist
3. Check GPU memory if OOM errors occur (reduce batch size)
4. Review the modular components in `src/`

## Summary

This refactoring provides a clean, modular, and maintainable codebase for VLM training. All components follow the established patterns in `edge_glass_modular` and can be easily extended or modified.
