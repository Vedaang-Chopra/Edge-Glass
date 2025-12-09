# Edge Glass Modular: Advanced VLM Agent Framework

**Production-ready, modular PyTorch foundation for multimodal agent training.**

This framework provides a complete infrastructure for training Vision-Language Models (VLMs) and Multimodal Agents on large-scale GPU clusters. It merges research flexibility with production reliability, featuring Distributed Data Parallel (DDP) training, crash-safe checkpointing, modular encoder-decoder architectures, and advanced representation learning techniques like MRL.

---

## 🚀 Key Features

### 🏗 Architecture
- **Dual Decoders**: Support for **Qwen (7B/14B)** via LoRA/8-bit quantization and a custom lightweight **TRM (Tiny Recursive Model)**.
- **Multimodal Encoders**: 
  - **Vision**: CLIP with learnable **Attention Pooling** and **Matryoshka Representation Learning (MRL)** up to 4096 dimensions.
  - **Audio**: Whisper encoder.
  - **Text**: Sentence-BERT.
- **Efficient Compression**: **Perceiver Resampler** for variable-length vision/audio sequence compression.

### ⚙️ Production Training
- **Robust Infrastructure**: 
  - **DDP** Support (Distributed Data Parallel).
  - **Mixed Precision** (BF16/FP16).
  - **Gradient Accumulation** & Clipping.
- **Resilience**: Crash-safe checkpointing (auto-resume from latest step) and best-model tracking.
- **Observability**: Integrated **WandB** logging, automated training curve plotting, and embedding space visualization.

### 🧠 Advanced Capabilities
- **Matryoshka Representation Learning (MRL)**: Trains embeddings to be useful at multiple scales (128 to 4096 dim) simultaneously.
- **Attention Pooling**: Replaces static pooling with learnable query-based attention for better feature extraction.
- **Text Dropout**: Regularization technique to force stronger reliance on visual modalities.

---

## 📂 Project Structure

```bash
edge_glass_modular/
├── configs/                  # Experiment configurations (YAML)
│   ├── pixmo_alignment.yaml  # Vision-Text alignment
│   ├── trimodal_qwen.yaml    # V+A+T with Qwen
│   ├── trimodal_trm.yaml     # V+A+T with TRM
│   └── ...
├── src/                      # Source code
│   ├── data/                 # Data loading & processing
│   │   ├── dataset_builder.py # Parquet dataset & Transforms
│   │   └── downloader.py     # Parallel downloaders
│   ├── encoders/             # Vision, Audio, Text encoders
│   ├── decoders/             # Qwen & TRM definitions
│   ├── models/               # Fusion & Alignment logic
│   ├── training/             # Trainer, Callbacks, Checkpointing
│   └── utils/                # Logging, Viz, Registry
├── notebooks/                # Interactive tutorials
│   ├── 02_pixmo_vision_text_alignment.ipynb
│   └── 03_trm_vlm_qa_training_FIXED.ipynb
├── scripts/                  # Entry points
│   ├── train_alignment.py    # Main training script
│   └── download_datasets.py  # Data preparation
└── README.md                 # This file
```

---

## 🛠 Installation

1.  **Prerequisites**:
    *   Python 3.10+
    *   NVIDIA GPU (Recommended: A100/H100 for Qwen runs, modest GPUs for TRM).

2.  **Install Package**:
    ```bash
    cd edge_glass_modular
    pip install -e .
    ```

---

## 🚦 Quick Start

### 1. Download Data
Download and prepare the PixMo (or other) datasets.
```bash
# Download PixMo-Cap (example: 20k samples)
python scripts/download_pixmo_audio.py --dataset pixmo --num_samples 20000
```

### 2. Run Training (CLI)
Train using pre-defined configurations for different experiments.

**Vision-Text Alignment (Single GPU)**:
```bash
python scripts/train_alignment.py --config configs/pixmo_alignment.yaml
```

**Tri-Modal (Vision+Audio+Text) with Qwen (2 GPUs)**:
```bash
torchrun --nproc_per_node=2 scripts/train_alignment.py \
    --config configs/trimodal_qwen.yaml
```

**TRM Decoder (Lightweight)**:
```bash
# Faster training, lower memory
python scripts/train_alignment.py --config configs/trimodal_trm.yaml
```

### 3. Run Notebooks
For interactive debugging, exploration, or debugging specific components:

- **`notebooks/02_pixmo_vision_text_alignment.ipynb`**: End-to-end guide for Stage 1 alignment training. Visualizes embedding spaces and loss curves.
- **`notebooks/03_trm_vlm_qa_training_FIXED.ipynb`**: Stage 2 QA training with the TRM decoder. Includes the **Label Masking** fix for robust generation.

---

## 🔧 Configuration

Experiments are controlled via YAML files in `configs/`.

**Example: `configs/pixmo_alignment.yaml`**
```yaml
dataset:
  train_parquet: path/to/train.parquet
  batch_size: 64
  text_dropout_prob: 0.1  # Regularization

vision_encoder:
  projection_dim: 4096    # MRL Top Dimension
  use_attention_pooling: true  # Better than CLS token

losses:
  contrastive: 0.25       # CLIP Loss weight
  mrl: 1.0                # MRL Loss weight
  sample_single_mrl_dim: true # Efficient training

optimization:
  lr: 0.0002
  warmup_ratio: 0.1
  max_grad_norm: 1.0
```

---

## 📊 Outputs & Monitoring

**Visualizations**: `outputs/<experiment_name>/`
- `training_curves.png`: Loss progression.
- `loss_components.png`: CLIP vs MRL contribution.
- `embedding_space.png`: PCA/t-SNE of vision-text alignment.
- `metrics.csv`: Tabular history.

**Checkpoints**: `checkpoints/<experiment_name>/`
- `checkpoint_best.pt`: Model with lowest validation loss.
- `checkpoint_latest.pt`: Autosave for crash recovery.

**WandB**: Set `use_wandb: true` in config or passed to trainer to sync runs to Weights & Biases.

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| **OOM (Out Of Memory)** | Reduce `batch_size`, enable `gradient_checkpointing`, or freeze encoder layers. |
| **0.0000 Validation Loss** | Known issue with `None` loss handling. **Fixed** in `ImprovedMultimodalTrainer` (Stage 1). Ensure you are using the latest code. |
| **Empty/Jibberish Generation** | Caused by incorrect label masking. **Fixed** in `03_trm_vlm_qa_training_FIXED.ipynb`. Ensure `answer_mask` is used. |
| **Training Not Resuming** | Check if `checkpoint_latest.pt` exists in the experiment directory. |

---

## 📅 Architecture History
- **v1**: Basic research notebooks.
- **v2**: Monolithic script.
- **v3**: Initial modularization.
- **Current (Edge Glass Modular)**: Fully merged, production-grade package with all capabilities from v1-v3 combined.

**Status**: Ready to train on 2×H200 GPUs.
