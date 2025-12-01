# Edge Glass Modular - Multimodal Alignment Framework

A production-ready, highly modular framework for multimodal alignment and instruction tuning.

## Features

- **Modular Architecture**: Clean separation of encoders, decoders, models, data, and training
- **Multi-GPU Training**: Built-in DDP support for 2×H200 GPUs
- **Crash Recovery**: Automatic checkpointing and resumption
- **Flexible Configuration**: YAML-based experiment configuration
- **Multiple Experiments**:
  - Vision-Text alignment with Qwen decoder
  - Tri-modal (vision-audio-text) with Qwen decoder
  - Tri-modal with TRM lightweight decoder
  - MRL ablation studies
  - Perceiver ablation studies

## Project Structure

```
edge_glass_modular/
├── src/                      # Source code
│   ├── encoders/            # Vision, Audio, Text encoders
│   ├── decoders/            # Qwen, TRM decoders
│   ├── models/              # Alignment models, fusion, projectors
│   ├── data/                # Dataset builders and downloaders
│   ├── training/            # Trainer, callbacks, metrics
│   └── utils/               # Checkpointing, logging, distributed
├── configs/                  # YAML configuration files
├── notebooks/               # Jupyter notebooks for experiments
├── scripts/                 # Training scripts
└── pyproject.toml          # Package configuration
```

## Installation

```bash
cd edge_glass_modular
pip install -e .
```

## Quick Start

### 1. Download Datasets

```bash
python scripts/download_datasets.py --num_samples 20000
```

### 2. Run Training

```bash
# Vision-Text alignment
torchrun --nproc_per_node=2 scripts/train.py --config configs/vision_text_qwen.yaml

# Tri-modal alignment
torchrun --nproc_per_node=2 scripts/train.py --config configs/trimodal_qwen.yaml
```

### 3. Use Notebooks

All notebooks are in `notebooks/` directory:
- `01_vision_text_alignment.ipynb` - Vision-text with Qwen
- `02_trimodal_alignment.ipynb` - Vision-audio-text with Qwen
- `03_trimodal_trm.ipynb` - Vision-audio-text with TRM
- `04_mrl_ablation.ipynb` - MRL ablation study

## Configuration

All experiments are configured via YAML files in `configs/`. Key parameters:

- **Encoders**: Vision (CLIP), Audio (Whisper), Text (Sentence-BERT)
- **Decoder**: Qwen or TRM
- **Training**: Learning rate, batch size, epochs, gradient accumulation
- **Ablations**: Perceiver, MRL, fusion strategies

## Citation

```bibtex
@software{edge_glass_modular,
  title = {Edge Glass Modular: Multimodal Alignment Framework},
  year = {2025}
}
```
