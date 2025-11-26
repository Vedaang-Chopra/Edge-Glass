"""
Multimodal Alignment Package

A clean, modular implementation for vision-text alignment and LLM integration.

Modules:
    - core: Core components (encoders, adapters, losses)
    - data: Dataset and DataLoader utilities
    - train: Training and evaluation functions
    - llm_integration: Phase 2 LLM connection
"""

from .core import (
    AlignmentConfig,
    VisionEncoder,
    TextEncoder,
    MLPAdapter,
    VisionTextAligner,
    clip_contrastive_loss,
    matryoshka_loss,
    compute_retrieval_metrics,
    get_device,
    set_seed,
    count_parameters,
)

from .data import (
    FeatureDataset,
    ImageTextDataset,
    SimpleImageTextDataset,
    create_dataloader,
    collate_features,
    collate_images,
)

from .train import (
    build_optimizer,
    build_scheduler,
    train_one_epoch,
    evaluate,
    train_alignment,
    save_checkpoint,
    load_checkpoint,
)

from .llm_integration import (
    LLMConfig,
    LLMDecoder,
    VisionToLLMProjector,
    MultimodalLLM,
)

__version__ = "0.1.0"
__all__ = [
    # Core
    "AlignmentConfig",
    "VisionEncoder",
    "TextEncoder", 
    "MLPAdapter",
    "VisionTextAligner",
    "clip_contrastive_loss",
    "matryoshka_loss",
    "compute_retrieval_metrics",
    "get_device",
    "set_seed",
    "count_parameters",
    # Data
    "FeatureDataset",
    "ImageTextDataset",
    "SimpleImageTextDataset",
    "create_dataloader",
    "collate_features",
    "collate_images",
    # Training
    "build_optimizer",
    "build_scheduler",
    "train_one_epoch",
    "evaluate",
    "train_alignment",
    "save_checkpoint",
    "load_checkpoint",
    # LLM
    "LLMConfig",
    "LLMDecoder",
    "VisionToLLMProjector",
    "MultimodalLLM",
]
