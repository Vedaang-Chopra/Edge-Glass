#!/usr/bin/env python
"""Launches Phase-1 multimodal alignment on 2×H200 via torchrun."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src import load_config
from src.data.datamodule import MultimodalDataModule
from src.data.transforms import build_transforms
from src.models import MultimodalAlignmentModel
from src.training import MultimodalTrainer


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    transforms = build_transforms()
    model = MultimodalAlignmentModel(cfg)
    datamodule = MultimodalDataModule(cfg.dataset, transforms)
    datamodule.setup()
    train_loader = datamodule.train_dataloader()
    trainer = MultimodalTrainer(cfg, model)
    trainer.train(train_loader)


if __name__ == "__main__":
    main()
