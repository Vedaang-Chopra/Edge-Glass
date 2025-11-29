#!/usr/bin/env python
"""Instruction tuning launcher across modalities."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src import load_config
from src.models import MultimodalAlignmentModel
from src.training import MultimodalTrainer


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    model = MultimodalAlignmentModel(cfg)
    trainer = MultimodalTrainer(cfg, model)
    # Instruction dataloaders are configured inside notebooks with dataset builder.
    trainer.logger.info("Instruction tuning setup ready. Launch via notebooks.")


if __name__ == "__main__":
    main()
