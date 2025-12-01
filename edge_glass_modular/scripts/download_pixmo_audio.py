#!/usr/bin/env python
"""Utility to download PixMo-Cap images and Common Voice audio subsets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.data.downloader import (
    build_instruction_corpus,
    download_common_voice_subset,
    download_pixmo_subset,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pixmo-out", type=str, required=True)
    parser.add_argument("--audio-out", type=str, required=True)
    parser.add_argument("--instruction-out", type=str, default="./checkpoints/datasets/instruction")
    parser.add_argument("--samples", type=int, default=20000)
    parser.add_argument("--workers", type=int, default=64)
    return parser.parse_args()


def main():
    args = parse_args()
    pixmo_meta = download_pixmo_subset(args.pixmo_out, samples=args.samples, workers=args.workers)
    download_common_voice_subset(args.audio_out, samples=args.samples, workers=args.workers)
    build_instruction_corpus(args.instruction_out, pixmo_meta)


if __name__ == "__main__":
    main()
