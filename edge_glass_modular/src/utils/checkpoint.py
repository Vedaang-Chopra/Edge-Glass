"""Checkpointing utilities with crash-safe semantics."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict

import torch


def save_checkpoint(state: Dict[str, Any], ckpt_dir: str, step: int) -> Path:
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = ckpt_dir / f"tmp_step_{step}.pt"
    path = ckpt_dir / f"step_{step}.pt"
    torch.save(state, tmp_path)
    shutil.move(tmp_path, path)
    return path


def latest_checkpoint(ckpt_dir: str) -> Path | None:
    ckpt_dir = Path(ckpt_dir)
    if not ckpt_dir.exists():
        return None
    checkpoints = sorted(ckpt_dir.glob("step_*.pt"))
    return checkpoints[-1] if checkpoints else None


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu"):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return torch.load(path, map_location=map_location, weights_only=False)
