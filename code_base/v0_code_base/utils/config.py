from dataclasses import dataclass
from typing import Dict, Any
from pathlib import Path
import yaml, os, random
import numpy as np
import torch

# code_base/ path (this file lives in code_base/utils/)
CODEBASE_DIR = Path(__file__).resolve().parents[1]

@dataclass
class Paths:
    data: str
    out: str
    embeds: str

    def resolve(self) -> Dict[str, Path]:
        """Return absolute paths resolved from code_base/"""
        return {
            "data": (CODEBASE_DIR / self.data).resolve(),
            "out": (CODEBASE_DIR / self.out).resolve(),
            "embeds": (CODEBASE_DIR / self.embeds).resolve(),
        }

@dataclass
class Seeds:
    python: int
    torch: int

@dataclass
class Encoders:
    vision: str
    text: str
    audio: str

@dataclass
class Config:
    device: str
    dtype: str
    seeds: Seeds
    paths: Paths
    encoders: Encoders
    cfg_path: Path

def _to_config(d: Dict[str, Any], cfg_path: Path) -> Config:
    return Config(
        device=d.get("device", "cuda"),
        dtype=d.get("dtype", "fp16"),
        seeds=Seeds(**d["seeds"]),
        paths=Paths(**d["paths"]),
        encoders=Encoders(**d["encoders"]),
        cfg_path=cfg_path,
    )

def load_config(path: str = None) -> Config:
    """If path is None, default to code_base/configs/base.yaml"""
    cfg_path = Path(path) if path else (CODEBASE_DIR / "configs" / "base.yaml")
    with open(cfg_path, "r") as f:
        raw = yaml.safe_load(f)
    cfg = _to_config(raw, cfg_path)
    _apply_seeds(cfg)
    _ensure_dirs(cfg)
    return cfg

def _apply_seeds(cfg: Config):
    py = cfg.seeds.python
    torch_seed = cfg.seeds.torch
    random.seed(py)
    np.random.seed(py)
    torch.manual_seed(torch_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(torch_seed)

def _ensure_dirs(cfg: Config):
    paths = cfg.paths.resolve()
    for key in ("out", "embeds", "data"):
        paths[key].mkdir(parents=True, exist_ok=True)

def select_device(cfg: Config) -> torch.device:
    if cfg.device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def select_dtype(cfg: Config):
    return {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }.get(cfg.dtype, torch.float32)
