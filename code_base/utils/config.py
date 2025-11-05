from dataclasses import dataclass
from typing import Dict, Any
import yaml, os, random
import numpy as np
import torch

@dataclass
class Paths:
    data: str
    out: str
    embeds: str

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

def _to_config(d: Dict[str, Any]) -> Config:
    return Config(
        device=d.get("device", "cuda"),
        dtype=d.get("dtype", "fp16"),
        seeds=Seeds(**d["seeds"]),
        paths=Paths(**d["paths"]),
        encoders=Encoders(**d["encoders"]),
    )

def load_config(path: str = "configs/base.yaml") -> Config:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    cfg = _to_config(raw)
    _apply_seeds(cfg)
    return cfg

def _apply_seeds(cfg: Config):
    py = cfg.seeds.python
    torch_seed = cfg.seeds.torch
    random.seed(py)
    np.random.seed(py)
    torch.manual_seed(torch_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(torch_seed)

def select_device(cfg: Config) -> torch.device:
    if cfg.device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def select_dtype(cfg: Config):
    # Map string to torch dtype (keep it simple for now)
    return {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }.get(cfg.dtype, torch.float32)
