# config.py

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Any, Dict

import os
import random

import numpy as np
import torch
import yaml  # pip install pyyaml if needed


# ==========================
# Dataclasses
# ==========================


@dataclass
class DatasetsConfig:
    # Paths to your pre-built feature index files
    pixmo_train_index: str
    pixmo_val_index: str
    librispeech_train_index: Optional[str] = None

    # Flags in case you want to disable a modality
    use_pixmo: bool = True
    use_librispeech: bool = True

    def resolve(self) -> None:
        """Normalize paths to absolute paths."""
        self.pixmo_train_index = str(Path(self.pixmo_train_index).expanduser().resolve())
        self.pixmo_val_index = str(Path(self.pixmo_val_index).expanduser().resolve())
        if self.librispeech_train_index is not None:
            self.librispeech_train_index = str(
                Path(self.librispeech_train_index).expanduser().resolve()
            )

    @property
    def pixmo_train_path(self) -> Path:
        return Path(self.pixmo_train_index)

    @property
    def pixmo_val_path(self) -> Path:
        return Path(self.pixmo_val_index)

    @property
    def librispeech_train_path(self) -> Optional[Path]:
        return Path(self.librispeech_train_index) if self.librispeech_train_index else None

@dataclass
class PathsConfig:
    root_dir: str
    features_dir: Optional[str] = None

    def resolve(self) -> None:
        """Normalize to absolute paths and derive defaults."""
        root = Path(self.root_dir).expanduser().resolve()
        self.root_dir = str(root)

        if self.features_dir is None:
            self.features_dir = str(root / "features")
        else:
            self.features_dir = str(Path(self.features_dir).expanduser().resolve())

    @property
    def root_path(self) -> Path:
        return Path(self.root_dir)

    @property
    def features_path(self) -> Path:
        assert self.features_dir is not None
        return Path(self.features_dir)


@dataclass
class ModelsConfig:
    vision_model_name: str
    llm_model_name: str
    audio_model_name: Optional[str] = None


@dataclass
class ArchitectureConfig:
    perceiver_dim: Optional[int] = None   # will often be set later from encoder
    num_latents: int = 64
    num_perceiver_layers: int = 4
    num_attn_heads: int = 8
    mlp_ratio: float = 4.0


@dataclass
class TrainingConfig:
    batch_size: int = 16
    num_epochs: int = 5
    learning_rate: float = 3e-4
    weight_decay: float = 0.01

    warmup_steps: int = 500
    max_grad_norm: float = 1.0

    log_every_steps: int = 50

    train_subset_size: int = 2000
    val_subset_size: int = 500


@dataclass
class MRLConfig:
    mrl_dims: Optional[List[int]] = None
    mrl_weight: float = 1.0
    mrl_temp: float = 0.07


@dataclass
class MiscConfig:
    dtype: str = "float16"   # "float16" or "bfloat16"
    seed: int = 42

    device: str = "auto"     # "auto", "cuda", "cpu"

    use_wandb: bool = False
    wandb_project: str = "edge_glass"
    wandb_run_name: str = "debug_run"


@dataclass
class Config:
    paths: PathsConfig
    models: ModelsConfig
    architecture: ArchitectureConfig
    training: TrainingConfig
    mrl: MRLConfig
    misc: MiscConfig
    datasets: DatasetsConfig  # ⬅️ add this line


    # Derived fields (not from YAML)
    torch_device: torch.device = torch.device("cpu")
    torch_dtype: torch.dtype = torch.float16

    def resolve_device_and_dtype(self) -> None:
        # device
        if self.misc.device == "auto":
            if torch.cuda.is_available():
                self.torch_device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                # Just in case you run on Mac
                self.torch_device = torch.device("mps")
            else:
                self.torch_device = torch.device("cpu")
        else:
            self.torch_device = torch.device(self.misc.device)

        # dtype
        if self.misc.dtype.lower() == "float16":
            self.torch_dtype = torch.float16
        elif self.misc.dtype.lower() == "bfloat16":
            self.torch_dtype = torch.bfloat16
        else:
            raise ValueError(f"Unsupported dtype in config.misc.dtype: {self.misc.dtype}")


# ==========================
# YAML Loader
# ==========================

def _get_section(d: Dict[str, Any], key: str, default: Dict[str, Any]) -> Dict[str, Any]:
    """Helper: get section dict or default."""
    return d.get(key, {}) or default


def load_config(yaml_path: str = "config.yaml") -> Config:
    """
    Load config.yaml and return a fully-initialized Config instance.
    """
    path = Path(yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {yaml_path}")

    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}

    # Extract subsections with sane defaults
    paths_raw = _get_section(raw, "paths", {"root_dir": "./runs", "features_dir": None})
    models_raw = _get_section(raw, "models", {})
    datasets_raw = _get_section(raw, "datasets", {})  # ⬅️ new
    arch_raw = _get_section(raw, "architecture", {})
    training_raw = _get_section(raw, "training", {})
    mrl_raw = _get_section(raw, "mrl", {})
    misc_raw = _get_section(raw, "misc", {})


    paths = PathsConfig(**paths_raw)
    models = ModelsConfig(**models_raw)
    datasets = DatasetsConfig(**datasets_raw)   # ⬅️ new
    arch = ArchitectureConfig(**arch_raw)
    training = TrainingConfig(**training_raw)
    mrl = MRLConfig(**mrl_raw)
    misc = MiscConfig(**misc_raw)

    # Resolve paths, device, dtype, etc.
    paths.resolve()

    cfg = Config(
        paths=paths,
        models=models,
        architecture=arch,
        training=training,
        mrl=mrl,
        misc=misc,
        datasets=datasets
    )
    cfg.resolve_device_and_dtype()

    return cfg


# ==========================
# Seed + Directory Setup
# ==========================

def set_global_seed(seed: int) -> None:
    """
    Set seeds for reproducibility across random, numpy, and torch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Optional: make CUDA deterministic (can slow things)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_directories(cfg: Config) -> None:
    """
    Create root_dir, features_dir, and any other top-level dirs.
    """
    cfg.paths.root_path.mkdir(parents=True, exist_ok=True)
    cfg.paths.features_path.mkdir(parents=True, exist_ok=True)


# ==========================
# Weights & Biases Setup
# ==========================

def init_wandb(cfg: Config, extra_config: Optional[Dict[str, Any]] = None):
    """
    Initialize Weights & Biases using the config dataclass.
    You can pass extra_config to add more fields (e.g., model dim discovered later).
    """
    if not cfg.misc.use_wandb:
        return None

    try:
        import wandb
    except ImportError:
        print("[WARN] wandb not installed, skipping W&B init.")
        return None

    # Convert dataclasses to plain dict for W&B
    base_cfg = {
        "paths": asdict(cfg.paths),
        "models": asdict(cfg.models),
        "datasets": asdict(cfg.datasets),
        "architecture": asdict(cfg.architecture),
        "training": asdict(cfg.training),
        "mrl": asdict(cfg.mrl),
        "misc": asdict(cfg.misc),
    }

    if extra_config:
        base_cfg.update(extra_config)

    run = wandb.init(
        project=cfg.misc.wandb_project,
        name=cfg.misc.wandb_run_name,
        config=base_cfg,
    )
    return run


# ==========================
# Convenience one-liner
# ==========================

def setup_from_yaml(yaml_path: str = "config.yaml") -> Config:
    """
    Full Phase-0 setup:
      - load YAML
      - resolve paths / device / dtype
      - set seeds
      - create dirs
      - (optional) init W&B

    Returns:
        Config object (you can pass cfg.torch_device, cfg.torch_dtype downstream)
    """
    cfg = load_config(yaml_path=yaml_path)
    set_global_seed(cfg.misc.seed)
    create_directories(cfg)
    init_wandb(cfg)

    print(f"[Config] Device: {cfg.torch_device}, dtype: {cfg.torch_dtype}")
    print(f"[Config] root_dir: {cfg.paths.root_dir}")
    print(f"[Config] features_dir: {cfg.paths.features_dir}")

    return cfg


