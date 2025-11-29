"""Dataclass-backed experiment configuration utilities."""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class DatasetConfig:
    name: str
    batch_size: int = 32
    subset: Optional[str] = None
    cache_dir: str = "./checkpoints/datasets"
    image_root: Optional[str] = None
    audio_root: Optional[str] = None
    text_root: Optional[str] = None
    num_samples: Optional[int] = None
    num_workers: int = 8
    persistent_workers: bool = True
    pin_memory: bool = True


@dataclass
class EncoderConfig:
    model_name: str
    trainable: bool = False
    projection_dim: int = 1024
    use_perceiver: bool = False
    perceiver_dim: int = 512
    perceiver_layers: int = 2
    use_mrl: bool = False
    mrl_dimensions: Optional[List[int]] = None
    dropout: float = 0.1


@dataclass
class DecoderConfig:
    model_name: str
    type: str = "qwen"  # or "trm"
    max_seq_len: int = 1024
    use_lora: bool = True
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    freeze_backbone: bool = False


@dataclass
class OptimizationConfig:
    lr: float = 2e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    total_steps: int = 50000
    gradient_clip: float = 1.0
    grad_accum_steps: int = 1
    bf16: bool = True
    fp16: bool = False


@dataclass
class TrainerConfig:
    epochs: int = 3
    batch_size: int = 32
    eval_batch_size: int = 32
    log_every: int = 50
    save_every: int = 1000
    seed: int = 23
    devices: int = 2
    num_nodes: int = 1
    strategy: str = "ddp"
    ckpt_dir: str = "./checkpoints"
    resume_from: Optional[str] = None
    max_steps: Optional[int] = None


@dataclass
class ExperimentConfig:
    name: str
    dataset: DatasetConfig
    text_encoder: EncoderConfig
    vision_encoder: Optional[EncoderConfig] = None
    audio_encoder: Optional[EncoderConfig] = None
    decoder: Optional[DecoderConfig] = None
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    losses: Dict[str, Any] = field(default_factory=lambda: {"contrastive": 1.0})
    instruction_dataset: Optional[DatasetConfig] = None

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))


def _dataclass_from_dict(dc_cls, data: Dict[str, Any]):
    if data is None:
        return None
    field_names = {f.name for f in dataclasses.fields(dc_cls)}
    filtered = {k: v for k, v in data.items() if k in field_names}
    return dc_cls(**filtered)


def load_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    cfg_dict = yaml.safe_load(path.read_text())
    dataset = _dataclass_from_dict(DatasetConfig, cfg_dict.get("dataset"))
    text_encoder = _dataclass_from_dict(EncoderConfig, cfg_dict.get("text_encoder"))
    vision_encoder = _dataclass_from_dict(EncoderConfig, cfg_dict.get("vision_encoder"))
    audio_encoder = _dataclass_from_dict(EncoderConfig, cfg_dict.get("audio_encoder"))
    decoder = _dataclass_from_dict(DecoderConfig, cfg_dict.get("decoder"))
    optimization = _dataclass_from_dict(
        OptimizationConfig, cfg_dict.get("optimization", {})
    )
    trainer = _dataclass_from_dict(TrainerConfig, cfg_dict.get("trainer", {}))
    instruction_ds = _dataclass_from_dict(
        DatasetConfig, cfg_dict.get("instruction_dataset")
    )
    return ExperimentConfig(
        name=cfg_dict["name"],
        dataset=dataset,
        text_encoder=text_encoder,
        vision_encoder=vision_encoder,
        audio_encoder=audio_encoder,
        decoder=decoder,
        optimization=optimization,
        trainer=trainer,
        losses=cfg_dict.get("losses", {"contrastive": 1.0}),
        instruction_dataset=instruction_ds,
    )
