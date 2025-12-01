"""Configuration system for Edge Glass experiments."""

from dataclasses import dataclass, field
from typing import List, Optional, Literal
import yaml
from pathlib import Path


@dataclass
class EncoderConfig:
    """Configuration for a single encoder (vision/audio/text)."""

    model_name: str
    projection_dim: int = 1024
    freeze: bool = True

    # Perceiver options
    use_perceiver: bool = False
    perceiver_num_latents: int = 64
    perceiver_latent_dim: int = 512
    perceiver_num_layers: int = 3
    perceiver_num_heads: int = 8

    # MRL options
    use_mrl: bool = False
    mrl_dimensions: List[int] = field(default_factory=lambda: [512, 256, 128])
    mrl_loss_weight: float = 0.05


@dataclass
class DecoderConfig:
    """Configuration for decoder LLM."""

    type: Literal["qwen", "trm"] = "qwen"
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    freeze: bool = False

    # Quantization
    load_in_8bit: bool = True
    load_in_4bit: bool = False

    # LoRA configuration
    use_lora: bool = True
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    # TRM specific
    trm_vocab_size: int = 32000
    trm_hidden_dim: int = 512
    trm_num_layers: int = 6
    trm_num_heads: int = 8
    trm_max_seq_len: int = 2048


@dataclass
class FusionConfig:
    """Configuration for multimodal fusion."""

    strategy: Literal["concat", "cross_attention", "gated"] = "concat"
    fusion_dim: int = 2048
    num_fusion_layers: int = 2
    num_heads: int = 8
    dropout: float = 0.1


@dataclass
class DatasetConfig:
    """Configuration for datasets."""

    # Paths
    data_dir: str = "./data"
    cache_dir: str = "./cache"

    # Dataset selection
    use_vision: bool = True
    use_audio: bool = False
    use_text: bool = True

    # Dataset sizes
    num_train_samples: int = 20000
    num_val_samples: int = 2000

    # Data loading
    batch_size: int = 32
    num_workers: int = 8
    prefetch_factor: int = 2
    persistent_workers: bool = True
    pin_memory: bool = True

    # Image settings
    image_size: int = 224

    # Audio settings
    audio_sample_rate: int = 16000
    audio_max_duration: float = 10.0
    audio_num_mels: int = 128

    # Text settings
    max_text_length: int = 512

    # Instruction tuning
    instruction_dataset: str = "Open-Orca/OpenOrca"
    instruction_samples: int = 50000


@dataclass
class OptimizationConfig:
    """Optimization and training hyperparameters."""

    # Optimizer
    optimizer: Literal["adamw", "adam", "sgd"] = "adamw"
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8

    # Learning rate schedule
    lr_scheduler: Literal["cosine", "linear", "constant"] = "cosine"
    warmup_steps: int = 500
    warmup_ratio: float = 0.1

    # Gradient
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 1

    # Mixed precision
    mixed_precision: Literal["no", "fp16", "bf16"] = "bf16"

    # Loss weights
    contrastive_loss_weight: float = 1.0
    mrl_loss_weight: float = 0.05
    lm_loss_weight: float = 1.0


@dataclass
class TrainingConfig:
    """Training loop configuration."""

    # Training
    num_epochs: int = 3
    max_steps: Optional[int] = None
    eval_steps: int = 500
    save_steps: int = 1000
    logging_steps: int = 100

    # Checkpointing
    output_dir: str = "./checkpoints"
    save_total_limit: int = 3
    resume_from_checkpoint: Optional[str] = None

    # Distributed training
    local_rank: int = -1
    world_size: int = 1
    ddp_backend: str = "nccl"

    # Evaluation
    eval_strategy: Literal["steps", "epoch", "no"] = "steps"
    metric_for_best_model: str = "eval_loss"

    # Reproducibility
    seed: int = 42
    deterministic: bool = True

    # Logging
    log_level: str = "info"
    report_to: List[str] = field(default_factory=lambda: ["wandb"])
    wandb_project: str = "edge-glass"
    wandb_run_name: Optional[str] = None


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration."""

    # Experiment metadata
    name: str = "default"
    description: str = ""
    tags: List[str] = field(default_factory=list)

    # Model components
    vision_encoder: Optional[EncoderConfig] = None
    audio_encoder: Optional[EncoderConfig] = None
    text_encoder: Optional[EncoderConfig] = None
    decoder: Optional[DecoderConfig] = None
    fusion: Optional[FusionConfig] = None

    # Training configuration
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # Experiment type
    mode: Literal["alignment", "instruction_tuning"] = "alignment"
    use_instruction_tuning: bool = False

    def __post_init__(self):
        """Validate configuration."""
        # Ensure at least one encoder is configured
        encoders = [self.vision_encoder, self.audio_encoder, self.text_encoder]
        if not any(encoders):
            raise ValueError("At least one encoder must be configured")

        # Set wandb run name if not provided
        if self.training.wandb_run_name is None:
            self.training.wandb_run_name = self.name

    def save(self, path: str):
        """Save configuration to YAML file."""
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        def _convert(obj):
            if hasattr(obj, '__dataclass_fields__'):
                return {k: _convert(v) for k, v in obj.__dict__.items()}
            elif isinstance(obj, (list, tuple)):
                return [_convert(item) for item in obj]
            else:
                return obj
        return _convert(self)

    @classmethod
    def from_dict(cls, config_dict: dict) -> "ExperimentConfig":
        """Load configuration from dictionary."""
        # Parse nested configs
        if "vision_encoder" in config_dict and config_dict["vision_encoder"] is not None:
            config_dict["vision_encoder"] = EncoderConfig(**config_dict["vision_encoder"])
        if "audio_encoder" in config_dict and config_dict["audio_encoder"] is not None:
            config_dict["audio_encoder"] = EncoderConfig(**config_dict["audio_encoder"])
        if "text_encoder" in config_dict and config_dict["text_encoder"] is not None:
            config_dict["text_encoder"] = EncoderConfig(**config_dict["text_encoder"])
        if "decoder" in config_dict and config_dict["decoder"] is not None:
            config_dict["decoder"] = DecoderConfig(**config_dict["decoder"])
        if "fusion" in config_dict and config_dict["fusion"] is not None:
            config_dict["fusion"] = FusionConfig(**config_dict["fusion"])
        if "dataset" in config_dict:
            config_dict["dataset"] = DatasetConfig(**config_dict["dataset"])
        if "optimization" in config_dict:
            config_dict["optimization"] = OptimizationConfig(**config_dict["optimization"])
        if "training" in config_dict:
            config_dict["training"] = TrainingConfig(**config_dict["training"])

        return cls(**config_dict)

    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentConfig":
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)


def load_config(config_path: str) -> ExperimentConfig:
    """Load experiment configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        ExperimentConfig object
    """
    return ExperimentConfig.from_yaml(config_path)
