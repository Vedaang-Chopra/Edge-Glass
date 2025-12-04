"""Configuration system for Edge Glass experiments."""

from dataclasses import dataclass, field
from typing import List, Optional, Literal
import yaml
from pathlib import Path


@dataclass
class LossConfig:
    """Configuration for alignment loss weights and options."""

    contrastive: float = 1.0  # CLIP/main contrastive loss weight
    mrl: float = 1.0  # Matryoshka loss weight
    sample_single_mrl_dim: bool = True

    def __getitem__(self, key):
        return getattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)


@dataclass
class TrainerConfig:
    """Trainer configuration for checkpointing/logging orchestration."""

    epochs: int = 3
    num_epochs: Optional[int] = None  # Optional alias
    batch_size: Optional[int] = None  # Optional override
    save_every: int = 1
    log_every: int = 100
    ckpt_dir: str = "./checkpoints"
    output_dir: str = "./outputs"
    devices: int = 1
    strategy: str = "ddp"
    use_wandb: bool = True
    wandb_project: str = "edge_glass"
    wandb_run_name: Optional[str] = None
    retrieval_eval_samples: Optional[int] = None
    eval_batch_size: Optional[int] = None
    save_optimizer_state: bool = True  # Toggle optimizer/scheduler/scaler saving
    best_weights_only: bool = False  # Save only model weights for best checkpoint to reduce size

    def __getitem__(self, key):
        return getattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)


@dataclass
class EncoderConfig:
    """Configuration for a single encoder (vision/audio/text)."""

    model_name: str
    projection_dim: int = 1024
    freeze: bool = True
    trainable: bool = False

    # Perceiver options
    use_perceiver: bool = False
    perceiver_num_latents: int = 64
    perceiver_latent_dim: int = 512
    perceiver_num_layers: int = 3
    perceiver_num_heads: int = 8
    perceiver_dropout: float = 0.1

    # MRL options
    use_mrl: bool = False
    mrl_dimensions: List[int] = field(default_factory=lambda: [512, 256, 128])
    mrl_loss_weight: float = 0.05

    # Attention pooling options
    use_attention_pooling: bool = False
    pooling_type: str = "simple"  # "simple" or "multihead"


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

    # Dataset name
    name: str = "default"

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
    base_batch_size: int = 32
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
    text_dropout_prob: float = 0.0

    # Parquet file paths (for Pixmo and similar datasets)
    train_parquet: Optional[str] = None
    val_parquet: Optional[str] = None
    test_parquet: Optional[str] = None

    # Instruction tuning
    instruction_dataset: str = "Open-Orca/OpenOrca"
    instruction_samples: int = 50000


@dataclass
class OptimizationConfig:
    """Optimization and training hyperparameters."""

    # Optimizer
    optimizer: Literal["adamw", "adam", "sgd"] = "adamw"
    learning_rate: float = 2e-4
    lr: Optional[float] = None  # Alias for learning_rate
    weight_decay: float = 0.01
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8

    # Learning rate schedule
    lr_scheduler: Literal["cosine", "linear", "constant"] = "cosine"
    warmup_steps: Optional[int] = None
    warmup_ratio: float = 0.1
    total_steps: Optional[int] = None
    min_lr_ratio: float = 0.0

    # Gradient
    max_grad_norm: float = 1.0
    gradient_clip: Optional[float] = None  # Alias for max_grad_norm
    gradient_accumulation_steps: int = 1
    grad_accum_steps: Optional[int] = 1  # Alias for gradient_accumulation_steps

    # Mixed precision
    mixed_precision: Literal["no", "fp16", "bf16"] = "bf16"
    fp16: bool = False
    bf16: bool = False

    # Loss weights
    contrastive_loss_weight: float = 1.0
    mrl_loss_weight: float = 0.05
    lm_loss_weight: float = 1.0

    def __post_init__(self):
        # Keep lr and learning_rate in sync
        if self.lr is None:
            self.lr = self.learning_rate
        else:
            self.learning_rate = self.lr

        # Normalize gradient accumulation alias
        if self.grad_accum_steps is None:
            self.grad_accum_steps = self.gradient_accumulation_steps
        else:
            self.gradient_accumulation_steps = self.grad_accum_steps

        # Normalize gradient clipping alias
        if self.gradient_clip is None:
            self.gradient_clip = self.max_grad_norm
        else:
            self.max_grad_norm = self.gradient_clip


@dataclass
class TrainingConfig:
    """Training loop configuration."""

    # Training
    num_epochs: int = 3
    max_steps: Optional[int] = None
    eval_steps: int = 500
    save_steps: int = 1000
    logging_steps: int = 100
    warmup_steps: Optional[int] = None
    gradient_accumulation_steps: int = 1

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
    seed: int = 42

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

    # Loss configuration (dataclass with dict-like access)
    losses: LossConfig = field(default_factory=LossConfig)

    # Trainer configuration (dataclass with dict-like access)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)

    # Experiment type
    mode: Literal["alignment", "instruction_tuning"] = "alignment"
    use_instruction_tuning: bool = False

    def __post_init__(self):
        """Validate configuration."""
        # Normalize flexible sections first
        if isinstance(self.losses, dict) or self.losses is None:
            self.losses = LossConfig(**(self.losses or {}))
        if isinstance(self.trainer, dict) or self.trainer is None:
            self.trainer = TrainerConfig(**(self.trainer or {}))

        # Ensure at least one encoder is configured
        encoders = [self.vision_encoder, self.audio_encoder, self.text_encoder]
        if not any(encoders):
            raise ValueError("At least one encoder must be configured")

        # Set wandb run name if not provided
        if self.training.wandb_run_name is None:
            self.training.wandb_run_name = self.name
        if self.trainer.wandb_run_name is None:
            self.trainer.wandb_run_name = self.name
        self.training.seed = self.seed

        # Normalize trainer epochs alias
        if self.trainer.num_epochs is not None:
            self.trainer.epochs = self.trainer.num_epochs
        elif not self.trainer.epochs:
            self.trainer.epochs = self.training.num_epochs

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
        if "losses" in config_dict:
            losses_cfg = config_dict["losses"]
            config_dict["losses"] = LossConfig(**losses_cfg) if losses_cfg is not None else LossConfig()
        if "trainer" in config_dict:
            trainer_cfg = config_dict["trainer"]
            config_dict["trainer"] = TrainerConfig(**trainer_cfg) if trainer_cfg is not None else TrainerConfig()

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
