"""DDP-ready training orchestration."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader

from ..config import ExperimentConfig
from ..models import MultimodalAlignmentModel
from ..utils import get_rank, init_distributed, setup_logger
from .callbacks import CheckpointCallback


@dataclass
class TrainerState:
    global_step: int = 0
    epoch: int = 0


class MultimodalTrainer:
    def __init__(self, cfg: ExperimentConfig, model: MultimodalAlignmentModel):
        self.cfg = cfg
        self.logger = setup_logger(cfg.name)
        self.model = model
        self.device = torch.device("cuda", get_rank()) if torch.cuda.is_available() else torch.device("cpu")
        self.model.to(self.device)
        if cfg.trainer.strategy == "ddp" and torch.cuda.device_count() > 1:
            init_distributed()
            self.model = DDP(self.model, device_ids=[self.device.index])
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=cfg.optimization.lr,
            weight_decay=cfg.optimization.weight_decay,
        )
        self.scaler = torch.cuda.amp.GradScaler(enabled=cfg.optimization.fp16 or cfg.optimization.bf16)
        self.callbacks = [
            CheckpointCallback(cfg.trainer.ckpt_dir, cfg.trainer.save_every),
        ]
        self.state = TrainerState()

    def train(self, dataloader: DataLoader):
        self.model.train()
        for epoch in range(self.state.epoch, self.cfg.trainer.epochs):
            self.state.epoch = epoch
            for step, batch in enumerate(dataloader):
                batch = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in batch.items()}
                with torch.cuda.amp.autocast(enabled=self.cfg.optimization.fp16 or self.cfg.optimization.bf16):
                    outputs = self.model(batch)
                    loss = outputs["loss"] / self.cfg.optimization.grad_accum_steps
                self.scaler.scale(loss).backward()
                if (step + 1) % self.cfg.optimization.grad_accum_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.optimization.gradient_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    self.state.global_step += 1
                    if self.state.global_step % self.cfg.trainer.log_every == 0 and get_rank() == 0:
                        self.logger.info("step %s loss %.4f", self.state.global_step, float(loss))
                    model_to_save = self.model.module if isinstance(self.model, DDP) else self.model
                    for callback in self.callbacks:
                        callback(
                            {
                                "model": model_to_save.state_dict(),
                                "optimizer": self.optimizer.state_dict(),
                                "scaler": self.scaler.state_dict(),
                                "state": dataclasses.asdict(self.state),
                            },
                            self.state.global_step,
                        )
