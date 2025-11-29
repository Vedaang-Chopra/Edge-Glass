"""Training callbacks."""

from __future__ import annotations

from dataclasses import dataclass

from ..utils import save_checkpoint


@dataclass
class CheckpointCallback:
    ckpt_dir: str
    save_every: int

    def __call__(self, state, step: int):
        if step % self.save_every == 0:
            save_checkpoint(state, self.ckpt_dir, step)
