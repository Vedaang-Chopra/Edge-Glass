"\"\"\"Distributed helpers for torchrun/DDP.\"\"\""

from __future__ import annotations

import os
from contextlib import contextmanager

import torch
import torch.distributed as dist


def init_distributed():
    if dist.is_available() and dist.is_initialized():
        return
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")


def get_rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


@contextmanager
def local_seed(seed: int):
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    cpu_state = torch.random.get_rng_state()
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    try:
        yield
    finally:
        torch.random.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
