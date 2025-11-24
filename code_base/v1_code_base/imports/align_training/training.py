from typing import Callable, Dict, Any, Optional

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

from imports.align_training.steps import AlignmentModules, AlignmentConfig, forward_alignment_step


def build_alignment_optimizer(
    modules: AlignmentModules,
    learning_rate: float,
    weight_decay: float,
) -> AdamW:
    """Build AdamW optimizer over all parameters that require grad.

    We collect params from vision_adapter, audio_adapter, perceiver, and projector.
    Any of them can be None (e.g., if you are only training vision).
    """
    params = []
    for m in (modules.vision_adapter, modules.audio_adapter, modules.perceiver, modules.projector):
        if m is None:
            continue
        params.extend([p for p in m.parameters() if p.requires_grad])

    if not params:
        raise ValueError("No trainable parameters found for optimizer")

    return AdamW(params, lr=learning_rate, weight_decay=weight_decay)


def train_one_epoch(
    dataloader: DataLoader,
    modality: str,
    modules: AlignmentModules,
    cfg: AlignmentConfig,
    text_embed_fn: Callable[[list[str], int], torch.Tensor],
    optimizer: AdamW,
    device: torch.device,
    epoch: int = 0,
    log_every: int = 50,
    log_fn: Optional[Callable[[Dict[str, float]], None]] = None,
) -> float:
    """Train for one epoch on a single modality (vision or audio)."""

    # Put relevant modules in train mode
    if modality == "vision" and modules.vision_adapter is not None:
        modules.vision_adapter.train()
    if modality == "audio" and modules.audio_adapter is not None:
        modules.audio_adapter.train()
    if modules.perceiver is not None:
        modules.perceiver.train()
    modules.projector.train()

    running_loss = 0.0
    n_steps = 0

    for step, batch in enumerate(dataloader, start=1):
        optimizer.zero_grad(set_to_none=True)

        loss, metrics = forward_alignment_step(
            batch=batch,
            modality=modality,
            modules=modules,
            cfg=cfg,
            text_embed_fn=text_embed_fn,
            device=device,
        )

        loss.backward()
        # Gradient clipping (projector is usually the largest)
        torch.nn.utils.clip_grad_norm_(modules.projector.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item()
        n_steps += 1

        if step % log_every == 0:
            log_data: Dict[str, float] = {
                "epoch": float(epoch),
                "step": float(step),
                "loss": float(loss.item()),
            }
            log_data.update(metrics)
            if log_fn is not None:
                log_fn(log_data)
            else:
                print(
                    f"[Epoch {epoch} | {modality}] step {step:04d} | loss {loss.item():.4f}"
                )

    mean_loss = running_loss / max(n_steps, 1)
    return mean_loss


def train_alignment(
    train_loaders: Dict[str, DataLoader],
    modules: AlignmentModules,
    cfg: AlignmentConfig,
    text_embed_fn: Callable[[list[str], int], torch.Tensor],
    optimizer: AdamW,
    device: torch.device,
    num_epochs: int,
    log_every: int = 50,
    log_fn: Optional[Callable[[Dict[str, float]], None]] = None,
    modalities: tuple[str, ...] = ("vision", "audio"),
):
    """Simple multi-epoch training loop over both modalities.

    Args:
        train_loaders: {"vision": vision_loader, "audio": audio_loader}
        modalities: order of modalities to train each epoch
    """
    for epoch in range(num_epochs):
        for modality in modalities:
            loader = train_loaders.get(modality)
            if loader is None:
                continue

            mean_loss = train_one_epoch(
                dataloader=loader,
                modality=modality,
                modules=modules,
                cfg=cfg,
                text_embed_fn=text_embed_fn,
                optimizer=optimizer,
                device=device,
                epoch=epoch,
                log_every=log_every,
                log_fn=log_fn,
            )

            if log_fn is None:
                print(
                    f"[Epoch {epoch}] modality={modality} | mean_loss={mean_loss:.4f}"
                )
