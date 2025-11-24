import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Callable, Tuple

from imports.align_training.steps import AlignmentModules, AlignmentConfig, forward_alignment_step


@torch.no_grad()
def eval_alignment(
    dataloader: DataLoader,
    modality: str,
    modules: AlignmentModules,
    cfg: AlignmentConfig,
    text_embed_fn: Callable[[list[str], int], torch.Tensor],
    device: torch.device,
) -> Dict[str, float]:
    """
    Evaluates:
      - validation Matryoshka loss
      - retrieval accuracy (R@1)

    R@1 = percentage of samples where the correct caption
          is the highest-scored match among the batch.
    """

    modules.projector.eval()
    if modules.perceiver is not None:
        modules.perceiver.eval()
    if modality == "vision" and modules.vision_adapter is not None:
        modules.vision_adapter.eval()
    if modality == "audio" and modules.audio_adapter is not None:
        modules.audio_adapter.eval()

    total_loss = 0.0
    total_batches = 0
    total_correct = 0
    total_samples = 0

    for batch in dataloader:
        # Run forward pass (produces loss already averaged)
        loss, metrics = forward_alignment_step(
            batch=batch,
            modality=modality,
            modules=modules,
            cfg=cfg,
            text_embed_fn=text_embed_fn,
            device=device,
        )

        total_loss += loss.item()
        total_batches += 1

        # --- Retrieval accuracy (R@1) ---
        feats = metrics.get("z_mod", None)
        if feats is None:
            # compute embeddings for retrieval manually:
            encoder_feats = batch["features"].to(device)
            encoder_mask = batch["feature_mask"].to(device)
            texts = batch["raw_text"]

            if modality == "vision":
                adapter = modules.vision_adapter
            else:
                adapter = modules.audio_adapter

            tokens = adapter(encoder_feats)
            if modules.perceiver is not None:
                latents = modules.perceiver(tokens, encoder_mask=encoder_mask)
            else:
                latents = tokens

            latent_tokens_llm = modules.projector(latents)
            h_mod = latent_tokens_llm.mean(dim=1)
            h_txt = text_embed_fn(texts, cfg.max_text_length)

        else:
            # if metrics injects h_mod in future versions
            h_mod, h_txt = metrics["z_mod"], metrics["z_txt"]

        # Normalize
        h_mod = F.normalize(h_mod, dim=-1)
        h_txt = F.normalize(h_txt, dim=-1)

        sim = h_mod @ h_txt.T   # (B, B)

        # For each row, the diagonal should be the max
        preds = sim.argmax(dim=1)
        labels = torch.arange(sim.size(0), device=device)

        correct = (preds == labels).sum().item()
        total_correct += correct
        total_samples += sim.size(0)

    avg_loss = total_loss / max(1, total_batches)
    acc = total_correct / max(1, total_samples)

    return {
        "val_loss": avg_loss,
        "val_r1": acc,
        "num_batches": total_batches,
        "num_samples": total_samples,
    }
