"""DataModule that consumes offline metadata exports."""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader

from .dataset_builder import build_datasets


class MultimodalDataModule:
    def __init__(self, dataset_cfg, transforms):
        self.dataset_cfg = dataset_cfg
        self.transforms = transforms
        self.datasets: Optional[Dict[str, object]] = None

    def setup(self):
        self.datasets = build_datasets(self.dataset_cfg, self.transforms)

    def collate(self, batch: List[Dict]):
        collated = {"sample_id": [item["sample_id"] for item in batch]}
        if "image" in batch[0]:
            collated["image"] = torch.stack([item["image"] for item in batch])
        if "audio" in batch[0]:
            collated["audio"] = torch.stack([item["audio"] for item in batch])
        collated["text"] = [item["text"] for item in batch]
        return collated

    def train_dataloader(self):
        if "tri_modal" in self.datasets:
            dataset = self.datasets["tri_modal"]
        else:
            dataset = self.datasets["vision_text"]
        return DataLoader(
            dataset,
            batch_size=self.dataset_cfg.batch_size,
            shuffle=True,
            num_workers=self.dataset_cfg.num_workers,
            pin_memory=self.dataset_cfg.pin_memory,
            persistent_workers=self.dataset_cfg.persistent_workers,
            collate_fn=self.collate,
        )
