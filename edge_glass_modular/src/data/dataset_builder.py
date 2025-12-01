"""Dataset builders for multimodal training."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.io import read_image


class ImageTextDataset(Dataset):
    def __init__(
        self,
        metadata=None,
        metadata_path: Optional[str | Path] = None,
        transform: Optional[Callable] = None,
        image_transforms: Optional[Callable] = None,
        max_text_length: Optional[int] = None,
    ):
        if metadata is None and metadata_path is None:
            raise ValueError("Provide either metadata or metadata_path.")

        if metadata is None and metadata_path is not None:
            import json

            metadata = json.loads(Path(metadata_path).read_text())

        # Allow either transform or image_transforms for compatibility
        if transform is None and image_transforms is not None:
            transform = image_transforms

        self.metadata = metadata
        self.transform = transform
        self.max_text_length = max_text_length

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        item = self.metadata[idx]
        image = read_image(item["image_path"]).float() / 255.0
        if self.transform:
            image = self.transform(image)
        text = item["caption"]
        if self.max_text_length is not None:
            text = text[: self.max_text_length]
        return {
            "image": image,
            "text": text,
            "sample_id": item["sample_id"],
        }


class AudioTextDataset(Dataset):
    def __init__(self, metadata, audio_loader: Callable, transform: Optional[Callable] = None, target_length: int = 16000 * 10):
        self.metadata = metadata
        self.audio_loader = audio_loader
        self.transform = transform
        self.target_length = target_length

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        item = self.metadata[idx]
        audio = self.audio_loader(item["audio_path"])
        if audio.size(-1) > self.target_length:
            audio = audio[..., : self.target_length]
        else:
            pad = self.target_length - audio.size(-1)
            audio = F.pad(audio, (0, pad))
        if self.transform:
            audio = self.transform(audio)
        return {
            "audio": audio,
            "text": item["caption"],
            "sample_id": item["sample_id"],
        }


class InstructionDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def _read_metadata(path: Path):
    import json

    return json.loads(Path(path).read_text())


class TriModalDataset(Dataset):
    def __init__(
        self,
        vision_meta,
        audio_meta,
        transform,
        audio_loader,
        audio_transform,
    ):
        self.vision_meta = vision_meta
        self.audio_meta = audio_meta
        self.transform = transform
        self.audio_loader = audio_loader
        self.audio_transform = audio_transform
        self.length = min(len(vision_meta), len(audio_meta))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        v = self.vision_meta[idx]
        a = self.audio_meta[idx]
        image = read_image(v["image_path"]).float() / 255.0
        if self.transform:
            image = self.transform(image)
        audio = self.audio_loader(a["audio_path"])
        if self.audio_transform:
            audio = self.audio_transform(audio)
        return {
            "image": image,
            "audio": audio,
            "text": v["caption"],
            "sample_id": f"{v['sample_id']}__{a['sample_id']}",
        }


def build_datasets(dataset_cfg, transforms) -> Dict[str, Dataset]:
    base = Path(dataset_cfg.cache_dir)
    datasets = {}
    vision_meta = audio_meta = None
    if dataset_cfg.image_root:
        vision_meta = _read_metadata(base / dataset_cfg.image_root / "metadata.json")
        datasets["vision_text"] = ImageTextDataset(vision_meta, transform=transforms.get("vision"))
    if dataset_cfg.audio_root:
        audio_meta = _read_metadata(base / dataset_cfg.audio_root / "metadata.json")
        datasets["audio_text"] = AudioTextDataset(
            audio_meta,
            audio_loader=transforms["audio_loader"],
            transform=transforms.get("audio"),
        )
    if vision_meta and audio_meta:
        datasets["tri_modal"] = TriModalDataset(
            vision_meta,
            audio_meta,
            transform=transforms.get("vision"),
            audio_loader=transforms["audio_loader"],
            audio_transform=transforms.get("audio"),
        )
    if dataset_cfg.text_root:
        metadata = _read_metadata(base / dataset_cfg.text_root / "instruction.json")
        datasets["instruction"] = InstructionDataset(metadata)
    return datasets
