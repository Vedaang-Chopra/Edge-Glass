"""Dataset builders for multimodal training."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Optional
from io import BytesIO

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.io import read_image
from PIL import Image
import pandas as pd


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


class PixmoParquetImageTextDataset(Dataset):
    """Dataset for loading Pixmo parquet files with embedded image bytes.

    This dataset reads from parquet files that contain:
    - image_bytes: Raw image bytes
    - caption: Text caption for the image
    - sample_id: Unique identifier
    - image_url: Optional URL (not used for loading)

    Args:
        parquet_path: Path to the parquet file
        image_transforms: Optional image transforms to apply
        max_text_length: Maximum text length (characters)
        text_dropout_prob: Probability of dropping text during training
    """

    def __init__(
        self,
        parquet_path: str | Path,
        image_transforms: Optional[Callable] = None,
        max_text_length: Optional[int] = None,
        text_dropout_prob: float = 0.0,
    ):
        self.parquet_path = Path(parquet_path)
        self.image_transforms = image_transforms
        self.max_text_length = max_text_length
        self.text_dropout_prob = text_dropout_prob

        # Load parquet file
        self.df = pd.read_parquet(self.parquet_path)

        # Validate required columns
        required_cols = ['image_bytes', 'caption', 'sample_id']
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in parquet: {missing_cols}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Decode image bytes to PIL Image
        image_bytes = row['image_bytes']
        image = Image.open(BytesIO(image_bytes)).convert('RGB')

        # Convert to tensor [0, 1] range
        import torchvision.transforms as T
        to_tensor = T.ToTensor()
        image_tensor = to_tensor(image)

        # Apply transforms
        if self.image_transforms:
            image_tensor = self.image_transforms(image_tensor)

        # Get caption
        caption = row['caption']
        if self.max_text_length is not None:
            caption = caption[:self.max_text_length]

        # Apply text dropout during training
        if self.text_dropout_prob > 0 and torch.rand(1).item() < self.text_dropout_prob:
            caption = ""  # Empty caption forces model to rely on image

        return {
            "image": image_tensor,
            "text": caption,
            "sample_id": row['sample_id'],
        }


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


def build_image_datasets_from_parquet(
    cfg,
    train_parquet_path: str | Path,
    val_parquet_path: str | Path,
    test_parquet_path: Optional[str | Path] = None,
    train_transforms: Optional[Callable] = None,
    val_transforms: Optional[Callable] = None,
    max_text_length: Optional[int] = None,
    text_dropout_prob: float = 0.1,
) -> Dict[str, Dataset]:
    """Build image-text datasets from parquet files.

    Args:
        cfg: Dataset configuration
        train_parquet_path: Path to training parquet file
        val_parquet_path: Path to validation parquet file
        test_parquet_path: Optional path to test parquet file
        train_transforms: Transforms for training images
        val_transforms: Transforms for validation/test images
        max_text_length: Maximum text length
        text_dropout_prob: Probability of dropping text during training

    Returns:
        Dictionary with 'train', 'val', and optionally 'test' datasets
    """
    datasets = {}

    # Training dataset with text dropout
    datasets['train'] = PixmoParquetImageTextDataset(
        parquet_path=train_parquet_path,
        image_transforms=train_transforms,
        max_text_length=max_text_length,
        text_dropout_prob=text_dropout_prob,
    )

    # Validation dataset (no text dropout)
    datasets['val'] = PixmoParquetImageTextDataset(
        parquet_path=val_parquet_path,
        image_transforms=val_transforms,
        max_text_length=max_text_length,
        text_dropout_prob=0.0,  # No dropout during validation
    )

    # Optional test dataset
    if test_parquet_path is not None:
        datasets['test'] = PixmoParquetImageTextDataset(
            parquet_path=test_parquet_path,
            image_transforms=val_transforms,
            max_text_length=max_text_length,
            text_dropout_prob=0.0,  # No dropout during testing
        )

    return datasets
