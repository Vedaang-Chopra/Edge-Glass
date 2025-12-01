"""
data.py - Dataset and DataLoader utilities for multimodal alignment

Supports:
- Pre-extracted features (from .pt files)
- On-the-fly image loading (from HuggingFace datasets)
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
import json
import random

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

try:
    from PIL import Image
    import requests
    from io import BytesIO
except ImportError:
    Image = None


# ============================================================
# Pre-extracted Feature Dataset
# ============================================================

class FeatureDataset(Dataset):
    """
    Dataset for pre-extracted features stored as .pt files.
    
    Each .pt file should contain:
        {
            "features": Tensor (T, D) or (D,),
            "caption": str,
            ...optional metadata...
        }
    
    Index file format (JSON):
        [
            {"file": "path/to/feat_0.pt", ...},
            {"file": "path/to/feat_1.pt", ...},
            ...
        ]
    """
    
    def __init__(
        self,
        index_file: str | Path,
        modality: str = "vision",
    ):
        self.index_file = Path(index_file)
        self.base_dir = self.index_file.parent
        self.modality = modality
        
        with open(self.index_file, "r") as f:
            raw_index = json.load(f)
        
        # Validate and resolve paths
        self.index = []
        for rec in raw_index:
            try:
                path = self._resolve_path(rec["file"])
                rec = dict(rec)
                rec["resolved_path"] = str(path)
                self.index.append(rec)
            except FileNotFoundError:
                continue
        
        print(f"[FeatureDataset] Loaded {len(self.index)}/{len(raw_index)} valid entries")
    
    def _resolve_path(self, raw_path: str) -> Path:
        """Try to resolve the feature file path."""
        p = Path(raw_path)
        
        # Absolute path exists
        if p.is_absolute() and p.exists():
            return p
        
        # Relative to index directory
        candidate = self.base_dir / p
        if candidate.exists():
            return candidate
        
        # Just filename in base_dir
        candidate = self.base_dir / p.name
        if candidate.exists():
            return candidate
        
        # Try common fixes (data/X -> data/data/X)
        s = str(raw_path)
        for pattern in ["data/pixmo", "data/librispeech"]:
            if pattern in s:
                fixed = s.replace(pattern, f"data/{pattern}")
                if Path(fixed).exists():
                    return Path(fixed)
        
        raise FileNotFoundError(f"Cannot find: {raw_path}")
    
    def __len__(self) -> int:
        return len(self.index)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.index[idx]
        path = Path(rec["resolved_path"])
        
        blob = torch.load(path, map_location="cpu")
        
        feats = blob["features"]
        text = blob.get("caption", blob.get("text", ""))
        
        # Ensure features have sequence dimension
        if feats.dim() == 1:
            feats = feats.unsqueeze(0)  # (D,) -> (1, D)
        
        return {
            "features": feats,
            "text": text,
            "modality": self.modality,
        }


# ============================================================
# On-the-fly Image Dataset (for HuggingFace datasets)
# ============================================================

class ImageTextDataset(Dataset):
    """
    On-the-fly image loading from HuggingFace dataset.
    
    Works with datasets that have:
        - "image" column (PIL Image) or "image_url" (URL string)
        - "caption" or "text" column
    """
    
    def __init__(
        self,
        hf_dataset,
        image_column: str = "image",
        text_column: str = "caption",
        max_retries: int = 3,
    ):
        self.ds = hf_dataset
        self.image_col = image_column
        self.text_col = text_column
        self.max_retries = max_retries
        
        # Check columns
        cols = hf_dataset.column_names
        if self.image_col not in cols:
            if "image_url" in cols:
                self.image_col = "image_url"
            else:
                raise ValueError(f"No image column found. Available: {cols}")
        
        if self.text_col not in cols:
            if "text" in cols:
                self.text_col = "text"
            else:
                raise ValueError(f"No text column found. Available: {cols}")
        
        self.is_url = self.image_col == "image_url"
        print(f"[ImageTextDataset] Using columns: image={self.image_col}, text={self.text_col}")
    
    def __len__(self) -> int:
        return len(self.ds)
    
    def _load_from_url(self, url: str) -> "Image.Image":
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        return Image.open(BytesIO(resp.content)).convert("RGB")
    
    def _get_example(self, idx: int) -> Dict[str, Any]:
        ex = self.ds[idx]
        
        if self.is_url:
            img = self._load_from_url(ex[self.image_col])
        else:
            img = ex[self.image_col]
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img).convert("RGB")
            else:
                img = img.convert("RGB")
        
        text = ex[self.text_col]
        
        return {
            "image": img,
            "text": text,
        }
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        n = len(self.ds)
        
        for attempt in range(self.max_retries):
            try:
                return self._get_example((idx + attempt) % n)
            except Exception:
                continue
        
        # Random fallback
        for _ in range(self.max_retries):
            try:
                return self._get_example(random.randint(0, n - 1))
            except Exception:
                continue
        
        raise RuntimeError(f"Failed to load any image after {self.max_retries * 2} attempts")


# ============================================================
# Collate Functions
# ============================================================

def collate_features(batch: List[Dict]) -> Dict[str, Any]:
    """
    Collate function for FeatureDataset.
    
    Handles variable-length feature sequences.
    
    Returns:
        {
            "features": (B, T_max, D) padded
            "feature_mask": (B, T_max) bool
            "texts": List[str]
        }
    """
    feats_list = [b["features"] for b in batch]
    texts = [b["text"] for b in batch]
    
    # Get lengths
    lengths = [f.size(0) for f in feats_list]
    
    # Pad sequences
    padded = pad_sequence(feats_list, batch_first=True)  # (B, T_max, D)
    
    # Create mask
    B, T_max = padded.size(0), padded.size(1)
    mask = torch.zeros(B, T_max, dtype=torch.bool)
    for i, L in enumerate(lengths):
        mask[i, :L] = True
    
    return {
        "features": padded,
        "feature_mask": mask,
        "texts": texts,
    }


def collate_images(batch: List[Dict]) -> Dict[str, Any]:
    """
    Collate function for ImageTextDataset.
    
    Returns:
        {
            "images": List[PIL.Image]
            "texts": List[str]
        }
    """
    return {
        "images": [b["image"] for b in batch],
        "texts": [b["text"] for b in batch],
    }


# ============================================================
# DataLoader Factory
# ============================================================

def create_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    collate_fn: Optional[Callable] = None,
    drop_last: bool = True,
) -> DataLoader:
    """Create a DataLoader with sensible defaults."""
    
    # Auto-detect collate function
    if collate_fn is None:
        if isinstance(dataset, FeatureDataset):
            collate_fn = collate_features
        elif isinstance(dataset, ImageTextDataset):
            collate_fn = collate_images
        else:
            collate_fn = None  # use default
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=drop_last,
        pin_memory=torch.cuda.is_available(),
    )


# ============================================================
# Quick Dataset from Lists (for testing)
# ============================================================

class SimpleImageTextDataset(Dataset):
    """Simple dataset from lists of images and texts."""

    def __init__(self, images: List, texts: List[str]):
        assert len(images) == len(texts)
        self.images = images
        self.texts = texts

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            "image": self.images[idx],
            "text": self.texts[idx],
        }


# ============================================================
# Usage Note: In-Memory Datasets
# ============================================================
"""
For faster training with repeated epochs over the same data, consider using
the InMemoryImageTextDataset and InMemoryAudioTextDataset from in_memory_datasets.py.

These datasets pre-load all images/audio into memory at initialization,
avoiding repeated network requests or file I/O during training.

Example:
    from datasets import load_dataset
    from in_memory_datasets import InMemoryImageTextDataset

    # Load HuggingFace dataset
    hf_ds = load_dataset("allenai/pixmo-cap", split="train")

    # Create in-memory dataset (pre-loads all images)
    dataset = InMemoryImageTextDataset(
        hf_dataset=hf_ds,
        img_col="image_url",
        txt_col="caption",
        max_samples=10000,
        image_size=(224, 224),
    )

    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=collate_images,
    )

Benefits:
    - Much faster training (no repeated downloads/loading)
    - Consistent iteration times
    - Better for multi-epoch training

Considerations:
    - Requires sufficient RAM
    - Initial loading takes time
    - Use max_samples to limit memory usage
"""
