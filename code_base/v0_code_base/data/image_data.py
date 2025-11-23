"""
image_caption_data.py

Generic image-caption dataset loading, inspection, and feature caching.

- Works with any Hugging Face dataset that has:
    * an image URL or path column
    * a caption/text column

- Independent of any global Config/YAML.
- Takes simple constructor arguments / dataclass.

Responsibilities:
    - Load HF dataset (non-streaming).
    - Filter bad rows (missing/empty image or caption).
    - Subsample train / val to desired sizes.
    - Cache image patch features with a provided encoder.
    - Provide a FeaturesDataset that reads cached .pt feature files.

Expected encoder contract (duck-typing):
    encoder.encode([pil_image], pooled=False) -> Tensor (1, T, D)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Union, Any

import json
from io import BytesIO
from pathlib import Path

import requests
from datasets import load_dataset, Dataset as HFDataset, DatasetDict
from PIL import Image
from tqdm.auto import tqdm

import torch
from torch.utils.data import Dataset


# ---------------------------
# Config dataclass
# ---------------------------

@dataclass
class ImageCaptionDatasetConfig:
    """
    Configuration for a generic image-caption dataset.

    Example for PixMo-Cap:

        cfg = ImageCaptionDatasetConfig(
            dataset_name="allenai/pixmo-cap",
            hf_config_name=None,      # or "default"
            url_column="image",
            caption_column="caption",
            train_split="train",
            val_split=None,           # derive val from train
            train_subset_size=2000,
            val_subset_size=500,
        )
    """
    dataset_name: str
    hf_config_name: Optional[str] = None

    url_column: str = "image"
    caption_column: str = "caption"

    train_split: str = "train"
    val_split: Optional[str] = None

    train_subset_size: int = 2000
    val_subset_size: int = 500


# ---------------------------
# Data module
# ---------------------------

class ImageCaptionDataModule:
    """
    Data module for image-caption datasets.

    Usage:

        cfg = ImageCaptionDatasetConfig(
            dataset_name="allenai/pixmo-cap",
            url_column="image",
            caption_column="caption",
            train_subset_size=2000,
            val_subset_size=500,
        )

        dm = ImageCaptionDataModule(cfg)
        dm.prepare_data()

        train_ds = dm.train_dataset
        val_ds   = dm.val_dataset

        train_index_path, val_index_path = dm.cache_image_features(
            encoder=vision_encoder,
            out_dir="features/pixmo",
            prefix="pixmo",
            dtype=torch.float16,
        )

        train_feat_ds = ImageCaptionFeaturesDataset(train_index_path)
        val_feat_ds   = ImageCaptionFeaturesDataset(val_index_path)
    """

    def __init__(self, cfg: ImageCaptionDatasetConfig):
        self.cfg = cfg

        self._raw: Optional[DatasetDict] = None
        self.train_dataset: Optional[HFDataset] = None
        self.val_dataset: Optional[HFDataset] = None

    # Public API -----------------

    def prepare_data(self) -> None:
        """
        Load HF dataset, filter invalid rows, and create train/val subsets.
        """
        self._load_raw()
        train_filtered, val_filtered = self._filter_splits()
        self._create_subsets(train_filtered, val_filtered)

        print(
            f"[ImageCaption] Prepared datasets: "
            f"train={len(self.train_dataset)} examples, "
            f"val={len(self.val_dataset)} examples."
        )

    def cache_image_features(
        self,
        encoder: Any,
        out_dir: Union[str, Path],
        prefix: str = "imgcap",
        dtype: torch.dtype = torch.float16,
        timeout: float = 10.0,
    ) -> Tuple[Path, Path]:
        """
        Encode each image into patch-level features using the provided encoder.

        encoder: must implement:
            encoder.encode([PIL.Image], pooled=False) -> Tensor (1, T, D)

        Saves for each example a .pt file with:
            {
              "features": (T, D),
              "caption": str,
              "image_id": original ID/index if available,
              "image_source": the URL/path used,
            }

        Also writes two JSON index files:
            prefix_train_index.json
            prefix_val_index.json

        Returns:
            (train_index_path, val_index_path)
        """
        out_dir = Path(out_dir).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        train_index_path = out_dir / f"{prefix}_train_index.json"
        val_index_path = out_dir / f"{prefix}_val_index.json"

        print("[ImageCaption] Caching train split features...")
        train_index = self._cache_split(
            ds=self._get_split_dataset("train"),
            split_name="train",
            encoder=encoder,
            out_dir=out_dir,
            prefix=prefix,
            dtype=dtype,
            timeout=timeout,
        )
        with open(train_index_path, "w") as f:
            json.dump(train_index, f, indent=2)

        print("[ImageCaption] Caching val split features...")
        val_index = self._cache_split(
            ds=self._get_split_dataset("val"),
            split_name="val",
            encoder=encoder,
            out_dir=out_dir,
            prefix=prefix,
            dtype=dtype,
            timeout=timeout,
        )
        with open(val_index_path, "w") as f:
            json.dump(val_index, f, indent=2)

        return train_index_path, val_index_path

    def get_samples(
        self,
        split: str = "train",
        n: int = 3,
    ) -> List[Tuple[str, str]]:
        """
        Return up to n (image_source, caption) pairs from a split for sanity checks.
        """
        ds = self._get_split_dataset(split)
        n = min(n, len(ds))
        out: List[Tuple[str, str]] = []

        for i in range(n):
            ex = ds[i]
            url = ex[self.cfg.url_column]
            caption = ex[self.cfg.caption_column]
            if isinstance(caption, list):
                caption = " ".join(caption)
            out.append((url, caption))

        return out

    # Internal helpers -----------

    def _load_raw(self) -> None:
        print(
            f"[ImageCaption] Loading dataset '{self.cfg.dataset_name}' "
            f"(config={self.cfg.hf_config_name})..."
        )

        if self.cfg.hf_config_name is not None:
            ds_dict = load_dataset(self.cfg.dataset_name, self.cfg.hf_config_name)
        else:
            ds_dict = load_dataset(self.cfg.dataset_name)

        if not isinstance(ds_dict, DatasetDict):
            ds_dict = DatasetDict({"train": ds_dict})

        self._raw = ds_dict
        print(f"[ImageCaption] Available splits: {list(ds_dict.keys())}")

    def _filter_splits(self) -> Tuple[HFDataset, Optional[HFDataset]]:
        assert self._raw is not None, "Call _load_raw() first."

        url_col = self.cfg.url_column
        cap_col = self.cfg.caption_column

        def is_valid(example: Dict) -> bool:
            url = example.get(url_col)
            caption = example.get(cap_col)

            if not isinstance(url, str) or not url.strip():
                return False

            if caption is None:
                return False
            if isinstance(caption, str):
                return caption.strip() != ""
            if isinstance(caption, list):
                return len(caption) > 0 and any(
                    isinstance(c, str) and c.strip() for c in caption
                )
            return False

        ds_dict = self._raw

        if self.cfg.train_split not in ds_dict:
            raise ValueError(
                f"Train split '{self.cfg.train_split}' not found. "
                f"Available: {list(ds_dict.keys())}"
            )

        print("[ImageCaption] Filtering train split...")
        train_raw = ds_dict[self.cfg.train_split]
        train_filtered = train_raw.filter(is_valid)

        val_filtered: Optional[HFDataset] = None
        if self.cfg.val_split is not None and self.cfg.val_split in ds_dict:
            print("[ImageCaption] Filtering val split...")
            val_raw = ds_dict[self.cfg.val_split]
            val_filtered = val_raw.filter(is_valid)
        else:
            print(
                "[ImageCaption] No explicit val split; will derive from train subset."
            )

        print(
            f"[ImageCaption] After filtering: train={len(train_filtered)}"
            + (f", val={len(val_filtered)}" if val_filtered is not None else "")
        )
        return train_filtered, val_filtered

    def _create_subsets(
        self,
        train_filtered: HFDataset,
        val_filtered: Optional[HFDataset],
    ) -> None:
        ts = self.cfg.train_subset_size
        vs = self.cfg.val_subset_size

        if val_filtered is not None:
            n_train = min(ts, len(train_filtered))
            n_val = min(vs, len(val_filtered))

            self.train_dataset = train_filtered.select(range(n_train))
            self.val_dataset = val_filtered.select(range(n_val))
        else:
            n_train = min(ts, len(train_filtered))
            self.train_dataset = train_filtered.select(range(n_train))

            remaining = train_filtered.select(range(n_train, len(train_filtered)))
            n_val = min(vs, len(remaining))
            self.val_dataset = remaining.select(range(n_val))

        assert self.train_dataset is not None
        assert self.val_dataset is not None

    def _get_split_dataset(self, split: str) -> HFDataset:
        if split == "train":
            if self.train_dataset is None:
                raise ValueError("train_dataset not ready. Call prepare_data().")
            return self.train_dataset

        if split in ("val", "validation"):
            if self.val_dataset is None:
                raise ValueError("val_dataset not ready. Call prepare_data().")
            return self.val_dataset

        raise ValueError(f"Unknown split '{split}'. Use 'train' or 'val'.")

    def _fetch_image(self, source: str, timeout: float = 10.0) -> Image.Image:
        """
        Fetch an image from a URL or local path.
        """
        if source.startswith("http://") or source.startswith("https://"):
            resp = requests.get(source, timeout=timeout)
            resp.raise_for_status()
            img = Image.open(BytesIO(resp.content)).convert("RGB")
        else:
            img = Image.open(source).convert("RGB")
        return img

    def _cache_split(
        self,
        ds: HFDataset,
        split_name: str,
        encoder: Any,
        out_dir: Path,
        prefix: str,
        dtype: torch.dtype,
        timeout: float,
    ) -> Dict:
        items = []

        for i in tqdm(range(len(ds)), desc=f"Caching {split_name} images"):
            ex = ds[i]
            source = ex[self.cfg.url_column]
            caption = ex[self.cfg.caption_column]
            if isinstance(caption, list):
                caption = " ".join(caption)

            try:
                img = self._fetch_image(source, timeout=timeout)
            except Exception as e:
                print(f"[ImageCaption] Warning: failed to fetch {source}: {e}")
                continue

            with torch.no_grad():
                feats = encoder.encode([img], pooled=False)  # (1, T, D)
                feats = feats.squeeze(0).to(dtype=dtype).cpu()  # (T, D)

            T, D = feats.shape
            filename = f"{prefix}_{split_name}_{i:06d}.pt"
            filepath = out_dir / filename

            torch.save(
                {
                    "features": feats,
                    "caption": caption,
                    "image_source": source,
                    "orig_idx": i,
                },
                filepath,
            )

            items.append(
                {
                    "file": filename,
                    "orig_idx": i,
                    "num_tokens": int(T),
                    "feat_dim": int(D),
                    "caption": caption,
                    "image_source": source,
                }
            )

        return {
            "split": split_name,
            "features_dir": str(out_dir),
            "items": items,
        }


# ---------------------------
# Features Dataset
# ---------------------------

class ImageCaptionFeaturesDataset(Dataset):
    """
    Dataset that reads cached image features and metadata from a JSON index.

    Each item:
        {
          "features": Tensor (T, D),
          "caption": str,
          "image_source": str,
          "orig_idx": int,
        }
    """

    def __init__(self, index_path: Union[str, Path]):
        index_path = Path(index_path).expanduser().resolve()
        with open(index_path, "r") as f:
            index = json.load(f)

        self.split: str = index["split"]
        self.features_dir = Path(index["features_dir"]).expanduser().resolve()
        self.items: List[Dict] = index["items"]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict:
        meta = self.items[idx]
        feat_path = self.features_dir / meta["file"]
        data = torch.load(feat_path)

        return {
            "features": data["features"],
            "caption": data["caption"],
            "image_source": data["image_source"],
            "orig_idx": data["orig_idx"],
        }
