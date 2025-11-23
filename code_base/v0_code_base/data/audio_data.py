"""
audio_data.py

Generic audio-text dataset loading and feature caching.

- Works with any Hugging Face audio dataset that has:
    * an "audio" column (array + sampling_rate) or similar dict
    * a text/label column

- Independent of any global Config/YAML.
- Takes constructor args / dataclass.

Expected encoder contract:
    encoder.encode(waveform_array_or_tensor, sr, pooled=False) -> Tensor (1, T, D)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Dict, Union, Any, Tuple

import json
from pathlib import Path

import numpy as np
from datasets import load_dataset, Dataset as HFDataset, DatasetDict
from tqdm.auto import tqdm

import torch
from torch.utils.data import Dataset


# ---------------------------
# Config dataclass
# ---------------------------

@dataclass
class AudioDatasetConfig:
    """
    Config for a generic HF audio + text dataset.

    Example for LibriSpeech-style:

        cfg = AudioDatasetConfig(
            dataset_name="openslr/librispeech_asr",
            hf_config_name="clean",
            audio_column="audio",
            text_column="text",
            train_split="train.100",
            val_split="validation.clean",
            train_subset_size=2000,
            val_subset_size=500,
        )
    """
    dataset_name: str
    hf_config_name: Optional[str] = None

    audio_column: str = "audio"
    text_column: str = "text"

    train_split: str = "train"
    val_split: Optional[str] = None

    train_subset_size: int = 2000
    val_subset_size: int = 500


# ---------------------------
# Data module
# ---------------------------

class AudioDataModule:
    """
    Data module for audio-text datasets.

    Usage:

        cfg = AudioDatasetConfig(
            dataset_name="openslr/librispeech_asr",
            hf_config_name="clean",
            audio_column="audio",
            text_column="text",
            train_split="train.100",
            val_split="validation.clean",
            train_subset_size=2000,
            val_subset_size=500,
        )

        dm = AudioDataModule(cfg)
        dm.prepare_data()

        train_ds = dm.train_dataset
        val_ds   = dm.val_dataset

        train_index, val_index = dm.cache_audio_features(
            encoder=audio_encoder,
            out_dir="features/audio",
            prefix="audio",
            dtype=torch.float16,
        )

        train_feat_ds = AudioFeaturesDataset(train_index)
        val_feat_ds   = AudioFeaturesDataset(val_index)
    """

    def __init__(self, cfg: AudioDatasetConfig):
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
            f"[Audio] Prepared datasets: "
            f"train={len(self.train_dataset)} examples, "
            f"val={len(self.val_dataset)} examples."
        )

    def cache_audio_features(
        self,
        encoder: Any,
        out_dir: Union[str, Path],
        prefix: str = "audio",
        dtype: torch.dtype = torch.float16,
    ) -> Tuple[Path, Path]:
        """
        Encode each example's waveform into token-level features.

        encoder: must implement:
            encoder.encode(waveform, sr, pooled=False) -> Tensor (1, T, D)

        Assumes dataset audio column is a dict with:
            "array": np.ndarray
            "sampling_rate": int

        Returns:
            (train_index_path, val_index_path)
        """
        out_dir = Path(out_dir).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        train_index_path = out_dir / f"{prefix}_train_index.json"
        val_index_path = out_dir / f"{prefix}_val_index.json"

        print("[Audio] Caching train split features...")
        train_index = self._cache_split(
            ds=self._get_split_dataset("train"),
            split_name="train",
            encoder=encoder,
            out_dir=out_dir,
            prefix=prefix,
            dtype=dtype,
        )
        with open(train_index_path, "w") as f:
            json.dump(train_index, f, indent=2)

        print("[Audio] Caching val split features...")
        val_index = self._cache_split(
            ds=self._get_split_dataset("val"),
            split_name="val",
            encoder=encoder,
            out_dir=out_dir,
            prefix=prefix,
            dtype=dtype,
        )
        with open(val_index_path, "w") as f:
            json.dump(val_index, f, indent=2)

        return train_index_path, val_index_path

    # Internal helpers -----------

    def _load_raw(self) -> None:
        print(
            f"[Audio] Loading dataset '{self.cfg.dataset_name}' "
            f"(config={self.cfg.hf_config_name})..."
        )

        if self.cfg.hf_config_name is not None:
            ds_dict = load_dataset(self.cfg.dataset_name, self.cfg.hf_config_name)
        else:
            ds_dict = load_dataset(self.cfg.dataset_name)

        if not isinstance(ds_dict, DatasetDict):
            ds_dict = DatasetDict({"train": ds_dict})

        self._raw = ds_dict
        print(f"[Audio] Available splits: {list(ds_dict.keys())}")

    def _filter_splits(self) -> Tuple[HFDataset, Optional[HFDataset]]:
        assert self._raw is not None, "Call _load_raw() first."

        audio_col = self.cfg.audio_column
        text_col = self.cfg.text_column

        def is_valid(example: Dict) -> bool:
            audio = example.get(audio_col)
            text = example.get(text_col)

            if audio is None or "array" not in audio or "sampling_rate" not in audio:
                return False
            if text is None:
                return False
            if isinstance(text, str):
                return text.strip() != ""
            return False

        ds_dict = self._raw

        if self.cfg.train_split not in ds_dict:
            raise ValueError(
                f"Train split '{self.cfg.train_split}' not found. "
                f"Available: {list(ds_dict.keys())}"
            )

        print("[Audio] Filtering train split...")
        train_raw = ds_dict[self.cfg.train_split]
        train_filtered = train_raw.filter(is_valid)

        val_filtered: Optional[HFDataset] = None
        if self.cfg.val_split is not None and self.cfg.val_split in ds_dict:
            print("[Audio] Filtering val split...")
            val_raw = ds_dict[self.cfg.val_split]
            val_filtered = val_raw.filter(is_valid)
        else:
            print(
                "[Audio] No explicit val split; will derive val subset from train."
            )

        print(
            f"[Audio] After filtering: train={len(train_filtered)}"
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

    def _cache_split(
        self,
        ds: HFDataset,
        split_name: str,
        encoder: Any,
        out_dir: Path,
        prefix: str,
        dtype: torch.dtype,
    ) -> Dict:
        audio_col = self.cfg.audio_column
        text_col = self.cfg.text_column

        items: List[Dict] = []

        for i in tqdm(range(len(ds)), desc=f"Caching {split_name} audio"):
            ex = ds[i]
            audio = ex[audio_col]  # {"array": ..., "sampling_rate": ...}
            text = ex[text_col]

            array = np.array(audio["array"], dtype=np.float32)
            sr = int(audio["sampling_rate"])

            with torch.no_grad():
                feats = encoder.encode(array, sr=sr, pooled=False)  # (1, T, D)
                feats = feats.squeeze(0).to(dtype=dtype).cpu()      # (T, D)

            T, D = feats.shape
            filename = f"{prefix}_{split_name}_{i:06d}.pt"
            filepath = out_dir / filename

            torch.save(
                {
                    "features": feats,
                    "text": text,
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
                    "text": text,
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

class AudioFeaturesDataset(Dataset):
    """
    Dataset that reads cached audio token features from a JSON index.

    Each item:
        {
          "features": Tensor (T, D),
          "text": str,
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
            "text": data["text"],
            "orig_idx": data["orig_idx"],
        }
