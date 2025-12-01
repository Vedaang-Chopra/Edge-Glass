"""
dataset_builders.py

Utility functions to create local Parquet subsets of the PixMo-Cap
and MusicCaps datasets for alignment experiments.
"""

from pathlib import Path
from typing import Optional

from datasets import load_dataset



def build_pixmocap_parquet(
    output_path: Path,
    split: str = "train",
    max_samples: Optional[int] = None,
    shuffle_seed: int = 42,
) -> None:
    """
    Download a subset of PixMo-Cap and save it as a Parquet file.

    The resulting Parquet file will keep all original columns, including:
    - `image_url`: used by InMemoryImageTextDataset
    - `caption`   : used as text field

    Args:
        output_path: Where to save the Parquet file.
        split: HF split to use (default: "train").
        max_samples: If provided, randomly select at most this many samples.
        shuffle_seed: Seed for shuffling before subsetting.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"📥 Loading PixMo-Cap split='{split}' from Hugging Face...")
    ds = load_dataset("allenai/pixmo-cap", split=split)

    print(f"   Total available samples: {len(ds):,}")
    if max_samples is not None and max_samples < len(ds):
        print(f"   Shuffling and selecting first {max_samples:,} samples (seed={shuffle_seed})...")
        ds = ds.shuffle(seed=shuffle_seed).select(range(max_samples))
    else:
        print("   Using full split (no subsetting).")


    print(f"💾 Saving subset to Parquet: {output_path}")
    ds.to_parquet(str(output_path))
    print("✅ Done! PixMo-Cap subset saved.")


def build_musiccaps_parquet(
    output_path: Path,
    split: str = "train",
    max_samples: Optional[int] = None,
    shuffle_seed: int = 42,
) -> None:
    """
    Download a subset of MusicCaps and save it as a Parquet file.

    The resulting Parquet file will keep all original columns, including:
    - `audio`   : HF Audio column (waveforms + metadata)
    - `caption` : used as text field

    Args:
        output_path: Where to save the Parquet file.
        split: HF split to use (default: "train").
        max_samples: If provided, randomly select at most this many samples.
        shuffle_seed: Seed for shuffling before subsetting.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"📥 Loading MusicCaps split='{split}' from Hugging Face...")
    ds = load_dataset("google/MusicCaps", split=split)

    print(f"   Total available samples: {len(ds):,}")
    if max_samples is not None and max_samples < len(ds):
        print(f"   Shuffling and selecting first {max_samples:,} samples (seed={shuffle_seed})...")
        ds = ds.shuffle(seed=shuffle_seed).select(range(max_samples))
    else:
        print("   Using full split (no subsetting).")


    print(f"💾 Saving subset to Parquet: {output_path}")
    ds.to_parquet(str(output_path))
    print("✅ Done! MusicCaps subset saved.")
