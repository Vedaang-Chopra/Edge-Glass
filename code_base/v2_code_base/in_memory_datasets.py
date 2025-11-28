"""
in_memory_datasets.py - In-memory dataset implementations for faster training

Pre-loads all images and audio into memory to avoid repeated downloads/loading during training.
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial

import torch
from torch.utils.data import Dataset

try:
    from PIL import Image
    import requests
    from io import BytesIO
except ImportError:
    Image = None
    requests = None

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False


# ============================================================
# Worker Functions for Multiprocessing
# ============================================================

def _load_image_worker(args):
    """
    Worker function to load a single image in parallel.

    Args:
        args: Tuple of (idx, img_data, caption_data, image_size)

    Returns:
        Tuple of (idx, loaded_image or None, caption_str)
    """
    idx, img_data, caption_data, image_size = args

    # Load image
    try:
        if isinstance(img_data, str):
            # It's a URL
            import requests
            from io import BytesIO
            from PIL import Image

            resp = requests.get(img_data, timeout=10)
            resp.raise_for_status()
            img = Image.open(BytesIO(resp.content)).convert("RGB")
        elif hasattr(img_data, 'convert'):
            # Already a PIL Image
            from PIL import Image
            img = img_data.convert("RGB")
        else:
            # Try to convert from array
            from PIL import Image
            import numpy as np
            img = Image.fromarray(np.array(img_data)).convert("RGB")

        # Resize to target size
        from PIL import Image
        img = img.resize(image_size, Image.Resampling.LANCZOS)
    except Exception as e:
        img = None

    # Process caption
    caption = caption_data
    if isinstance(caption, list):
        caption = caption[0] if caption else ""
    if isinstance(caption, dict):
        caption = caption.get("raw", str(caption))
    caption = str(caption)

    return (idx, img, caption)


def _load_audio_worker(args):
    """
    Worker function to load a single audio sample in parallel.

    Args:
        args: Tuple of (idx, audio_item, caption_data, target_sr, max_length)

    Returns:
        Tuple of (idx, waveform or None, caption_str)
    """
    idx, audio_item, caption_data, target_sr, max_length = args

    # Load audio
    try:
        import numpy as np

        # If audio is already loaded (HuggingFace datasets format)
        if isinstance(audio_item, dict) and "array" in audio_item:
            waveform = audio_item["array"]
            orig_sr = audio_item.get("sampling_rate", target_sr)

            # Resample if needed
            if orig_sr != target_sr:
                try:
                    import librosa
                    waveform = librosa.resample(
                        waveform,
                        orig_sr=orig_sr,
                        target_sr=target_sr
                    )
                except ImportError:
                    pass  # No resampling if librosa not available

            # Truncate or pad to max_length
            if len(waveform) > max_length:
                waveform = waveform[:max_length]
            elif len(waveform) < max_length:
                waveform = np.pad(waveform, (0, max_length - len(waveform)))

            waveform = waveform.astype(np.float32)

        # If audio_item is a numpy array directly
        elif isinstance(audio_item, np.ndarray):
            waveform = audio_item.astype(np.float32)

            # Truncate or pad
            if len(waveform) > max_length:
                waveform = waveform[:max_length]
            elif len(waveform) < max_length:
                waveform = np.pad(waveform, (0, max_length - len(waveform)))
        else:
            waveform = None
    except Exception as e:
        waveform = None

    # Process caption
    caption = caption_data
    if isinstance(caption, list):
        caption = caption[0] if caption else ""
    if isinstance(caption, dict):
        caption = caption.get("raw", str(caption))
    caption = str(caption)

    return (idx, waveform, caption)


# ============================================================
# In-Memory Image-Text Dataset
# ============================================================

class InMemoryImageTextDataset(Dataset):
    """
    Pre-loads all images from PixMo-Cap (or similar) into memory.

    This avoids repeated network requests during training.
    Images are loaded once at initialization and kept in memory.
    Uses multiprocessing for faster loading.
    """

    def __init__(
        self,
        hf_dataset,
        img_col: str = "image_url",
        txt_col: str = "caption",
        max_samples: Optional[int] = None,
        image_size: tuple = (224, 224),
        num_workers: Optional[int] = None,
    ):
        """
        Args:
            hf_dataset: HuggingFace dataset object
            img_col: Column name for images/URLs
            txt_col: Column name for text captions
            max_samples: Limit number of samples (None = all)
            image_size: Resize images to this size (width, height)
            num_workers: Number of parallel workers (None = auto-detect)
        """
        self.img_col = img_col
        self.txt_col = txt_col
        self.image_size = image_size

        # Limit dataset size if requested
        if max_samples is not None:
            hf_dataset = hf_dataset.select(range(min(max_samples, len(hf_dataset))))

        # Auto-detect number of workers
        if num_workers is None:
            num_workers = min(cpu_count(), 32)  # Cap at 32 to avoid too many connections

        print(f"\n📥 Pre-loading {len(hf_dataset)} images into memory...")
        print(f"   Image size: {image_size}")
        print(f"   Using {num_workers} parallel workers")

        # Pre-load all images and captions using multiprocessing
        self.images = []
        self.captions = []
        self.failed_indices = []

        # Prepare data for parallel processing
        items_data = [
            (idx, hf_dataset[idx][img_col], hf_dataset[idx][txt_col], image_size)
            for idx in range(len(hf_dataset))
        ]

        # Use multiprocessing pool to load images in parallel
        with Pool(processes=num_workers) as pool:
            results = list(
                tqdm(
                    pool.imap(_load_image_worker, items_data, chunksize=max(1, len(items_data) // (num_workers * 4))),
                    total=len(items_data),
                    desc="Loading images"
                )
            )

        # Process results
        for idx, img, caption in results:
            if img is None:
                # Use a blank fallback image
                img = Image.new("RGB", image_size, color=(128, 128, 128))
                self.failed_indices.append(idx)

            self.images.append(img)
            self.captions.append(caption)

        print(f"✅ Loaded {len(self.images)} images into memory")
        if self.failed_indices:
            print(f"   ⚠️  {len(self.failed_indices)} images failed to load (using fallback)")

    def _load_image(self, img_data) -> Image.Image:
        """Load image from URL or PIL Image object."""
        if isinstance(img_data, str):
            # It's a URL
            resp = requests.get(img_data, timeout=10)
            resp.raise_for_status()
            img = Image.open(BytesIO(resp.content)).convert("RGB")
        elif isinstance(img_data, Image.Image):
            # Already a PIL Image
            img = img_data.convert("RGB")
        else:
            # Try to convert from array
            img = Image.fromarray(img_data).convert("RGB")

        # Resize to target size
        img = img.resize(self.image_size, Image.Resampling.LANCZOS)
        return img

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            "image": self.images[idx],
            "caption": self.captions[idx],
        }


# ============================================================
# In-Memory Audio-Text Dataset
# ============================================================

class InMemoryAudioTextDataset(Dataset):
    """
    Pre-loads all audio from MusicCaps (or similar) into memory.

    This avoids repeated loading/resampling during training.
    Audio waveforms are loaded once and kept in memory.
    Uses multiprocessing for faster loading.
    """

    def __init__(
        self,
        hf_dataset,
        audio_col: str = "audio",
        txt_col: str = "caption",
        max_samples: Optional[int] = None,
        target_sr: int = 16000,
        max_duration: float = 30.0,
        num_workers: Optional[int] = None,
    ):
        """
        Args:
            hf_dataset: HuggingFace dataset object
            audio_col: Column name for audio data
            txt_col: Column name for text captions
            max_samples: Limit number of samples (None = all)
            target_sr: Target sample rate for audio
            max_duration: Maximum audio duration in seconds
            num_workers: Number of parallel workers (None = auto-detect)
        """
        self.audio_col = audio_col
        self.txt_col = txt_col
        self.target_sr = target_sr
        self.max_duration = max_duration
        self.max_length = int(target_sr * max_duration)

        # Limit dataset size if requested
        if max_samples is not None:
            hf_dataset = hf_dataset.select(range(min(max_samples, len(hf_dataset))))

        # Auto-detect number of workers
        if num_workers is None:
            num_workers = min(cpu_count(), 16)  # Audio processing can be CPU intensive

        print(f"\n🎵 Pre-loading {len(hf_dataset)} audio samples into memory...")
        print(f"   Target SR: {target_sr} Hz")
        print(f"   Max duration: {max_duration}s")
        print(f"   Using {num_workers} parallel workers")

        # Pre-load all audio and captions using multiprocessing
        self.audio_waveforms = []
        self.captions = []
        self.failed_indices = []

        # Prepare data for parallel processing
        items_data = [
            (idx, hf_dataset[idx][audio_col], hf_dataset[idx][txt_col], target_sr, self.max_length)
            for idx in range(len(hf_dataset))
        ]

        # Use multiprocessing pool to load audio in parallel
        with Pool(processes=num_workers) as pool:
            results = list(
                tqdm(
                    pool.imap(_load_audio_worker, items_data, chunksize=max(1, len(items_data) // (num_workers * 4))),
                    total=len(items_data),
                    desc="Loading audio"
                )
            )

        # Process results
        for idx, waveform, caption in results:
            if waveform is None:
                # Use silent audio as fallback
                waveform = np.zeros(target_sr * 3, dtype=np.float32)
                self.failed_indices.append(idx)

            self.audio_waveforms.append(waveform)
            self.captions.append(caption)

        print(f"✅ Loaded {len(self.audio_waveforms)} audio samples into memory")
        if self.failed_indices:
            print(f"   ⚠️  {len(self.failed_indices)} audio samples failed to load (using fallback)")

    def _load_audio(self, audio_item) -> np.ndarray:
        """
        Load and resample audio.

        Returns waveform as numpy array at target sample rate.
        """
        # If audio is already loaded (HuggingFace datasets format)
        if isinstance(audio_item, dict) and "array" in audio_item:
            waveform = audio_item["array"]
            orig_sr = audio_item.get("sampling_rate", self.target_sr)

            # Resample if needed
            if orig_sr != self.target_sr and HAS_LIBROSA:
                waveform = librosa.resample(
                    waveform,
                    orig_sr=orig_sr,
                    target_sr=self.target_sr
                )

            # Truncate or pad to max_length
            if len(waveform) > self.max_length:
                waveform = waveform[:self.max_length]
            elif len(waveform) < self.max_length:
                waveform = np.pad(waveform, (0, self.max_length - len(waveform)))

            return waveform.astype(np.float32)

        # If audio_item is a numpy array directly
        elif isinstance(audio_item, np.ndarray):
            waveform = audio_item.astype(np.float32)

            # Truncate or pad
            if len(waveform) > self.max_length:
                waveform = waveform[:self.max_length]
            elif len(waveform) < self.max_length:
                waveform = np.pad(waveform, (0, self.max_length - len(waveform)))

            return waveform

        # Unknown format
        else:
            raise ValueError(f"Unknown audio format: {type(audio_item)}")

    def __len__(self) -> int:
        return len(self.audio_waveforms)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            "audio": self.audio_waveforms[idx],
            "caption": self.captions[idx],
        }


# ============================================================
# Collate Functions
# ============================================================

def collate_in_memory_images(batch: List[Dict]) -> Dict[str, Any]:
    """
    Collate function for InMemoryImageTextDataset.

    Returns:
        {
            "images": List[PIL.Image]
            "captions": List[str]
        }
    """
    return {
        "images": [item["image"] for item in batch],
        "captions": [item["caption"] for item in batch],
    }


def collate_in_memory_audio(batch: List[Dict]) -> Dict[str, Any]:
    """
    Collate function for InMemoryAudioTextDataset.

    Returns:
        {
            "audio": List[np.ndarray]
            "captions": List[str]
        }
    """
    return {
        "audio": [item["audio"] for item in batch],
        "captions": [item["caption"] for item in batch],
    }
