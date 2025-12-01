"""Deterministic dataset downloaders for offline experiments."""
from __future__ import annotations

import json
import logging
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
from urllib.parse import urlparse

import requests
import soundfile as sf
from datasets import load_dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)


def _download_file(args):
    url, target = args
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return str(target)
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        target.write_bytes(resp.content)
        return str(target)
    except Exception:
        return None


def download_pixmo_subset(
    output_dir: str,
    num_samples: int = 20_000,
    num_workers: int = 32,
    seed: int = 13,
    samples: int | None = None,
    workers: int | None = None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    effective_num_samples = samples if samples is not None else num_samples
    effective_num_workers = workers if workers is not None else num_workers

    dataset = load_dataset("allenai/pixmo-cap", split="train")
    dataset = dataset.shuffle(seed=seed).select(range(min(effective_num_samples, len(dataset))))
    download_tasks: List[Tuple[str, Path]] = []
    metadata: List[Dict[str, str]] = []
    for idx, row in enumerate(dataset):
        image_url = row["image_url"]
        caption = row.get("caption", "")
        suffix = Path(urlparse(image_url).path).suffix or ".jpg"
        sample_id = f"pixmo_{idx:07d}"
        image_path = output_dir / f"{sample_id}{suffix}"
        download_tasks.append((image_url, image_path))
        metadata.append({"image_path": str(image_path), "caption": caption, "sample_id": sample_id})

    with Pool(processes=effective_num_workers) as pool:
        results = list(
            tqdm(pool.imap_unordered(_download_file, download_tasks), total=len(download_tasks), desc="PixMo images")
        )

    kept = [m for m, status in zip(metadata, results) if status is not None]
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(kept, indent=2))
    logger.info("PixMo subset downloaded: %s samples kept", len(kept))
    return str(metadata_path)


def _save_audio(item):
    waveform = item["audio"]["array"]
    sr = item["audio"]["sampling_rate"]
    target = item["target"]
    target.parent.mkdir(parents=True, exist_ok=True)
    sf.write(target, waveform, sr)
    return str(target)


def download_common_voice_subset(
    output_dir: str,
    num_samples: int = 20_000,
    num_workers: int = 32,
    language: str = "en",
    samples: int | None = None,
    workers: int | None = None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    effective_num_samples = samples if samples is not None else num_samples
    effective_num_workers = workers if workers is not None else num_workers

    dataset = _load_common_voice(language)
    dataset = dataset.shuffle(seed=17).select(range(min(effective_num_samples, len(dataset))))
    metadata: List[Dict[str, str]] = []
    records: List[Dict[str, object]] = []
    for idx, row in enumerate(dataset):
        audio_path = output_dir / f"{language}_{idx}.flac"
        records.append({"audio": row["audio"], "target": audio_path})
        metadata.append(
            {
                "audio_path": str(audio_path),
                "caption": row.get("sentence", ""),
                "sample_id": f"{language}_{idx}",
            }
        )

    with Pool(processes=effective_num_workers) as pool:
        list(tqdm(pool.imap_unordered(_save_audio, records), total=len(records), desc="CommonVoice audio"))

    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))
    logger.info("Common Voice subset downloaded: %s samples kept", len(metadata))
    return str(metadata_path)


def build_instruction_corpus(output_dir: str, pixmo_meta: Iterable[Dict], audiocaps_subset: str = "train"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    audiocaps = load_dataset("microsoft/audiocaps", split=audiocaps_subset)
    openorca = load_dataset("Open-Orca/3_5b", split="train[:50000]")

    samples: List[Dict[str, str]] = []
    for item in pixmo_meta:
        samples.append(
            {
                "instruction": "Describe the visual scene and infer potential user intent.",
                "input": item["caption"],
                "output": item["caption"],
                "modality": "vision_text",
            }
        )
    for row in audiocaps.select(range(20000)):
        samples.append(
            {
                "instruction": "Summarize the audio clip in natural language.",
                "input": row["caption"],
                "output": row["caption"],
                "modality": "audio_text",
            }
        )
    for row in openorca:
        samples.append(
            {
                "instruction": row["system_prompt"],
                "input": row["question"],
                "output": row["response"],
                "modality": "text_only",
            }
        )

    (output_dir / "instruction.json").write_text(json.dumps(samples, indent=2))
    return samples


def download_instruction_dataset(output_dir: str, num_samples: int = 50_000):
    """Download and save a text-only instruction dataset."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset("Open-Orca/OpenOrca", split="train", streaming=True)
    records: List[Dict[str, str]] = []
    for idx, item in enumerate(tqdm(dataset, total=num_samples, desc="Instructions")):
        if idx >= num_samples:
            break
        records.append(
            {
                "instruction": item.get("system_prompt", ""),
                "input": item.get("question", ""),
                "output": item.get("response", ""),
                "sample_id": idx,
            }
        )

    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(records, indent=2))
    logger.info("Instruction dataset prepared: %s samples", len(records))
    return str(metadata_path)


def _load_common_voice(language: str):
    """Try loading Common Voice across recent versions."""
    dataset_versions = [
        "mozilla-foundation/common_voice_17_0",
        "mozilla-foundation/common_voice_13_0",
        "mozilla-foundation/common_voice_16_1",
    ]
    last_err = None
    for name in dataset_versions:
        try:
            logger.info("Loading Common Voice dataset: %s", name)
            return load_dataset(name, language, split="train")
        except Exception as e:
            last_err = e
            logger.warning("Failed to load %s: %s", name, e)
    raise RuntimeError(f"Could not load Common Voice for language={language}") from last_err
