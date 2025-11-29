"""Deterministic dataset downloaders for offline experiments."""

from __future__ import annotations

import json
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import requests
import soundfile as sf
from datasets import load_dataset
from tqdm import tqdm


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


def download_pixmo_subset(output_dir: str, samples: int = 20_000, workers: int = 32, seed: int = 13):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset("allenai/pixmo-cap", split="train", trust_remote_code=True)
    dataset = dataset.shuffle(seed=seed).select(range(min(samples, len(dataset))))
    download_tasks: List[Tuple[str, Path]] = []
    metadata: List[Dict[str, str]] = []
    for row in dataset:
        image_id = row["image_id"]
        image_url = row["image_url"]
        caption = row["caption"]
        image_path = output_dir / f"{image_id}.jpg"
        download_tasks.append((image_url, image_path))
        metadata.append(
            {
                "image_path": str(image_path),
                "caption": caption,
                "sample_id": image_id,
            }
        )

    with Pool(processes=workers) as pool:
        results = list(
            tqdm(pool.imap_unordered(_download_file, download_tasks), total=len(download_tasks), desc="PixMo images")
        )

    kept = [m for m, status in zip(metadata, results) if status is not None]
    (output_dir / "metadata.json").write_text(json.dumps(kept, indent=2))
    return kept


def _save_audio(item):
    waveform = item["audio"]["array"]
    sr = item["audio"]["sampling_rate"]
    target = item["target"]
    target.parent.mkdir(parents=True, exist_ok=True)
    sf.write(target, waveform, sr)
    return str(target)


def download_common_voice_subset(output_dir: str, samples: int = 20_000, workers: int = 32, language: str = "en"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset("mozilla-foundation/common_voice_16_1", language, split="train")
    dataset = dataset.shuffle(seed=17).select(range(min(samples, len(dataset))))
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

    with Pool(processes=workers) as pool:
        list(tqdm(pool.imap_unordered(_save_audio, records), total=len(records), desc="CommonVoice audio"))

    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return metadata


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
