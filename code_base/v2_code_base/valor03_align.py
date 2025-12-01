"""
valor03_align.py

Build a VALOR-32K-style image+caption dataset and save it as a Parquet file.

Assumptions
-----------
- You have an annotations JSON file like:

    [
      {
        "video_id": "x-2Abohj8VY_30.000_40.000",
        "desc": "With the rumble, on a moving bus, a crowd spoke."
      },
      ...
    ]

  where:
    * video_id = "<youtube_id>_<start>_<end>"
    * desc     = audiovisual caption

- You have already downloaded the corresponding video *segments* as .mp4 files,
  named exactly as:

    <video_id>.mp4   (e.g. x-2Abohj8VY_30.000_40.000.mp4)

  and stored them in some directory:  VIDEO_DIR

What this script does
---------------------
For each annotation:
  1. Opens VIDEO_DIR/<video_id>.mp4
  2. Grabs a middle frame (or any representative frame)
  3. Optionally resizes it
  4. Encodes the frame as JPEG bytes
  5. Stores a row in a table with:

     - video_id
     - video_path
     - start (float)
     - end (float)
     - caption (desc)
     - image_jpeg (binary JPEG bytes)

  Then writes everything to a Parquet file.

Usage
-----
    python valor03_align.py \\
        --annotations_json /path/to/desc_train.json \\
        --video_dir /path/to/valor_clips \\
        --output_parquet /path/to/valor32k_images.parquet \\
        --num_workers 8 \\
        --width 224 --height 224

NOTE:
-----
This script only builds image+caption pairs. Later, you can extend it to also
extract audio features or store audio waveforms in the Parquet if desired.
"""

import argparse
import json
import math
import multiprocessing as mp
from functools import partial
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import pandas as pd
from tqdm import tqdm


def parse_video_id(video_id: str) -> Tuple[str, float, float]:
    """Split a VALOR-style video_id into (yt_id, start, end).

    Example: "x-2Abohj8VY_30.000_40.000" -> ("x-2Abohj8VY", 30.0, 40.0)
    """
    parts = video_id.rsplit("_", 2)
    if len(parts) != 3:
        raise ValueError(f"Unexpected video_id format: {video_id}")
    yt_id, start_str, end_str = parts
    return yt_id, float(start_str), float(end_str)


def extract_middle_frame(video_path: Path, resize: Optional[Tuple[int, int]] = None) -> Optional[bytes]:
    """Extract a middle frame from a video and return it as JPEG bytes.

    Returns None if the video can't be opened or encoded.
    """
    if not video_path.exists():
        return None

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        cap.release()
        return None

    middle_idx = frame_count // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, middle_idx)
    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        return None

    if resize is not None:
        w, h = resize
        frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)

    success, buf = cv2.imencode(".jpg", frame)
    if not success:
        return None

    return buf.tobytes()


def _process_single_entry(
    entry: Dict[str, Any],
    video_dir: Path,
    resize: Optional[Tuple[int, int]] = None,
    ignore_missing: bool = True,
) -> Optional[Dict[str, Any]]:
    """Worker function: process one annotation dict into a row for Parquet."""
    video_id = entry.get("video_id")
    caption = entry.get("desc") or entry.get("caption") or ""

    if not video_id:
        return None

    # Assumes file is named <video_id>.mp4
    video_path = video_dir / f"{video_id}.mp4"

    try:
        yt_id, start, end = parse_video_id(video_id)
    except Exception as e:
        if ignore_missing:
            return None
        raise e

    frame_bytes = extract_middle_frame(video_path, resize=resize)
    if frame_bytes is None:
        if ignore_missing:
            return None
        raise RuntimeError(f"Failed to extract frame from: {video_path}")

    return {
        "video_id": video_id,
        "video_path": str(video_path),
        "yt_id": yt_id,
        "start": start,
        "end": end,
        "caption": caption,
        "image_jpeg": frame_bytes,
    }


def build_valor_parquet(
    annotations_json: Path,
    video_dir: Path,
    output_parquet: Path,
    num_workers: int = 8,
    max_samples: Optional[int] = None,
    width: int = 224,
    height: int = 224,
) -> None:
    """Main entry point: build VALOR-32K image+caption Parquet.

    Args:
        annotations_json: Path to annotation JSON file (list of dicts).
        video_dir: Directory containing video segments named <video_id>.mp4.
        output_parquet: Where to save the resulting Parquet file.
        num_workers: Number of processes for multiprocessing.
        max_samples: Optional cap on number of rows to process.
        width, height: Optional resize dimensions for images.
    """
    annotations_json = Path(annotations_json)
    video_dir = Path(video_dir)
    output_parquet = Path(output_parquet)
    output_parquet.parent.mkdir(parents=True, exist_ok=True)

    print(f"Annotations JSON: {annotations_json}")
    print(f"Video directory : {video_dir}")
    print(f"Output Parquet  : {output_parquet}")
    print(f"Workers         : {num_workers}")
    print(f"Resize          : ({width}, {height})")

    with annotations_json.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Annotations JSON must be a list of dicts.")

    if max_samples is not None and max_samples > 0:
        data = data[: max_samples]

    print(f"Total annotations to process: {len(data)}")



    resize = (width, height) if width > 0 and height > 0 else None
    worker = partial(_process_single_entry, video_dir=video_dir, resize=resize, ignore_missing=True)

    results = []
    if num_workers <= 1:
        # Single-process (debug mode)
        for entry in tqdm(data, desc="Processing VALOR entries"):
            row = worker(entry)
            if row is not None:
                results.append(row)
    else:
        # Multiprocessing
        with mp.Pool(processes=num_workers) as pool:
            for row in tqdm(pool.imap(worker, data, chunksize=8), total=len(data), desc="Processing VALOR entries"):
                if row is not None:
                    results.append(row)

    print(f"Valid rows collected: {len(results)}")


    if not results:
        print("No rows were processed successfully. Nothing to save.")
        return

    df = pd.DataFrame(results)
    df.to_parquet(output_parquet, index=False)
    print(f"✅ Saved VALOR image+caption parquet to: {output_parquet}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build VALOR-32K image+caption parquet dataset.")
    parser.add_argument("--annotations_json", type=str, required=True, help="Path to VALOR annotations JSON.")
    parser.add_argument("--video_dir", type=str, required=True, help="Directory with <video_id>.mp4 files.")
    parser.add_argument("--output_parquet", type=str, required=True, help="Path to output Parquet file.")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of worker processes.")
    parser.add_argument("--max_samples", type=int, default=0, help="Optional max samples (0 = all)."
                       )
    parser.add_argument("--width", type=int, default=224, help="Resize width (0 = no resize)."
                       )
    parser.add_argument("--height", type=int, default=224, help="Resize height (0 = no resize)."
                       )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_valor_parquet(
        annotations_json=Path(args.annotations_json),
        video_dir=Path(args.video_dir),
        output_parquet=Path(args.output_parquet),
        num_workers=args.num_workers,
        max_samples=args.max_samples if args.max_samples > 0 else None,
        width=args.width,
        height=args.height,
    )
