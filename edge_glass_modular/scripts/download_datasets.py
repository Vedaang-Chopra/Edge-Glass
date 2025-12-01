"""Download datasets for training."""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.downloader import (
    download_pixmo_subset,
    download_common_voice_subset,
    download_instruction_dataset,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Download datasets for Edge Glass training")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data",
        help="Output directory for datasets",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=20000,
        help="Number of samples to download",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=32,
        help="Number of parallel workers for downloading",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["pixmo", "common_voice", "instructions"],
        choices=["pixmo", "common_voice", "instructions", "all"],
        help="Which datasets to download",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets_to_download = args.datasets
    if "all" in datasets_to_download:
        datasets_to_download = ["pixmo", "common_voice", "instructions"]

    # Download PixMo-Cap
    if "pixmo" in datasets_to_download:
        logger.info("=" * 60)
        logger.info("Downloading PixMo-Cap dataset...")
        logger.info("=" * 60)
        try:
            metadata_path = download_pixmo_subset(
                output_dir=output_dir / "pixmo",
                num_samples=args.num_samples,
                num_workers=args.num_workers,
            )
            logger.info(f"✓ PixMo-Cap downloaded successfully: {metadata_path}")
        except Exception as e:
            logger.error(f"✗ Failed to download PixMo-Cap: {e}")

    # Download Common Voice
    if "common_voice" in datasets_to_download:
        logger.info("=" * 60)
        logger.info("Downloading Common Voice dataset...")
        logger.info("=" * 60)
        try:
            metadata_path = download_common_voice_subset(
                output_dir=output_dir / "common_voice",
                num_samples=args.num_samples,
                language="en",
            )
            logger.info(f"✓ Common Voice downloaded successfully: {metadata_path}")
        except Exception as e:
            logger.error(f"✗ Failed to download Common Voice: {e}")

    # Download instruction dataset
    if "instructions" in datasets_to_download:
        logger.info("=" * 60)
        logger.info("Downloading instruction dataset...")
        logger.info("=" * 60)
        try:
            metadata_path = download_instruction_dataset(
                output_dir=output_dir / "instructions",
                num_samples=args.num_samples * 2,  # More instruction samples
            )
            logger.info(f"✓ Instructions downloaded successfully: {metadata_path}")
        except Exception as e:
            logger.error(f"✗ Failed to download instructions: {e}")

    logger.info("=" * 60)
    logger.info("All downloads complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
