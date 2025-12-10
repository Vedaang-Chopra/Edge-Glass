import os
import sys
import torch
from pathlib import Path
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add src to path
current_dir = Path.cwd()
if str(current_dir).endswith("scripts"):
    root_dir = current_dir.parent
    os.chdir(root_dir)
    sys.path.insert(0, str(root_dir))
else:
    root_dir = current_dir
    sys.path.append(str(root_dir))

from src.config import load_config
from src.models.alignment import MultimodalAlignmentModel
from src.evaluation.benchmark import AlignmentBenchmark
from src.evaluation.zero_shot import ZeroShotClassifier
from src.evaluation.templates import OPENAI_IMAGENET_TEMPLATES
from src.data.dataset_builder import build_image_datasets_from_parquet
from src.data.transforms import get_image_transforms

def main():
    print("="*60)
    print("Freeze-Align Evaluation Script")
    print("="*60)
    
    # 1. Load Configuration
    config_path = root_dir / "configs/pixmo_alignment.yaml"
    print(f"Loading config from {config_path}...")
    config = load_config(str(config_path))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 2. Load Model
    print("Initializing model...")
    model = MultimodalAlignmentModel(config)
    
    # Load Best Checkpoint
    ckpt_path = root_dir / "notebooks/checkpoints/pixmo_alignment/checkpoint_best.pt"
    if not ckpt_path.exists():
        # Fallback
        ckpt_path = root_dir / "outputs/pixmo_alignment/checkpoint_best.pt"
        
    if ckpt_path.exists():
        print(f"Loading checkpoint: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        print("WARNING: No checkpoint found! Evaluating initialized weights.")
        
    model.to(device)
    model.eval()
    
    benchmark = AlignmentBenchmark(model, device=device)
    
    # 3. Zero-Shot Classification Evaluation
    print("\n" + "-"*40)
    print("1. Zero-Shot Classification Benchmarks")
    print("-"*40)
    
    zs_datasets = ["CIFAR10", "CIFAR100"] # Add "ImageNet" if available
    datasets_root = "./data"
    
    # SigLIP expects specific size (usually 336 or 224 depending on specific model variant)
    # Config usually has this.
    image_size = config.dataset.image_size
    print(f"Using image size: {image_size}")
    
    zs_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])
    
    for ds_name in zs_datasets:
        try:
            print(f"Loading {ds_name}...")
            dataset = ZeroShotClassifier.load_dataset(ds_name.lower(), root_dir=datasets_root, split="test", transform=zs_transform)
            loader = DataLoader(dataset, batch_size=64, num_workers=4, shuffle=False)
            
            benchmark.evaluate_zero_shot(
                dataset_name=ds_name,
                class_names=dataset.classes,
                templates=OPENAI_IMAGENET_TEMPLATES,
                dataloader=loader
            )
        except Exception as e:
            print(f"Skipping {ds_name} (Not found or error): {e}")

    # 4. Retrieval Evaluation (Pixmo Validation)
    print("\n" + "-"*40)
    print("2. Pixmo Retrieval Evaluation")
    print("-"*40)
    
    try:
        val_transforms = get_image_transforms(image_size, is_training=False)
        datasets = build_image_datasets_from_parquet(
             cfg=config,
             train_parquet_path=config.dataset.train_parquet,
             val_parquet_path=config.dataset.val_parquet,
             test_parquet_path=config.dataset.test_parquet,
             train_transforms=val_transforms,
             val_transforms=val_transforms,
        )
        val_loader = DataLoader(datasets["val"], batch_size=64, shuffle=False, num_workers=4)
        
        benchmark.run_full_evaluation(val_loader)
        
    except Exception as e:
        print(f"Pixmo Evaluation Failed: {e}")

if __name__ == "__main__":
    main()
