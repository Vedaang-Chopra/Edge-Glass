import os
import sys
import torch
import torch.nn as nn
from pathlib import Path
from accelerate import Accelerator
import pandas as pd
from PIL import Image
from io import BytesIO
import logging

# Add src to path
current_dir = Path.cwd()
src_path = current_dir / "edge_glass_modular/src"
project_root = current_dir / "edge_glass_modular"
sys.path.insert(0, str(src_path))
sys.path.insert(0, str(project_root))

from config import load_config
from models.alignment import MultimodalAlignmentModel
from models.trm_qwen_vlm import QwenVLM
from decoders.qwen import QwenDecoder
from data.transforms import get_image_transforms

def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Paths
    config_path = project_root / "configs/trm_vlm_qa_qwen1.5.yaml"
    alignment_config_path = project_root / "configs/pixmo_alignment.yaml"
    checkpoint_path = project_root / "outputs/demo_run_3b/checkpoint-debug"
    val_parquet_path = "/home/hice1/vchopra37/scratch/projects/edge_glass/dataset/final_dataset/pixmo_alignment/pixmo_qa_mixed_val.parquet"

    # Configs
    config = load_config(str(config_path))
    alignment_config = load_config(str(alignment_config_path))

    # Accelerator
    accelerator = Accelerator(mixed_precision="bf16")

    # 1. Load Alignment Model
    logger.info("Loading Alignment Model...")
    alignment_config.decoder = None
    alignment_config.text_encoder = None
    aligned_model = MultimodalAlignmentModel(alignment_config)
    
    # Load alignment checkpoint (assuming best exists, otherwise random but we care about QwenVLM loading)
    # Actually, for this debug, we assume alignment is fixed/frozen.
    # The user script loads it from 'checkpoints' or arg.
    # We will skip loading weights if we just want to test QwenVLM's vision_proj linkage, 
    # BUT if aligned_model generates zeros, QwenVLM will see zeros.
    # So we MUST load it if we want real features.
    # Let's try to find it.
    ckpt_path = project_root / "checkpoints/pixmo_alignment/checkpoint_best.pt"
    if ckpt_path.exists():
        logger.info(f"Loading alignment ckpt from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location='cpu')
        aligned_model.load_state_dict(ckpt['model_state_dict'], strict=False)
    else:
        logger.warning(f"Alignment checkpoint not found at {ckpt_path}. Using random init for encoder.")

    aligned_model.eval()
    aligned_model.to(accelerator.device)

    # 2. Load QwenVLM
    logger.info("Loading QwenVLM...")
    qwen_decoder = QwenDecoder(
        model_name=config.decoder.model_name,
        load_in_4bit=config.decoder.load_in_4bit,
        use_lora=config.decoder.use_lora,
    )
    
    vision_token_dim = alignment_config.vision_encoder.projection_dim
    model = QwenVLM(
        qwen_decoder=qwen_decoder,
        vision_token_dim=vision_token_dim,
        use_trm_recursion=False, # Simplify for debug
    )
    
    # Check vision_proj weights BEFORE loading
    logger.info("Checking vision_proj BEFORE loading checkpoint:")
    logger.info(f"Mean: {model.vision_proj.weight.mean().item()}, Std: {model.vision_proj.weight.std().item()}")
    
    # Load Checkpoint
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    # Since checkpoint-debug is a directory (DeepSpeed?), we use accelerator.load_state
    # OR we try to manually load if it's just bin/safetensors.
    # If it is DeepSpeed checkpoint, accelerator.load_state is needed.
    # We need to prepare model first.
    model = accelerator.prepare(model)
    

    print(f"Loading checkpoint from {checkpoint_path}")
    try:
        # Try loading directly from bin file if it exists, as accelerator.load_state is finicky with manual conversions
        possible_paths = [
            os.path.join(checkpoint_path, "pytorch_model.bin"),
            os.path.join(checkpoint_path, "pytorch_model.bin", "pytorch_model.bin")
        ]
        
        bin_path = None
        for p in possible_paths:
            if os.path.exists(p) and os.path.isfile(p):
                bin_path = p
                break
        
        if bin_path:
            print(f"Found pytorch_model.bin at {bin_path}, loading directly...")
            state_dict = torch.load(bin_path, map_location="cpu")
            # Filter prefix if needed or load directly
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            print(f"Loaded state dict. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
            # print(f"Missing keys: {missing[:10]}")
        else:
            accelerator.load_state(checkpoint_path)
            print("Successfully loaded checkpoint via Accelerator.")
    except Exception as e:
        import traceback
        traceback.print_exc()
        logging.error(f"Failed to load via Accelerator: {e}")
        return

    model = accelerator.unwrap_model(model)
    
    # Check vision_proj weights AFTER loading
    logger.info("Checking vision_proj AFTER loading checkpoint:")
    logger.info(f"Mean: {model.vision_proj.weight.mean().item()}, Std: {model.vision_proj.weight.std().item()}")

    # 3. Load Sample
    logger.info("Loading sample...")
    df = pd.read_parquet(val_parquet_path)
    sample = df.iloc[0]
    
    img_bytes = sample['image_bytes']
    image = Image.open(BytesIO(img_bytes)).convert('RGB')
    
    transforms = get_image_transforms(config.dataset.image_size, is_training=False)
    img_tensor = transforms(torch.tensor(np.array(image).transpose(2, 0, 1) / 255.0).float()) # Rough transform logic match
    # Wait, dataset builder uses T.ToTensor() which is 0-1.
    # Just reuse dataset builder logic roughly or use get_image_transforms on tensor
    # Actually, dataset builder:
    # image = read_image... -> float / 255.
    # transforms(image)
    
    # Use explicit transform
    import torchvision.transforms as T
    to_tensor = T.ToTensor()
    img_tensor = to_tensor(image)
    img_tensor = transforms(img_tensor)
    
    img_tensor = img_tensor.unsqueeze(0).to(accelerator.device)
    
    # 4. Encode
    with torch.no_grad():
        vision_output = aligned_model.vision_encoder(img_tensor, return_sequence=True)
        vision_tokens = vision_output.sequence
        logger.info(f"Vision Tokens: {vision_tokens.shape}, Mean: {vision_tokens.mean().item()}, Std: {vision_tokens.std().item()}")
        
        # Project
        vision_emb = model.vision_proj(vision_tokens)
        logger.info(f"Projected Emb: {vision_emb.shape}, Mean: {vision_emb.mean().item()}, Std: {vision_emb.std().item()}")
        
        # 5. Generate
        prompt = "Describe this image."
        inputs = model.qwen.tokenizer([prompt], return_tensors='pt').to(accelerator.device)
        
        logger.info("Generating...")
        gen_out = model.generate(
            vision_tokens=vision_tokens,
            question_ids=inputs.input_ids,
            max_new_tokens=50
        )
        
        decoded = model.qwen.tokenizer.batch_decode(gen_out, skip_special_tokens=True)
        logger.info(f"Output: {decoded[0]}")

if __name__ == "__main__":
    import numpy as np # Missing import
    main()
