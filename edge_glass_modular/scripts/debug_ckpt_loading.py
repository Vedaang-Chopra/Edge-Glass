
import os
import torch
import sys
from pathlib import Path
from accelerate import Accelerator

# Add src to path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))
sys.path.insert(0, str(project_root))

from config import load_config
from models.alignment import MultimodalAlignmentModel
from models.trm_qwen_vlm import QwenVLM
from decoders.qwen import QwenDecoder

def main():
    config_path = project_root / "configs/trm_vlm_qa_qwen2.5-3b_regularized.yaml"
    print(f"Loading config from {config_path}")
    config = load_config(str(config_path))
    
    print("Initializing model...")
    # Initialize components
    alignment_config = load_config(str(project_root / "configs/pixmo_alignment.yaml"))
    
    # Simulate the fix: Use explicit device map
    device_map = {"": "cuda:0"} # Force single device for debug
    
    qwen_decoder = QwenDecoder(
        model_name=config.decoder.model_name,
        load_in_8bit=config.decoder.load_in_8bit,
        load_in_4bit=config.decoder.load_in_4bit,
        use_lora=config.decoder.use_lora,
        lora_r=config.decoder.lora_r,
        lora_alpha=config.decoder.lora_alpha,
        lora_dropout=config.decoder.lora_dropout,
        device_map=device_map 
    )
    
    model = QwenVLM(
        qwen_decoder=qwen_decoder,
        vision_token_dim=alignment_config.vision_encoder.projection_dim,
        use_trm_recursion=True,
        num_trm_layers=4 # Match training script
    )
    
    print("Model initialized.")
    model_keys = set(model.state_dict().keys())
    print(f"Model has {len(model_keys)} keys.")
    
    # Find latest checkpoint
    output_dir = Path(config.trainer.output_dir)
    print(f"Scanning {output_dir}")
    all_checkpoints = sorted(output_dir.glob("checkpoint-epoch-*"), key=lambda p: int(p.name.split('-')[-1]), reverse=True)
    
    valid_ckpt = None
    for ckpt in all_checkpoints:
        if (ckpt / "scheduler.bin").exists():
            valid_ckpt = ckpt
            break
            
    if not valid_ckpt:
        print("No valid checkpoint found.")
        return

    print(f"Analyzing checkpoint: {valid_ckpt}")
    
    # Load checkpoint
    # In accelerate, save_state saves a folder.
    # We need to find the pytorch_model.bin or similar inside.
    # The output shows 'pytorch_model/mp_rank_00_model_states.pt'
    
    model_file = valid_ckpt / "pytorch_model" / "mp_rank_00_model_states.pt"
    if not model_file.exists():
         print(f"Model file not found at {model_file}")
         # Maybe it's a bin file?
         model_file = list((valid_ckpt / "pytorch_model").glob("*.bin"))
         if not model_file:
             print("No model file found.")
             return
         model_file = model_file[0]

    print(f"Loading state dict from {model_file}")
    state_dict = torch.load(model_file, map_location="cpu")
    if "module" in state_dict:
        state_dict = state_dict["module"]
        
    ckpt_keys = set(state_dict.keys())
    print(f"Checkpoint has {len(ckpt_keys)} keys.")
    
    missing = model_keys - ckpt_keys
    unexpected = ckpt_keys - model_keys
    
    print(f"Missing keys: {len(missing)}")
    if missing:
        print("Example missing:", list(missing)[:5])
        
    print(f"Unexpected keys: {len(unexpected)}")
    if unexpected:
        print("Example unexpected:", list(unexpected)[:5])
        
    # Check specifically for bitsandbytes
    bnb_keys = [k for k in unexpected if "bitsandbytes" in k]
    print(f"Unexpected bitsandbytes keys: {len(bnb_keys)}")
    
if __name__ == "__main__":
    main()
