
import argparse
import os
import sys
import torch
from pathlib import Path
from PIL import Image

# Disable torch compilation for inference (safe default)
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCH_DYNAMO_DISABLE"] = "1"
import torch._dynamo
try:
    torch._dynamo.config.suppress_errors = True
    torch._dynamo.config.disable = True
except:
    pass

# Add src to path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))
sys.path.insert(0, str(project_root))

try:
    from config import load_config
    from models.alignment import MultimodalAlignmentModel
    from models.trm_qwen_vlm import QwenVLM
    from decoders.qwen import QwenDecoder
    from data.transforms import get_image_transforms
except ImportError as e:
    print(f"Error importing local modules: {e}")
    raise

def parse_args():
    parser = argparse.ArgumentParser(description="Run VLM Inference")
    parser.add_argument("--image_path", type=str, required=True, help="Path to input image")
    parser.add_argument("--question", type=str, required=True, help="Question text")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to VLM checkpoint, or 'latest'/'best' to auto-resolve")
    parser.add_argument("--config", type=str, default=str(project_root / "configs/trm_vlm_qa_qwen1.5.yaml"), help="Path to config file")
    parser.add_argument("--alignment_config", type=str, default=str(project_root / "configs/pixmo_alignment.yaml"), help="Path to alignment config")
    parser.add_argument("--alignment_checkpoint", type=str, default="notebooks/checkpoints/pixmo_alignment/checkpoint_best.pt", help="Path to alignment checkpoint")
    parser.add_argument("--use_trm", action="store_true", help="Enable TRM recursion")
    return parser.parse_args()

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load configs
    config = load_config(args.config)
    alignment_config = load_config(args.alignment_config)
    alignment_config.decoder = None
    alignment_config.text_encoder = None
    
    # 1. Load Aligned Vision Encoder
    print("Loading vision encoder...")
    aligned_model = MultimodalAlignmentModel(alignment_config)
    
    if os.path.exists(args.alignment_checkpoint):
         ckpt = torch.load(args.alignment_checkpoint, map_location='cpu', weights_only=False)
         aligned_model.load_state_dict(ckpt['model_state_dict'], strict=False)
    else:
         print(f"Warning: Alignment checkpoint not found at {args.alignment_checkpoint}")

    aligned_model.eval().to(device)
    
    # 2. Load VLM
    print("Loading VLM...")
    qwen_decoder = QwenDecoder(
        model_name=config.decoder.model_name,
        load_in_8bit=config.decoder.load_in_8bit,
        load_in_4bit=config.decoder.load_in_4bit,
        use_lora=config.decoder.use_lora,
        lora_r=config.decoder.lora_r,
        lora_alpha=config.decoder.lora_alpha,
        lora_dropout=config.decoder.lora_dropout,
        lora_target_modules=config.decoder.lora_target_modules,
        device_map="auto", # Use auto for inference
        num_key_value_heads=getattr(config.decoder, "num_key_value_heads", None),
        intermediate_size=getattr(config.decoder, "intermediate_size", None),
    )
    
    vision_token_dim = alignment_config.vision_encoder.projection_dim
    model = QwenVLM(
        qwen_decoder=qwen_decoder,
        vision_token_dim=vision_token_dim,
        use_trm_recursion=args.use_trm,
        num_trm_layers=4,
        num_recursion_steps=4
    ) # .to(device) -> Do not force device if Qwen uses device_map="auto"
    
    # If device_map is auto, the model parameters are already placed. 
    # Moving QwenVLM might be redundant or conflicting if it tries to move the decoder.
    # But QwenVLM has extra layers (vision_proj, etc). We need them on the same device as Qwen's first layer/input.
    
    # Detect device
    if hasattr(model.qwen.model, "device"):
        model_device = model.qwen.model.device 
    else:
        # Fallback
        model_device = next(model.parameters()).device
        
    print(f"Model is on device: {model_device}")
    
    # Move extra components if needed (though QwenVLM move might have handled it if we called it?)
    # Let's simple call model.to(model_device) to ensure non-HF parts are aligned, assuming single GPU for now if 'auto' picked one.
    # If distributed/sharded, this might be complex. 
    # For inference, 'auto' usually fits on one GPU if possible.
    
    # Actually, simpler: Determine execution device from the model and move inputs there.
    # We should ensure QwenVLM's own parameters (vision_proj) are on the correct device.
    model.to(model_device)

    
    # Load Checkpoint
    checkpoint_path = args.checkpoint_path
    if checkpoint_path.lower() in ["latest", "best"]:
        # Resolve from config output dir
        output_dir = Path(config.trainer.output_dir)
        if not output_dir.exists():
             # Fallback to defaults or try relative
             # Config usually has relative path like ./outputs/... which depends on cwd
             # Let's try to resolve it relative to project root if it feels relative
             if str(output_dir).startswith("./"):
                 output_dir = project_root / str(output_dir).replace("./", "")
        
        if not output_dir.exists():
             raise FileNotFoundError(f"Cannot resolve output directory {output_dir} for auto-checkpoint loading")
             
        if checkpoint_path.lower() == "latest":
             all_checkpoints = sorted(output_dir.glob("checkpoint-epoch-*"), key=lambda p: int(p.name.split('-')[-1]))
             if all_checkpoints:
                 checkpoint_path = str(all_checkpoints[-1])
                 print(f"Auto-resolved 'latest' checkpoint to: {checkpoint_path}")
             else:
                 print("No epoch checkpoints found, checking for checkpoint_best...")
                 checkpoint_path = str(output_dir / "checkpoint_best")
        else: # best
             checkpoint_path = str(output_dir / "checkpoint_best")
             
        if not os.path.exists(checkpoint_path):
             # Try adding .pt or checking directory
              pass # Let the loader fail or handle it

    print(f"Loading checkpoint from {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    # 3. Process Input
    print(f"Processing image: {args.image_path}")
    image = Image.open(args.image_path).convert('RGB')
    transform = get_image_transforms(config.dataset.image_size, is_training=False)
    transform = get_image_transforms(config.dataset.image_size, is_training=False)
    # Move to model device
    image_tensor = transform(image).unsqueeze(0).to(model_device)
    
    # Encode vision
    with torch.no_grad():
        # Aligned model might be on different device?
        # In this script we did: aligned_model.eval().to(device) where device was 'cuda'.
        # If model_device is different (unlikely if single GPU), we need to handle.
        # Let's assume 'device' (from availability) is effectively where 'auto' put things or similar.
        # But specifically:
        
        vision_out = aligned_model.vision_encoder(image_tensor.to(aligned_model.vision_encoder.device), return_sequence=True)
        vision_tokens = vision_out.sequence
        
    # Prepare question
    question = args.question
    inputs = qwen_decoder.tokenizer([question], return_tensors='pt', padding=True).to(model_device)

    
    # Generate
    print(f"Question: {question}")
    with torch.no_grad():
        gen_ids = model.generate(
            vision_tokens=vision_tokens,
            question_ids=inputs.input_ids,
            max_new_tokens=128,
            temperature=0.2 # Slight temp for variety
        )
        
    answer = qwen_decoder.tokenizer.decode(gen_ids[0], skip_special_tokens=True)
    print("\n" + "="*40)
    print(f"Answer: {answer}")
    print("="*40)

if __name__ == "__main__":
    main()
