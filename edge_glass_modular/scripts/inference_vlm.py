
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
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to VLM checkpoint")
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
        load_in_4bit=config.decoder.load_in_4bit,
        use_lora=config.decoder.use_lora,
        device_map="auto" # Use auto for inference
    )
    
    vision_token_dim = alignment_config.vision_encoder.projection_dim
    model = QwenVLM(
        qwen_decoder=qwen_decoder,
        vision_token_dim=vision_token_dim,
        use_trm_recursion=args.use_trm,
        num_trm_layers=4,
        num_recursion_steps=4
    ).to(device)
    
    # Load Checkpoint
    print(f"Loading checkpoint from {args.checkpoint_path}")
    state_dict = torch.load(args.checkpoint_path, map_location='cpu')
    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    # 3. Process Input
    print(f"Processing image: {args.image_path}")
    image = Image.open(args.image_path).convert('RGB')
    transform = get_image_transforms(config.dataset.image_size, is_training=False)
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Encode vision
    with torch.no_grad():
        vision_out = aligned_model.vision_encoder(image_tensor, return_sequence=True)
        vision_tokens = vision_out.sequence
        
    # Prepare question
    question = args.question
    inputs = qwen_decoder.tokenizer([question], return_tensors='pt', padding=True).to(device)
    
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
