import argparse
import logging
import os
import sys
import json
import warnings
from pathlib import Path
from tqdm.auto import tqdm
import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed

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
    from data.dataset_builder import PixmoQADataset
    from data.transforms import get_image_transforms
    from evaluation.qa_metrics import evaluate_qa_metrics
except ImportError as e:
    print(f"Error importing local modules: {e}")
    raise

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate VLM with Accelerate")
    parser.add_argument("--config", type=str, default=str(project_root / "configs/trm_vlm_qa_qwen1.5.yaml"), help="Path to config file")
    parser.add_argument("--alignment_config", type=str, default=str(project_root / "configs/pixmo_alignment.yaml"), help="Path to alignment config")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to VLM checkpoint to evaluate")
    parser.add_argument("--output_file", type=str, default="eval_results.json", help="Path to save evaluation results")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size per device")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of data loader workers")
    parser.add_argument("--max_test_samples", type=int, default=None, help="Limit number of test samples")
    parser.add_argument("--use_trm", action="store_true", help="Enable TRM recursion")
    # Add argument to specify alignment checkpoint location if needed, though we can likely infer/default it
    parser.add_argument("--alignment_checkpoint", type=str, default="notebooks/checkpoints/pixmo_alignment/checkpoint_best.pt", help="Path to alignment checkpoint")
    
    return parser.parse_args()

def collate_fn(batch, tokenizer):
    # Filter None
    batch = [b for b in batch if b is not None]
    if not batch: return None
    
    # Just stack images and collect raw text for generation
    images = torch.stack([b['image'] for b in batch])
    
    return {
        'images': images,
        'questions': [b['question'] for b in batch], # Raw text questions
        'answers': [b['answer'] for b in batch],     # Raw text answers
        'image_paths': [b.get('image_path', '') for b in batch],
        'question_ids': [b['question_ids'] for b in batch] # Keep IDs if needed for something, but for gen we use IDs constructed in loop or model
    }

def main():
    args = parse_args()
    config = load_config(args.config)
    
    # Setup Accelerator
    accelerator = Accelerator(mixed_precision="bf16")
    
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)
    
    # 1. Load Aligned Vision Encoder
    logger.info("Loading aligned vision encoder...")
    alignment_config = load_config(args.alignment_config)
    alignment_config.decoder = None # Save memory
    alignment_config.text_encoder = None
    
    aligned_model = MultimodalAlignmentModel(alignment_config)
    
    # Load alignment checkpoint
    # Load alignment checkpoint
    ckpt_root_candidates = [
        Path.cwd() / 'checkpoints',
        Path.cwd().parent / 'checkpoints',
        # Fallback to absolute paths if running from script
        project_root / 'notebooks/checkpoints', 
    ]
    ckpt_root = next((p for p in ckpt_root_candidates if p.exists()), None)
    
    alignment_checkpoint_path = args.alignment_checkpoint
    if not os.path.exists(alignment_checkpoint_path) and ckpt_root:
         # Try to find it in the discovered root
         candidate = ckpt_root / 'pixmo_alignment/checkpoint_best.pt'
         if candidate.exists():
             alignment_checkpoint_path = str(candidate)

    if os.path.exists(alignment_checkpoint_path):
         ckpt = torch.load(alignment_checkpoint_path, map_location='cpu', weights_only=False)
         aligned_model.load_state_dict(ckpt['model_state_dict'], strict=False)
         logger.info(f"Loaded alignment checkpoint from {alignment_checkpoint_path}")
    else:
         logger.warning(f"Alignment checkpoint not found at {alignment_checkpoint_path}, using random init (bad for performance!)")
    
    aligned_model.eval()
    for p in aligned_model.parameters(): p.requires_grad = False
    aligned_model.vision_encoder.to(accelerator.device)
    
    def encode_images(images):
        with torch.no_grad():
            vision_output = aligned_model.vision_encoder(images, return_sequence=True)
            return vision_output.sequence
            
    # 2. Load VLM
    logger.info(f"Loading Qwen Decoder: {config.decoder.model_name}")
    qwen_decoder = QwenDecoder(
        model_name=config.decoder.model_name,
        load_in_4bit=config.decoder.load_in_4bit,
        use_lora=config.decoder.use_lora,
        lora_r=config.decoder.lora_r,
        lora_alpha=config.decoder.lora_alpha,
        lora_dropout=config.decoder.lora_dropout,
        lora_target_modules=config.decoder.lora_target_modules,
        device_map=None,
        num_key_value_heads=getattr(config.decoder, "num_key_value_heads", None),
        intermediate_size=getattr(config.decoder, "intermediate_size", None)
    )
    
    vision_token_dim = alignment_config.vision_encoder.projection_dim
    model = QwenVLM(
        qwen_decoder=qwen_decoder,
        vision_token_dim=vision_token_dim,
        use_trm_recursion=args.use_trm,
        num_trm_layers=4,
        num_recursion_steps=4
    )
    
    # Load VLM checkpoint
    # Prepare with Accelerate (Must create loader first to pass to prepare for DeepSpeed)
    
    # 3. Data Loader
    logger.info("Setting up validation dataset...")
    val_transforms = get_image_transforms(config.dataset.image_size, is_training=False)
    val_dataset = PixmoQADataset(
        parquet_path=config.dataset.val_parquet,
        tokenizer=qwen_decoder.tokenizer,
        image_transforms=val_transforms,
        max_question_length=128,
        max_answer_length=256,
    )
    
    # Handle limit manually
    if args.max_test_samples:
        indices = list(range(min(len(val_dataset), args.max_test_samples)))
        val_dataset = torch.utils.data.Subset(val_dataset, indices)
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda b: collate_fn(b, qwen_decoder.tokenizer)
    )
    
    # Prepare model AND loader together
    model, val_loader = accelerator.prepare(model, val_loader)
    
    # Load VLM checkpoint AFTER prepare (safest for DeepSpeed)
    logger.info(f"Loading VLM checkpoint from {args.checkpoint_path}")
    path = Path(args.checkpoint_path)
    
    if path.is_dir():
        # Check for weight-only checkpoint (fast save)
        has_weights = (path / "model.safetensors").exists() or (path / "pytorch_model.bin").exists()
        
        if has_weights:
            # Load weights manually
            logger.info("Detected weight-only checkpoint. Loading state dict...")
            from accelerate.utils import load_checkpoint_in_model
            # Use accelerate's utility which handles offloading if needed
            load_checkpoint_in_model(accelerator.unwrap_model(model), args.checkpoint_path)
            logger.info("Loaded weight-only checkpoint successfully.")
        else:
            # Assume full accelerate/deepspeed state
            try:
                accelerator.load_state(args.checkpoint_path)
                logger.info("Loaded full accelerator state.")
            except Exception as e:
                logger.error(f"Failed to load state from directory: {e}")
                raise
    else:
        # Assume pt file (standard torch save)
        state_dict = torch.load(path, map_location='cpu')
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        accelerator.unwrap_model(model).load_state_dict(state_dict, strict=False)
        logger.info("VLM Checkpoint loaded via torch.load")
    
    # 4. Inference Loop
    all_predictions = []
    all_targets = []
    
    logger.info("Starting inference...")
    for batch in tqdm(val_loader, disable=not accelerator.is_local_main_process):
        if batch is None: continue
        
        images = batch['images'] # Alrdy on device via prepare? or no bc custom collate? 
        # Accelerate prepare on loader usually handles device placement for tensors in dict/list
        
        with torch.no_grad():
            vision_tokens = encode_images(images)
            
            # Generate
            # We need to construct inputs for generate. 
            # QwenVLM.generate takes vision_tokens and question_ids/prompts
            
            # Since we have raw questions, let's re-tokenize or use question_ids from batch
            # But the model.generate expects specific format. 
            # Let's inspect model.generate signature or usage. 
            # It likely needs raw text or input_ids.
            # "question_ids" in batch are tokenized questions.
            
            # We need to pad question_ids in current batch if using batch_gen
            # But let's assume batch_size=1 for safety in eval or implement padding.
            
            # For simplicity, loop through batch (if small) or pass padded.
            # The collate_fn above didn't pad query ids. 
            
            # Let's rely on questions text and re-tokenize with padding on device OR use simple loop
            
            prompts = batch['questions']
            
            # Use the underlying generate method
            # QwenVLM.generate(vision_tokens, question_ids, ...)
            
            # Tokenize prompts
            inputs = qwen_decoder.tokenizer(prompts, return_tensors='pt', padding=True, truncation=True).to(accelerator.device)
            
            # Generate
            # Unwrap model if needed
            unwrapped_model = accelerator.unwrap_model(model)
            
            gen_out = unwrapped_model.generate(
                vision_tokens=vision_tokens,
                question_ids=inputs.input_ids,
                max_new_tokens=128,
                temperature=0.0 # Greedy
            )
            
            # Decode
            # gen_out could be list of ids or something.
            # Assuming logic similar to notebook, it returns generated_ids
            
            preds = qwen_decoder.tokenizer.batch_decode(gen_out, skip_special_tokens=True)
            
            # Gather results from all processes? 
            # Accelerate gather_for_metrics is useful here
            
            # Simple gather manually
            # We need to gather full lists. 
            
            all_predictions.extend(preds)
            all_targets.extend(batch['answers'])

    # Sync
    # For robust gathering, we should serialize and gather string lists.
    # Accelerate gather is for tensors. 
    # Workaround: Save partial results to disk per rank and merge, or just print metrics per rank if lazy.
    # Correct way: encode strings to tensor if needed, or use gather_object (if available in newer accelerate)
    
    # Let's try gather_object if available, else save to json per rank
    if hasattr(accelerator, 'gather_object'):
        all_predictions = accelerator.gather_object(all_predictions)
        all_targets = accelerator.gather_object(all_targets)
    
    if accelerator.is_main_process:
        # Compute metrics
        metrics = evaluate_qa_metrics(all_predictions, all_targets)
        print("\n" + "="*40)
        print(f"Evaluation Results (N={len(all_predictions)})")
        print("="*40)
        print(f"Exact Match: {metrics['exact_match']:.2f}%")
        print(f"F1 Score:    {metrics['f1']:.2f}%")
        print("="*40)
        
        # Save results
        results = [
            {"prediction": p, "target": t}
            for p, t in zip(all_predictions, all_targets)
        ]
        with open(args.output_file, 'w') as f:
            json.dump({"metrics": metrics, "samples": results}, f, indent=2)
        logger.info(f"Results saved to {args.output_file}")

if __name__ == "__main__":
    main()
