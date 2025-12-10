import os
import sys
import argparse
import random
import numpy as np

# Disable torch compilation features to prevent backend compiler errors (cudagraphs)
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCH_DYNAMO_DISABLE"] = "1"

import torch
import torch._dynamo
# Attempt to disable dynamo optimization if implicit
try:
    torch._dynamo.config.suppress_errors = True
    torch._dynamo.config.disable = True
except:
    pass

import math
import time
import wandb
from pathlib import Path
from tqdm.auto import tqdm
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from accelerate import DeepSpeedPlugin
import logging
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings("ignore") 

# Add src to path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))
sys.path.insert(0, str(project_root))

# Local imports
try:
    from config import load_config
    from models.alignment import MultimodalAlignmentModel
    from models.trm_qwen_vlm import QwenVLM
    from decoders.qwen import QwenDecoder
    from decoders.trm import TRMConfig
    from data.dataset_builder import PixmoQADataset
    from data.transforms import get_image_transforms
    from evaluation.qa_metrics import compute_bleu, compute_rouge_l, normalize_answer
except ImportError as e:
    print(f"Error importing local modules: {e}")
    print(f"sys.path: {sys.path}")
    raise

logger = get_logger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Train VLM with Accelerate")
    parser.add_argument("--config", type=str, default=str(project_root / "configs/trm_vlm_qa_qwen1.5.yaml"), help="Path to config file")
    parser.add_argument("--alignment_config", type=str, default=str(project_root / "configs/pixmo_alignment.yaml"), help="Path to alignment config")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (overrides config)")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size per device")
    parser.add_argument("--learning_rate", type=float, default=None, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of data loader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to alignment checkpoint")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Resume training from checkpoint")
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases")
    parser.add_argument("--max_steps", type=int, default=None, help="Max training steps (for debugging)")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode (fewer steps, more logging)")
    parser.add_argument("--deepspeed_stage", type=int, default=2, choices=[0, 1, 2, 3], help="ZeRO stage for DeepSpeed")
    parser.add_argument("--use_trm", action="store_true", help="Enable TRM recursion (overrides default True)")
    parser.add_argument("--disable_trm", action="store_true", help="Disable TRM recursion")
    parser.add_argument("--run_name", type=str, default=None, help="WandB run name (overrides config)")
    return parser.parse_args()

def get_lr_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def collate_fn(batch, tokenizer):
    from torch.nn.utils.rnn import pad_sequence
    pad_idx = tokenizer.pad_token_id
    
    # Filter None items if any
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    # Stack images
    images = torch.stack([b['image'] for b in batch])
    
    # Pad sequences
    q_padded = pad_sequence([b['question_ids'] for b in batch], batch_first=True, padding_value=pad_idx)
    a_padded = pad_sequence([b['answer_ids'] for b in batch], batch_first=True, padding_value=pad_idx)
    
    # Create attention masks (1 for real tokens, 0 for pad)
    # Note: The notebook logic for answer_mask was specific: 
    # "Create answer mask (1 for valid, 0 for pad)"
    # Replicating notebook collate logic more faithfully:
    
    bs = len(batch)
    max_a_len = a_padded.size(1)
    answer_mask = torch.zeros((bs, max_a_len), dtype=torch.long)
    
    for i in range(bs):
        a_len = batch[i]['answer_ids'].size(0)
        answer_mask[i, :a_len] = 1
        
    return {
        'images': images,
        'question_ids': q_padded,
        'answer_ids': a_padded,
        'answer_mask': answer_mask,
        'answers': [b['answer'] for b in batch],
    }

def main():
    args = parse_args()
    config = load_config(args.config)
    
    # Override config with args
    if args.output_dir:
        config.trainer.output_dir = args.output_dir
    if args.batch_size:
        config.dataset.batch_size = args.batch_size
    if args.learning_rate:
        config.optimization.learning_rate = args.learning_rate
    if args.num_epochs:
        config.trainer.num_epochs = args.num_epochs
    if args.run_name:
        config.trainer.wandb_run_name = args.run_name
        
    # Honor CLI overrides for TRM toggle
    use_trm_recursion = True
    if args.disable_trm:
        use_trm_recursion = False
    if args.use_trm:
        use_trm_recursion = True
    
    # Setup Accelerator
    # Note: Older accelerate versions don't accept bf16/fp16 kwargs on DeepSpeedPlugin
    ds_plugin = DeepSpeedPlugin(
        zero_stage=args.deepspeed_stage,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_clipping=config.optimization.max_grad_norm,
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=config.optimization.mixed_precision,
        log_with="wandb" if args.use_wandb or config.trainer.use_wandb else None,
        deepspeed_plugin=ds_plugin,
        project_config=ProjectConfiguration(
            project_dir=config.trainer.output_dir, 
            logging_dir=str(Path(config.trainer.output_dir) / "logs")
        )
    )
    
    # Ensure sentence-transformers loads on the accelerator device (GPU if available)
    os.environ["SENTENCE_TRANSFORMERS_DEFAULT_DEVICE"] = accelerator.device.type
    
    # Logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    
    # Set seed
    set_seed(args.seed)
    
    # Create output dir
    if accelerator.is_main_process:
        os.makedirs(config.trainer.output_dir, exist_ok=True)
        # Init WandB
        if args.use_wandb or config.trainer.use_wandb:
            accelerator.init_trackers(
                project_name=config.trainer.wandb_project,
                config=vars(args), # Upload args
                init_kwargs={"wandb": {"name": config.trainer.wandb_run_name}}
            )

    # 1. Load Aligned Vision Encoder (Frozen)
    logger.info("Loading aligned vision encoder...")
    alignment_config = load_config(args.alignment_config)
    
    # Optimization: Disable decoder and text encoder in alignment config to save memory
    # We only need the vision encoder part
    alignment_config.decoder = None
    alignment_config.text_encoder = None
    
    aligned_model = MultimodalAlignmentModel(alignment_config)
    
    # Load checkpoint
    # Load checkpoint
    ckpt_root_candidates = [
        Path.cwd() / 'checkpoints',
        Path.cwd().parent / 'checkpoints',
        project_root / 'notebooks/checkpoints', 
        Path('/home/hice1/vchopra37/scratch/projects/edge_glass/edge_glass_modular/notebooks/checkpoints') # Hardcoded fallback
    ]
    
    checkpoint_path = None
    if args.checkpoint_path:
        checkpoint_path = Path(args.checkpoint_path)
    else:
        # Search for the specific file in candidates
        for root in ckpt_root_candidates:
            candidate = root / 'pixmo_alignment/checkpoint_best.pt'
            if candidate.exists():
                checkpoint_path = candidate
                ckpt_root = root
                break
    
    if checkpoint_path is None or not checkpoint_path.exists():
        raise FileNotFoundError(f"Could not find checkpoint at {checkpoint_path} or in candidates: {ckpt_root_candidates}")
        
    logger.info(f"Loading alignment checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=accelerator.device, weights_only=False) # Load to CPU first
    
    # Load state dict with strict=False because we removed decoder/text_encoder
    aligned_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    logger.info("Loaded alignment checkpoint (strict=False)")
    
    # Freeze and set to eval
    aligned_model.eval()
    for param in aligned_model.parameters():
        param.requires_grad = False
        
    # Move only the vision encoder to the accelerator device; keep text encoder on CPU
    aligned_model.vision_encoder.to(accelerator.device)

    # Helper for encoding
    def encode_images(images):
        with torch.no_grad():
            vision_output = aligned_model.vision_encoder(images, return_sequence=True)
            return vision_output.sequence

    # 2. Load Qwen Decoder & VLM
    logger.info(f"Loading Qwen Decoder: {config.decoder.model_name}")
    
    # For 4-bit/8-bit loading with DDP, we strictly need a device_map.
    # We map the entire model to the current accelerator device.
    device_map = None
    logger.info(f"DEBUG: load_in_4bit={config.decoder.load_in_4bit}, load_in_8bit={config.decoder.load_in_8bit}")
    if config.decoder.load_in_4bit or config.decoder.load_in_8bit:
        device_map = {"": accelerator.device}
        logger.info(f"DEBUG: Setting device_map to {device_map}")

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
    
    # Debug model structure
    logger.info(f"DEBUG: QwenDecoder initialized with device_map={device_map}")
    
    vision_token_dim = alignment_config.vision_encoder.projection_dim
    
    # TRM settings
    use_trm = use_trm_recursion
    model = QwenVLM(
        qwen_decoder=qwen_decoder,
        vision_token_dim=vision_token_dim,
        use_trm_recursion=use_trm,
        num_trm_layers=4, # Configurable?
        num_recursion_steps=4,
        confidence_threshold=0.75
    )
    
    # Enable gradient checkpointing if available/needed
    if hasattr(model, "gradient_checkpointing_enable"):
         model.gradient_checkpointing_enable()

    # 3. Data Loaders
    logger.info("Setting up datasets...")
    train_transforms = get_image_transforms(config.dataset.image_size, is_training=True)
    val_transforms = get_image_transforms(config.dataset.image_size, is_training=False)
    
    train_dataset = PixmoQADataset(
        parquet_path=config.dataset.train_parquet,
        tokenizer=qwen_decoder.tokenizer,
        image_transforms=train_transforms,
        max_question_length=128,
        max_answer_length=256,
    )
    
    val_dataset = PixmoQADataset(
        parquet_path=config.dataset.val_parquet,
        tokenizer=qwen_decoder.tokenizer,
        image_transforms=val_transforms,
        max_question_length=128,
        max_answer_length=256,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.dataset.batch_size,
        shuffle=True, # Accelerate handles shuffling generally, but for DDP we need shuffle=True or Sampler
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=lambda b: collate_fn(b, qwen_decoder.tokenizer)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.dataset.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True, 
        collate_fn=lambda b: collate_fn(b, qwen_decoder.tokenizer)
    )

    # 4. Optimization
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.optimization.learning_rate,
        weight_decay=config.optimization.weight_decay
    )
    
    num_update_steps_per_epoch = len(train_loader) // args.gradient_accumulation_steps
    max_train_steps = config.trainer.num_epochs * num_update_steps_per_epoch
    # Approx warmup
    if args.max_steps:
        max_train_steps = args.max_steps
    else:
        max_train_steps = config.trainer.num_epochs * num_update_steps_per_epoch
    
    warmup_steps = int(max_train_steps * config.optimization.warmup_ratio)
    
    scheduler = get_lr_scheduler(optimizer, warmup_steps, max_train_steps)
    
    # Create a wrapper for the model to handle custom forward args if necessary, 
    # but QwenVLM forward signature should work with Accelerate prepare() if compatible.
    # QwenVLM.forward(vision_tokens, images, ...) - wait, Accelerate wraps the model (DDP). 
    # DDP usually expects inputs to forward to match.
    # We'll feed inputs from dict.

    # Prepare with Accelerate
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    # Training Loop
    logger.info("Starting training...")
    global_step = 0
    best_val_loss = float('inf')
    
    # Load checkpoint if resuming
    # Load checkpoint if resuming
    if args.resume_from_checkpoint is None:
        # Auto-detect if we should resume
        output_path = Path(config.trainer.output_dir)
        if output_path.exists():
            # Get all checkpoints sorted by epoch (newest first)
            all_checkpoints = sorted(output_path.glob("checkpoint-epoch-*"), key=lambda p: int(p.name.split('-')[-1]), reverse=True)
            
            for ckpt in all_checkpoints:
                # Validate checkpoint completeness (check for scheduler.bin as a proxy for successful completion)
                if (ckpt / "scheduler.bin").exists():
                    logger.info(f"detected latest VALID checkpoint: {ckpt}")
                    args.resume_from_checkpoint = str(ckpt)
                    break
                else:
                    logger.warning(f"Skipping incomplete/corrupted checkpoint: {ckpt}")

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint == "latest":
             # This block handles the explicit "latest" flag case, apply same robust logic
             output_path = Path(config.trainer.output_dir)
             all_checkpoints = sorted(output_path.glob("checkpoint-epoch-*"), key=lambda p: int(p.name.split('-')[-1]), reverse=True)
             found = False
             for ckpt in all_checkpoints:
                if (ckpt / "scheduler.bin").exists():
                    args.resume_from_checkpoint = str(ckpt)
                    logger.info(f"Auto-detected latest VALID checkpoint for 'latest' flag: {args.resume_from_checkpoint}")
                    found = True
                    break
                else:
                    logger.warning(f"Skipping incomplete checkpoint during 'latest' search: {ckpt}")
            
             if not found:
                logger.warning("No valid checkpoints found for 'latest' resume. Starting from scratch.")
                args.resume_from_checkpoint = None

    start_epoch = 0
    if args.resume_from_checkpoint:
        logger.info(f"Resuming from {args.resume_from_checkpoint} (strict=False)")
        # DeepSpeed uses 'load_module_strict', standard PyTorch uses 'strict'
        load_kwargs = {}
        if accelerator.distributed_type == "DEEPSPEED":
            load_kwargs["load_module_strict"] = False
        else:
            load_kwargs["strict"] = False
            
        accelerator.load_state(args.resume_from_checkpoint, **load_kwargs)
        
        # Parse epoch from checkpoint name
        try:
            ckpt_name = Path(args.resume_from_checkpoint).name
            if "checkpoint-epoch-" in ckpt_name:
                start_epoch = int(ckpt_name.split('-')[-1])
                logger.info(f"Setting start_epoch to {start_epoch}")
        except ValueError:
            logger.warning(f"Could not parse epoch from checkpoint name: {args.resume_from_checkpoint}")

    for epoch in range(start_epoch, config.trainer.num_epochs):
        model.train()
        total_loss = 0
        
        progress_bar = tqdm(train_loader, disable=not accelerator.is_local_main_process, desc=f"Epoch {epoch}")
        
        for batch in progress_bar:
            if batch is None: continue
            
            with accelerator.accumulate(model):
                # Encode images (using frozen encoder)
                # Need to handle device placement carefully for the frozen model
                images = batch['images'].to(accelerator.device) # Ensure images matches frozen model device
                # Note: 'aligned_model' is already on accelerator.device
                
                vision_tokens = encode_images(images)
                
                # Forward
                outputs = model(
                    vision_tokens=vision_tokens,
                    question_ids=batch['question_ids'],
                    answer_ids=batch['answer_ids'],
                    answer_mask=batch['answer_mask']
                )
                
                loss = outputs.loss
                accelerator.backward(loss)
                
                # Gradient clipping and norm
                total_norm = 0.0
                if accelerator.sync_gradients:
                    total_norm = accelerator.clip_grad_norm_(model.parameters(), config.optimization.max_grad_norm)
                    # Handle cases where total_norm might be a tensor
                    if isinstance(total_norm, torch.Tensor):
                        total_norm = total_norm.item()
                    
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
            # Logging
            if accelerator.sync_gradients:
                global_step += 1
                if global_step % config.trainer.log_every == 0 and accelerator.is_main_process:
                    current_loss = loss.item()
                    try:
                        perplexity = math.exp(current_loss)
                    except OverflowError:
                        perplexity = float('inf')
                        
                    lr = scheduler.get_last_lr()[0]
                    
                    log_data = {
                        "train_loss": current_loss,
                        "train_perplexity": perplexity,
                        "grad_norm": total_norm,
                        "lr": lr,
                        "epoch": epoch
                    }
                    accelerator.log(log_data, step=global_step)
                    progress_bar.set_postfix({'loss': f"{current_loss:.4f}", 'ppl': f"{perplexity:.2f}"})
                
                if args.max_steps and global_step >= args.max_steps:
                    logger.info(f"Reached max steps {args.max_steps}, stopping.")
                    # Save debug checkpoint before stopping
                    ckpt_dir = Path(config.trainer.output_dir) / "checkpoint-debug"
                    accelerator.save_state(ckpt_dir)
                    if accelerator.is_main_process:
                        logger.info(f"Saved debug checkpoint to {ckpt_dir}")
                    break
        
        if args.max_steps and global_step >= args.max_steps:
            break
            
        # Validation and Save Best
        model.eval()
        val_loss = 0.0
        val_bleu_total = 0.0
        val_rouge_total = 0.0
        val_metrics_count = 0
        num_batches = 0
        logger.info("Running validation...")
        
        for batch in tqdm(val_loader, disable=not accelerator.is_local_main_process, desc="Validation"):
            if batch is None: continue
            with torch.no_grad():
                 images = batch['images'].to(accelerator.device)
                 vision_tokens = encode_images(images)
                 outputs = model(
                    vision_tokens=vision_tokens,
                    question_ids=batch['question_ids'],
                    answer_ids=batch['answer_ids'],
                     answer_mask=batch['answer_mask']
                 )
                 val_loss += outputs.loss.item()
                 num_batches += 1
                  
                 # ---------------------------------------------------------
                 # Generation Metrics (BLEU/ROUGE) - Run on subset
                 # ---------------------------------------------------------
                 # Only run generation on the first few batches of validation to save time
                 # Step 0 is the first batch
                 if num_batches <= 5: # Evaluate on first 5 batches * batch_size samples
                     try:
                         # We need to unwrap model to access .generate if it's wrapped in DDP
                         unwrapped_model = accelerator.unwrap_model(model)
                         
                         # Access components
                         if hasattr(unwrapped_model, 'module'): # Handle DDP wrapper nesting
                             inner_model = unwrapped_model.module
                         else:
                             inner_model = unwrapped_model
                             
                         # Project vision tokens
                         prefix_embeds = inner_model.vision_proj(vision_tokens)
                         
                         # Generate
                         generated_ids = inner_model.qwen.generate(
                             input_ids=batch['question_ids'],
                             prefix_embeds=prefix_embeds,
                             attention_mask=None,
                             max_new_tokens=64,
                             do_sample=False,
                         )
                         
                         # Decode
                         generated_text = accelerator.unwrap_model(qwen_decoder).tokenizer.batch_decode(
                             generated_ids, skip_special_tokens=True
                         )
                         
                         # Targets
                         target_text = batch['answers']
                         
                         # Compute Metrics
                         for pred, target in zip(generated_text, target_text):
                             b_score = compute_bleu(pred, target)
                             r_score = compute_rouge_l(pred, target)
                             
                             val_bleu_total += b_score
                             val_rouge_total += r_score
                             val_metrics_count += 1
                             
                     except Exception as e:
                         # Use existing logger if available or print
                         pass
        
        # Average loss across batches
        if num_batches > 0:
            avg_val_loss = val_loss / num_batches
            try:
                avg_val_perplexity = math.exp(avg_val_loss)
            except OverflowError:
                avg_val_perplexity = float('inf')
                
            # Average metrics
            avg_val_bleu = val_bleu_total / val_metrics_count if val_metrics_count > 0 else 0.0
            avg_val_rouge = val_rouge_total / val_metrics_count if val_metrics_count > 0 else 0.0
        else:
            avg_val_loss = float('inf')
            avg_val_perplexity = float('inf')
            avg_val_bleu = 0.0
            avg_val_rouge = 0.0
            
        # Log validation loss
        if accelerator.is_main_process:
            logger.info(f"Epoch {epoch} Validation Loss: {avg_val_loss:.4f} | Perplexity: {avg_val_perplexity:.4f} | BLEU: {avg_val_bleu:.2f} | ROUGE: {avg_val_rouge:.2f}")
            accelerator.log({
                "val_loss": avg_val_loss,
                "val_perplexity": avg_val_perplexity,
                "val_bleu": avg_val_bleu,
                "val_rouge": avg_val_rouge
            }, step=global_step)
            
            # Save if best
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                ckpt_dir = Path(config.trainer.output_dir) / "checkpoint_best"
                
                logger.info(f"Starting lightweight checkpoint save to {ckpt_dir}...")
                save_start_time = time.time()
                
                # Lightweight save: just model weights
                if accelerator.distributed_type == "DEEPSPEED":
                     # For DeepSpeed stage 2, this gathers weights to rank 0
                     accelerator.save_model(model, ckpt_dir, safe_serialization=False)
                else:
                     unwrapped_model = accelerator.unwrap_model(model)
                     unwrapped_model.save_pretrained(
                        ckpt_dir,
                        is_main_process=accelerator.is_main_process,
                        save_function=accelerator.save,
                        state_dict=accelerator.get_state_dict(model),
                        safe_serialization=False
                     )
                
                save_duration = time.time() - save_start_time
                logger.info(f"New best checkpoint saved to {ckpt_dir} (Loss: {best_val_loss:.4f}). Time taken: {save_duration:.2f}s")
            else:
                 logger.info(f"Validation loss {avg_val_loss:.4f} did not improve from {best_val_loss:.4f}")

        # Regular massive checkpoint (Full state) - Save every epoch for recovery
        ckpt_dir = Path(config.trainer.output_dir) / f"checkpoint-epoch-{epoch+1}"
        logger.info(f"Saving full state checkpoint to {ckpt_dir}...")
        accelerator.save_state(ckpt_dir)
        
        # Write epoch progress log (for user tracking)
        if accelerator.is_main_process:
            progress_log_path = Path(config.trainer.output_dir) / "training_progress.log"
            with open(progress_log_path, 'a') as f:
                import datetime
                f.write(f"{datetime.datetime.now().isoformat()} | Epoch {epoch+1}/{config.trainer.num_epochs} completed | Val Loss: {avg_val_loss:.4f} | Best: {best_val_loss:.4f}\n")
            logger.info(f"Epoch progress logged to {progress_log_path}")
            
            # Also write a simple state file for easy tracking of last epoch
            state_file_path = Path(config.trainer.output_dir) / "training_state.txt"
            with open(state_file_path, 'w') as f:
                f.write(f"last_epoch: {epoch+1}\n")
                f.write(f"total_epochs: {config.trainer.num_epochs}\n")
                f.write(f"best_val_loss: {best_val_loss:.6f}\n")
                f.write(f"last_val_loss: {avg_val_loss:.6f}\n")
            logger.info(f"Training state saved to {state_file_path}")
        
        # Checkpoint rotation: Keep ONLY BEST and LAST checkpoints to save disk space
        # checkpoint_best is saved separately, so we just keep the very last epoch checkpoint
        import shutil
        all_checkpoints = sorted(Path(config.trainer.output_dir).glob("checkpoint-epoch-*"), key=lambda p: int(p.name.split('-')[-1]))
        
        if len(all_checkpoints) > 1:
            # Delete ALL checkpoints except the very last one (checkpoint_best is separate)
            for old_ckpt in all_checkpoints[:-1]:
                if accelerator.is_main_process:
                    logger.info(f"Deleting old checkpoint {old_ckpt} to save disk space...")
                    try:
                        shutil.rmtree(old_ckpt)
                    except Exception as e:
                        logger.warning(f"Failed to delete {old_ckpt}: {e}")

    # Final save (Full state for resume capability)
    if global_step > 0:
         ckpt_dir = Path(config.trainer.output_dir) / "checkpoint-final"
         logger.info(f"Starting FINAL FULL checkpoint save to {ckpt_dir} (this may take a while)...")
         save_start_time = time.time()
         
         accelerator.save_state(ckpt_dir)
         
         save_duration = time.time() - save_start_time
         if accelerator.is_main_process:
             logger.info(f"Saved final full state checkpoint to {ckpt_dir}. Time taken: {save_duration:.2f}s")

    accelerator.end_training()

if __name__ == "__main__":
    main()
