
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, CLIPVisionModel, WhisperModel, WhisperProcessor
from datasets import load_dataset
import io
import librosa
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
import random

# Imports from our refactored modules
from imports.dataset import PixmoVisionDataset, LibriSpeechAudioDataset, collate_alignment
from imports.model import MultiModalAlignmentModel
from imports.align_training.steps import AlignmentConfig, AlignmentModules
from imports.align_training.training import train_alignment, build_alignment_optimizer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
class Config:
    # Models
    vision_model_name = "openai/clip-vit-base-patch32"
    audio_model_name = "openai/whisper-base"
    llm_model_name = "Qwen/Qwen2.5-7B-Instruct"
    
    # Dimensions (will be set dynamically based on models)
    encoder_dim_vision = None
    encoder_dim_audio = None
    perceiver_dim = 512
    llm_hidden_size = None  # Will be set from Qwen config
    
    # Perceiver
    num_latents = 64
    
    # MRL Loss
    use_mrl = True
    mrl_dims = (128, 256, 512)
    mrl_temperature = 0.07
    
    # Training
    batch_size_vision = 4  # Small batch size for POC
    batch_size_audio = 4
    num_epochs = 1
    learning_rate = 1e-4
    weight_decay = 0.01
    seed = 42
    log_every_steps = 10
    
    # Data
    librispeech_max_samples = 3000
    max_audio_duration_s = 12.0
    audio_sample_rate = 16000
    vision_max_samples = 2048
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available():
        device = "mps"

cfg = Config()

# ---------------------------------------------------------------------------
# Text Embedding Helper (Qwen)
# ---------------------------------------------------------------------------
class QwenTextEmbedder:
    def __init__(self, model_name, device):
        print(f"Loading Qwen2.5-7B: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map=None, # We handle device manually to avoid conflicts
        ).to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
            
        self.device = device
        
        # Get hidden size
        self.hidden_size = getattr(self.model.config, "hidden_size", None)
        if isinstance(self.hidden_size, (list, tuple)):
            self.hidden_size = self.hidden_size[0]
            
    def encode_text(self, texts: List[str], max_length: int = 64) -> Dict[str, torch.Tensor]:
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(self.device)
        
        # (B, L, D)
        token_embs = self.model.get_input_embeddings()(enc.input_ids)
        
        return {
            "input_ids": enc.input_ids,
            "attention_mask": enc.attention_mask,
            "token_embs": token_embs,
        }

    def pooled_text_embedding(self, texts: List[str], max_length: int = 64) -> torch.Tensor:
        tok_out = self.encode_text(texts, max_length=max_length)
        token_embs = tok_out["token_embs"]          # (B, L, D_llm)
        attn_mask = tok_out["attention_mask"]       # (B, L)

        # masked mean-pooling over tokens
        mask = attn_mask.unsqueeze(-1)             # (B, L, 1)
        denom = mask.sum(dim=1).clamp_min(1)       # (B, 1)
        h_text = (token_embs * mask).sum(dim=1) / denom  # (B, D_llm)
        
        return h_text

# ---------------------------------------------------------------------------
# Data Loading Helpers
# ---------------------------------------------------------------------------
def load_librispeech_subset(cfg):
    print("\nLoading LibriSpeech ASR (streaming mode)...")
    librispeech_raw = load_dataset(
        "openslr/librispeech_asr",
        "all",
        streaming=True,
        split="train.clean.100"
    )
    
    audio_stream = librispeech_raw.decode(False)
    subset = []
    print(f"Taking up to {cfg.librispeech_max_samples} examples...")
    
    for ex in audio_stream:
        subset.append(ex)
        if len(subset) >= cfg.librispeech_max_samples:
            break
            
    # Filter by duration
    print(f"Filtering by duration <= {cfg.max_audio_duration_s}s...")
    filtered = []
    for ex in subset:
        audio_bytes = ex["audio"]["bytes"]
        if audio_bytes is None: continue
        
        # Load wav
        audio_file = io.BytesIO(audio_bytes)
        wav, sr = librosa.load(audio_file, sr=cfg.audio_sample_rate)
        dur = len(wav) / float(sr)
        
        if dur <= cfg.max_audio_duration_s:
            filtered.append({
                "waveform": wav,
                "sampling_rate": sr,
                "duration": dur,
                "text": ex["text"]
            })
            
    print(f"After filtering: {len(filtered)} examples")
    return filtered

def load_pixmo_subset(cfg):
    print("\nLoading PixMo-Cap dataset...")
    pixmo_raw = load_dataset("allenai/pixmo-cap", split="train")
    
    if len(pixmo_raw) > cfg.vision_max_samples:
        pixmo_subset = pixmo_raw.shuffle(seed=cfg.seed).select(range(cfg.vision_max_samples))
    else:
        pixmo_subset = pixmo_raw
        
    print(f"PixMo subset size: {len(pixmo_subset)}")
    return pixmo_subset

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # Set seed
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    
    print(f"Using device: {cfg.device}")
    
    # 1. Initialize Qwen (Text Embedder)
    qwen = QwenTextEmbedder(cfg.llm_model_name, cfg.device)
    cfg.llm_hidden_size = qwen.hidden_size
    print(f"Qwen hidden size: {cfg.llm_hidden_size}")
    
    # 2. Initialize Alignment Model
    print("\nInitializing Alignment Model...")
    model = MultiModalAlignmentModel(
        d_shared=cfg.perceiver_dim,
        d_latent=cfg.perceiver_dim,
        d_align=cfg.llm_hidden_size, # Project to LLM dim
        num_latents=cfg.num_latents,
        vision_model_name=cfg.vision_model_name,
        audio_model_name=cfg.audio_model_name,
        device=torch.device(cfg.device),
        dtype=torch.float32 # Use float32 for stability in POC
    )
    
    # 3. Prepare Datasets
    # Vision
    pixmo_subset = load_pixmo_subset(cfg)
    vision_processor = AutoProcessor.from_pretrained(cfg.vision_model_name)
    vision_ds = PixmoVisionDataset(
        pixmo_subset, 
        model.vision_encoder.model, # Pass the underlying HF model
        vision_processor,
        cfg.device
    )
    
    # Audio
    libri_subset = load_librispeech_subset(cfg)
    audio_processor = WhisperProcessor.from_pretrained(cfg.audio_model_name)
    audio_ds = LibriSpeechAudioDataset(
        libri_subset,
        audio_processor,
        model.audio_encoder.model, # Pass the underlying HF model
        cfg.device
    )
    
    # 4. DataLoaders
    # We need a partial for collate_fn to pass tokenizer
    collate_fn = lambda b: collate_alignment(b, qwen.tokenizer, device=cfg.device)
    
    vision_loader = DataLoader(
        vision_ds, 
        batch_size=cfg.batch_size_vision, 
        shuffle=True, 
        collate_fn=collate_fn
    )
    
    audio_loader = DataLoader(
        audio_ds, 
        batch_size=cfg.batch_size_audio, 
        shuffle=True, 
        collate_fn=collate_fn
    )
    
    train_loaders = {
        "vision": vision_loader,
        "audio": audio_loader
    }
    
    # 5. Setup Training
    # Create AlignmentModules bundle
    modules = AlignmentModules(
        vision_adapter=model.vision_adapter, # Will be None initially, created in forward? 
        # Wait, model.vision_adapter is created lazily in model.encode_vision.
        # But we need to pass it to optimizer. 
        # We should probably force creation or rely on the fact that we are training the whole model?
        # The training loop expects `modules.vision_adapter` etc.
        # But `MultiModalAlignmentModel` initializes them as None.
        # We need to ensure they are initialized.
        # Let's run a dummy forward pass to initialize adapters.
        audio_adapter=None, # placeholder
        perceiver=model.perceiver,
        projector=model.projector
    )
    
    # Force initialization of adapters by running a dummy input
    print("Initializing adapters with dummy forward pass...")
    dummy_img = torch.randn(1, 3, 224, 224).to(cfg.device) # Assuming standard size
    # We need to use the encoder to get features first to know the dim
    with torch.no_grad():
        # Vision
        enc_v = model.vision_encoder.encode_images(torch.zeros(1, 3, 224, 224)) # Dummy tensor
        model._ensure_vision_adapter(enc_v["feats"].shape[-1])
        
        # Audio
        # Dummy audio
        dummy_wav = torch.zeros(1, 16000).to(cfg.device)
        enc_a = model.audio_encoder.encode_waveforms(dummy_wav, sample_rates=16000)
        model._ensure_audio_adapter(enc_a["feats"].shape[-1])
        
    # Now update modules
    modules.vision_adapter = model.vision_adapter
    modules.audio_adapter = model.audio_adapter
    
    # Config for alignment step
    align_cfg = AlignmentConfig(
        mrl_dims=cfg.mrl_dims,
        mrl_temperature=cfg.mrl_temperature,
        max_text_length=64
    )
    
    # Optimizer
    # We want to train adapters, perceiver, projector. Encoders are frozen.
    trainable_params = nn.ModuleList([
        modules.vision_adapter,
        modules.audio_adapter,
        modules.perceiver,
        modules.projector
    ])
    
    optimizer = build_alignment_optimizer(
        trainable_params,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay
    )
    
    # 6. Run Training
    print("\nStarting Training...")
    train_alignment(
        train_loaders=train_loaders,
        modules=modules,
        cfg=align_cfg,
        text_embed_fn=qwen.pooled_text_embedding,
        optimizer=optimizer,
        device=torch.device(cfg.device),
        num_epochs=cfg.num_epochs,
        log_every=cfg.log_every_steps
    )
    
    print("\nTraining Complete!")

if __name__ == "__main__":
    main()
