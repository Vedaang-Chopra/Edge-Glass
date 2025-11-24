### Part 0 – Imports, config, and utilities
# ============================================
# Part 0 – Imports, config, and utilities
# ============================================

import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import torch
import torch.nn as nn
import numpy as np

from datasets import load_dataset, Audio
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModel,
    WhisperProcessor,
    WhisperModel,
)


# ---- Device & dtype ----
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# prefer bfloat16 on newer GPUs, else float16
if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
    default_dtype = torch.bfloat16
else:
    default_dtype = torch.float16

torch.set_default_dtype(default_dtype)

print("Device:", device)
print("Default dtype:", default_dtype)


# ---- Repro utilities ----
def set_seed(seed: int = 42):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



# ============================================
# Part 0.1 – Global config (OPTIMIZED)
# ============================================
from accelerate import Accelerator
from accelerate.utils import set_seed



@dataclass
class Config:
    # --- Model names ---
    vision_model_name: str = "openai/clip-vit-base-patch32"
    audio_model_name: str = "openai/whisper-base"
    audio_sample_rate: int = 16000
    llm_model_name: str = "Qwen/Qwen2.5-7B-Instruct"

    # --- Dimensions ---
    encoder_dim_vision: int = 768     # CLIP-base dim
    encoder_dim_audio: int = 512      # Whisper-base dim
    llm_hidden_size: int = 3584       # Qwen 7B dim
    
    # === IMPROVEMENT 1: Model Capacity ===
    perceiver_dim: int = 768          # Increased from 512 to match Vision
    num_latents: int = 64
    num_perceiver_layers: int = 6     # Increased from 2 to 6 (Deeper logic)
    num_attn_heads: int = 8
    mlp_ratio: float = 4.0

    # --- Matryoshka loss (MRL) ---
    use_mrl: bool = True
    # Adjusted MRL dims to match new perceiver_dim
    mrl_dims: Tuple[int, ...] = (128, 256, 512, 768, 3584) 
    mrl_temperature: float = 0.07
    mrl_weight: float = 0.1

    # === IMPROVEMENT 2: Training Dynamics ===
    # Contrastive loss needs large batches. 
    # If you get OOM, lower to 32 but use Gradient Cache if possible.
    batch_size_vision: int = 512       
    batch_size_audio: int = 512       
    
    # Train longer (200 steps is too short)
    max_train_steps_vision: int = 1000 
    max_train_steps_audio: int = 2000 
    
    learning_rate: float = 5e-4       # Slightly higher LR for Perceiver
    weight_decay: float = 0.01

    # === IMPROVEMENT 3: Data Scale ===
    # We need more than 3k samples for alignment
    librispeech_max_samples: int = 20000 
    vision_max_samples: int = 25000     
    max_audio_duration_s: float =  50
    
    # --- Training Dynamics ---
    max_grad_norm: float = 1.0        # Fix: Adds gradient clipping (prevents exploding gradients)
    num_rounds: int = 1               # Fix: Moving this to config
    grad_accum_steps: int = 1         # Fix: Explicitly define this (default was 4 in loop)

    # --- Paths & Misc ---
    vision_features_root: Path = Path("./features_vision")
    audio_features_root: Path = Path("./features_audio_librispeech")
    seed: int = 42
    log_every_steps: int = 20
    save_dir: Path = Path("./runs_perceiver_mrl_qwen")
    run_name: str = "optimized_alignment_run"

cfg = Config()
set_seed(cfg.seed)
print("Optimized Config Loaded.")

# --------------------------------------------
# W&B Init
# --------------------------------------------
import wandb
from dataclasses import asdict

run_name = cfg.run_name if hasattr(cfg, "run_name") else "tri_modal_alignment"

wandb.init(
    project=getattr(cfg, "wandb_project", "edgeglass-multimodal"),
    name=run_name,
    config=asdict(cfg),
)


### Phase-1: - Loading the Encoders
# ============================================
# Part 1 – Load models: vision, audio, text (Qwen2.5-7B)
# ============================================

# ------------------------------
# 1.1 Vision encoder (CLIP-style)
# ------------------------------
# For now we use CLIP as a simple vision encoder.
# Later you can swap this for your PixMo vision encoder or precomputed features.

from transformers import CLIPVisionModel, CLIPImageProcessor

print("\nLoading vision encoder:", cfg.vision_model_name)
vision_processor = CLIPImageProcessor.from_pretrained(cfg.vision_model_name)
vision_model = CLIPVisionModel.from_pretrained(
    cfg.vision_model_name,
    torch_dtype=default_dtype,
    device_map=None,
).to(device)
vision_model.eval()

for p in vision_model.parameters():
    p.requires_grad = False

# 🔥 Add this:
cfg.encoder_dim_vision = vision_model.config.hidden_size
print("Vision encoder_dim_vision:", cfg.encoder_dim_vision)


# ------------------------------
# 1.2 Audio encoder (Whisper)
# ------------------------------

print("\nLoading audio encoder:", cfg.audio_model_name)
audio_processor = WhisperProcessor.from_pretrained(cfg.audio_model_name)
audio_model = WhisperModel.from_pretrained(
    cfg.audio_model_name,
    torch_dtype=torch.float32,
    device_map=None,
).to(device)
audio_model.eval()

for p in audio_model.parameters():
    p.requires_grad = False

cfg.encoder_dim_audio = audio_model.config.d_model
print("Audio hidden size:", cfg.encoder_dim_audio)


# ------------------------------
# 1.3 Qwen2.5-7B (text encoder/decoder)
# ------------------------------
print("\nLoading Qwen2.5-7B:", cfg.llm_model_name)
qwen_tokenizer = AutoTokenizer.from_pretrained(
    cfg.llm_model_name,
    use_fast=True,
)
if qwen_tokenizer.pad_token is None:
    qwen_tokenizer.pad_token = qwen_tokenizer.eos_token

qwen_model = AutoModelForCausalLM.from_pretrained(
    cfg.llm_model_name,
    torch_dtype=default_dtype,
    device_map=None,  # <--- IMPORTANT: Disable auto split
    # attn_implementation="flash_attention_2" # <--- Enable for H200 speedup
)
qwen_model.eval()

for p in qwen_model.parameters():
    p.requires_grad = False

# 🔥 Robust extraction: handle int / list / tuple
hidden_size = getattr(qwen_model.config, "hidden_size", None)
if hidden_size is None:
    raise ValueError("Could not find hidden_size in Qwen config!")

if isinstance(hidden_size, (list, tuple)):
    hidden_size = hidden_size[0]

cfg.llm_hidden_size = int(hidden_size)

print("Qwen hidden_size (from config):", hidden_size)
print("cfg.llm_hidden_size:", cfg.llm_hidden_size, type(cfg.llm_hidden_size))

# Ensure we capture the full Qwen hidden dimension
# For Qwen2.5-7B, this is likely 3584
if hasattr(qwen_model.config, "hidden_size"):
    cfg.llm_hidden_size = qwen_model.config.hidden_size
else:
    # Fallback if config structure is different
    cfg.llm_hidden_size = 3584 

print(f"Full LLM Hidden Size: {cfg.llm_hidden_size}")

# FIX: The MRL loss MUST include the full embedding dimension.
# Previous config was (128, 256, 512).
# If we don't add 3584, the projector weights for indices 512 -> 3584 will never update.

current_mrl_dims = list(cfg.mrl_dims)

# Only append if it's not already there
if cfg.llm_hidden_size not in current_mrl_dims:
    current_mrl_dims.append(cfg.llm_hidden_size)

# Sort and freeze back to tuple
cfg.mrl_dims = tuple(sorted(current_mrl_dims))

print(f"✅ Corrected MRL Dimensions: {cfg.mrl_dims}")
# Expected output: (128, 256, 512, 3584)
### Phase-2: - Adding MLP layer for MRL
# ============================================
# Part 2 – Quick text embedding helper (for later MRL)
# ============================================

# def encode_text_with_qwen(
#     texts: List[str],
#     max_length: int = 64,
# ) -> Dict[str, torch.Tensor]:
#     """
#     Tokenize a batch of texts and return:
#         - input_ids
#         - attention_mask
#         - token_embeddings (from embedding layer, no LM forward yet)
#     """
#     model_device = next(qwen_model.parameters()).device
    
    
#     enc = qwen_tokenizer(
#         texts,
#         padding=True,
#         truncation=True,
#         max_length=max_length,
#         return_tensors="pt",
#     )

#     input_ids = enc.input_ids.to(model_device)
#     attn_mask = enc.attention_mask.to(model_device)

#     # (B, L, D_llm)
#     token_embs = qwen_model.get_input_embeddings()(input_ids)

#     # (B, L, D)
#     # token_embs = qwen_model.get_input_embeddings()(enc.input_ids)

#     return {
#         "input_ids": enc.input_ids,
#         "attention_mask": enc.attention_mask,
#         "token_embs": token_embs,
#     }


def encode_text_with_qwen(
    texts: List[str],
    max_length: int = 64,
) -> Dict[str, torch.Tensor]:
    """
    Tokenize a batch of texts and return:
        - input_ids
        - attention_mask
        - token_embeddings (from embedding layer, no LM forward yet)
    """
    model_device = next(qwen_model.parameters()).device
    
    enc = qwen_tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    input_ids = enc.input_ids.to(model_device)
    attn_mask = enc.attention_mask.to(model_device)

    # (B, L, D_llm)
    token_embs = qwen_model.get_input_embeddings()(input_ids)

    return {
        "input_ids": input_ids,
        "attention_mask": attn_mask,
        "token_embs": token_embs,
    }

# print("Text embedding helper ready.")


print("Text embedding helper ready.")

### Phase-3: - Load the Dataset
#### Load the Audio Dataset
# ============================================
# Part 3 – LibriSpeech (Streaming) Audio–Text Dataset
# ============================================

from datasets import load_dataset
import io
import librosa
import numpy as np

print("\nLoading LibriSpeech ASR (streaming mode)...")

# Load only train.clean.100 from the giant 124GB dataset
librispeech_raw = load_dataset(
    "openslr/librispeech_asr",
    "all",
    streaming=True,
    split="train.clean.100"
)

print("Loaded streaming dataset:", librispeech_raw)

# Disable automatic decoding → we want raw bytes for librosa
audio_stream = librispeech_raw.decode(False)

# We will collect up to cfg.librispeech_max_samples
max_samples = cfg.librispeech_max_samples  # rename in your config if needed
subset = []

print(f"\nTaking up to {max_samples} examples in streaming mode...")

for ex in audio_stream:
    subset.append(ex)
    if len(subset) >= max_samples:
        break

print("\nSubset collected:", len(subset))
print("Keys:", subset[0].keys())
print("Example 0:", subset[0])

# Helper: convert LibriSpeech streaming example → waveform
def load_waveform_from_streaming_example(example, target_sr=16000):
    audio_info = example["audio"]

    audio_bytes = audio_info["bytes"]
    if audio_bytes is None:
        raise ValueError("No audio bytes in example.")

    # Convert raw bytes → file-like object
    audio_file = io.BytesIO(audio_bytes)

    # librosa loads PCM data and resamples to target_sr
    wav, sr = librosa.load(audio_file, sr=target_sr)

    return wav, sr


# Helper: compute duration in seconds
def compute_duration(wav, sr):
    return len(wav) / float(sr)

# We'll filter to keep only clips <= cfg.max_audio_duration_s
filtered = []

print("\nFiltering by duration ≤", cfg.max_audio_duration_s, "seconds...")

for ex in subset:
    wav, sr = load_waveform_from_streaming_example(ex, cfg.audio_sample_rate)
    dur = compute_duration(wav, sr)

    # if dur <= cfg.max_audio_duration_s:
    if True:
        filtered.append({
            "waveform": wav,
            "sampling_rate": sr,
            "duration": dur,
            "text": ex["text"]
        })
print("After duration filtering:", len(filtered), "examples")
print("\nShowing a few filtered samples...")
for i in range(min(5, len(filtered))):
    ex = filtered[i]
    print(f"\nSample {i}:")
    print("  Duration:", round(ex["duration"], 2), "s")
    print("  Transcript:", ex["text"])
    print("  Waveform shape:", ex["waveform"].shape)
len(filtered)

# ============================================
# New PixmoVisionDataset (uses HF 'image' column if available)
# ============================================

from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import requests
from io import BytesIO
import random

print("\nLoading PixMo-Cap vision–text dataset (allenai/pixmo-cap)...")

pixmo_raw = load_dataset("allenai/pixmo-cap", split="train")
print("PixMo-Cap split size:", len(pixmo_raw))
print("PixMo columns:", pixmo_raw.column_names)

# We only need a small subset for the POC
vision_max = getattr(cfg, "vision_max_samples", 2048)
if len(pixmo_raw) > vision_max:
    pixmo_subset = pixmo_raw.shuffle(seed=cfg.seed).select(range(vision_max))
else:
    pixmo_subset = pixmo_raw

print("PixMo subset size:", len(pixmo_subset))

# Fields from the dataset card:
#  - "image_url": URL to the image
#  - "caption": long caption text
img_col = "image_url"
txt_col = "caption"

cols = pixmo_raw.column_names
HAS_IMAGE_COL = "image" in cols

if HAS_IMAGE_COL:
    img_col = "image"
else:
    img_col = "image_url"

txt_col = "caption"

print(f"Using image column: {img_col}")

class PixmoVisionDataset(Dataset):
    """
    On-the-fly image loading + CLIP feature extraction.

    If 'image' column exists: uses HF-managed images (no manual HTTP).
    Else: falls back to 'image_url' with robust skipping of bad URLs.

    Returns:
        {
          "features": Tensor(T, d_vision),
          "text": str
        }
    """
    def __init__(
        self,
        hf_dataset,
        vision_model,
        vision_processor,
        max_retries: int = 5,
    ):
        self.ds = hf_dataset
        self.vision_model = vision_model
        self.vision_processor = vision_processor
        self.max_retries = max_retries

    def __len__(self):
        return len(self.ds)

    # -----------------------------
    # Low-level helpers
    # -----------------------------
    def _load_image_from_url(self, url: str) -> Image.Image:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content)).convert("RGB")
        return img

    def _encode_image(self, img: Image.Image) -> torch.Tensor:
        proc = self.vision_processor(images=img, return_tensors="pt")
        pixel_values = proc["pixel_values"].to(device)

        with torch.no_grad():
            out = self.vision_model(pixel_values=pixel_values)
            # (1, T, d_vision)
            feats = out.last_hidden_state.squeeze(0).to("cpu")  # (T, d_vision)

        return feats

    # -----------------------------
    # Core example fetcher
    # -----------------------------
    def _get_example(self, idx: int) -> dict:
        ex = self.ds[idx]
        caption = ex[txt_col]

        if HAS_IMAGE_COL:
            # HF has already downloaded/cached images
            img = ex[img_col]
            # Robust conversion to PIL RGB
            if isinstance(img, Image.Image):
                img = img.convert("RGB")
            else:
                # handle HF Image object / numpy array / etc.
                img = Image.fromarray(np.array(img)).convert("RGB")
        else:
            url = ex[img_col]
            img = self._load_image_from_url(url).convert("RGB")

        feats = self._encode_image(img)
        return {
            "features": feats,
            "text": caption,
        }

    # -----------------------------
    # Robust __getitem__
    # -----------------------------
    def __getitem__(self, idx: int) -> dict:
        """
        Try up to max_retries times with different indices if something fails
        (HTTP error, decoding error, etc). If everything fails, return a dummy
        sample instead of crashing.

        This is important in multi-GPU training: a single bad URL should not
        kill the whole job.
        """
        n = len(self.ds)
        attempt = 0
        cur_idx = idx

        # 1) Try sequential neighbours
        while attempt < self.max_retries:
            try:
                return self._get_example(cur_idx)
            except Exception:
                # Uncomment for debugging if needed (but will be noisy on many ranks):
                # print(f"[PixmoVisionDataset] Failed idx={cur_idx}, attempt={attempt+1}, err={e}")
                attempt += 1
                cur_idx = (cur_idx + 1) % n

        # 2) Try random indices
        for _ in range(self.max_retries):
            j = random.randint(0, n - 1)
            try:
                return self._get_example(j)
            except Exception:
                continue

        # 3) Final fallback: return a dummy sample instead of raising
        dummy_feats = torch.zeros(1, cfg.encoder_dim_vision, dtype=default_dtype)
        dummy_text = "dummy caption (fallback)"

        # Optional: only log from rank 0 to avoid spam
        # if int(os.environ.get("RANK", "0")) == 0:
        #     print("[PixmoVisionDataset] All attempts failed, returning dummy sample.")

        return {
            "features": dummy_feats,
            "text": dummy_text,
        }




### Part-4:- 
# ============================================
# Part 4 – Audio features dataset (FIXED)
# ============================================

from torch.utils.data import Dataset
from torchaudio import transforms as T_audio

def whisper_encode_sequence(wav: np.ndarray, sr: int, duration_sec: float):
    """
    Encodes audio and SLICES out the padding (Crucial Fix).
    """
    # 1. Process raw waveform -> log-Mel (pad to 30s internally)
    inputs = audio_processor(
        wav,
        sampling_rate=sr,
        return_tensors="pt",
    )
    input_features = inputs["input_features"].to(device) # (1, 80, 3000)

    # 2. SpecAugment (only during training)
    if audio_model.training:
        freq_mask = T_audio.FrequencyMasking(freq_mask_param=15)
        time_mask = T_audio.TimeMasking(time_mask_param=35)
        input_features = freq_mask(input_features)
        input_features = time_mask(input_features)

    # 3. Encoder Forward
    with torch.no_grad():
        enc_out = audio_model.encoder(input_features)
        hidden = enc_out.last_hidden_state.squeeze(0) # (1500, 512)

    # === CRITICAL FIX: Slice to actual duration ===
    # Whisper frame rate is 50Hz (20ms per frame)
    # 10 seconds of audio = 500 frames. The rest (1000 frames) is garbage padding.
    valid_frames = int(duration_sec * 50)
    
    # Safety clamp (min 1 frame, max 1500)
    valid_frames = max(1, min(valid_frames, 1500))
    
    # Return only valid frames
    feats = hidden[:valid_frames, :].to(torch.float16).cpu() 
    return feats

class LibriSpeechAudioDataset(Dataset):
    def __init__(self, examples, max_len: int | None = None):
        self.examples = examples
        if max_len is not None and max_len < len(examples):
            self.examples = examples[:max_len]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int):
        ex = self.examples[idx]
        wav = ex["waveform"]
        sr = ex["sampling_rate"]
        dur = ex["duration"]
        
        # === FIX 2: Text Normalization ===
        # LibriSpeech is ALL CAPS. LLMs expect Normal case.
        raw_text = ex["text"]
        text = raw_text.lower().capitalize()

        # Pass duration to the encoder
        feats = whisper_encode_sequence(wav, sr, duration_sec=dur)

        return {
            "features": feats,
            "text": text,
            "duration": dur,
        }

# Re-init dataset
audio_max = getattr(cfg, "librispeech_max_samples", len(filtered))
audio_dataset = LibriSpeechAudioDataset(filtered, max_len=audio_max)

print("Audio dataset fixed (Padding Slicing + Text Norm).")
print("Example 0 features shape:", audio_dataset[0]["features"].shape) # Should NOT be (1500, 512) anymore unless audio is exactly 30s
print("Example 0 text:", audio_dataset[0]["text"])
### Part-5
# ============================================
# Part 5 – Unified Adapters, Perceiver Resampler & Projector
# ============================================

import math
import torch.nn.functional as F

# --------------------------------------------
# 5.0 – Ensure Perceiver hyperparams exist in cfg
# --------------------------------------------

if not hasattr(cfg, "num_perceiver_layers"):
    cfg.num_perceiver_layers = 2          # depth of Perceiver
if not hasattr(cfg, "num_attn_heads"):
    cfg.num_attn_heads = 8                # multi-head attention
if not hasattr(cfg, "mlp_ratio"):
    cfg.mlp_ratio = 4.0                   # width of MLP inside Perceiver

print("Perceiver config:")
print("  perceiver_dim:", cfg.perceiver_dim)
print("  num_latents:", cfg.num_latents)
print("  num_perceiver_layers:", cfg.num_perceiver_layers)
print("  num_attn_heads:", cfg.num_attn_heads)
print("  mlp_ratio:", cfg.mlp_ratio)
# --------------------------------------------
# 5.1 – Modality adapters: vision & audio → perceiver_dim
# --------------------------------------------

class ModalityAdapter(nn.Module):
    """
    Simple linear adapter: maps encoder dim → perceiver_dim.
    Used separately for vision and audio encoders.
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, in_dim) or (T, in_dim)
        returns: (B, T, out_dim) or (T, out_dim)
        """
        return self.proj(x)


vision_adapter = ModalityAdapter(cfg.encoder_dim_vision, cfg.perceiver_dim).to(device)
audio_adapter  = ModalityAdapter(cfg.encoder_dim_audio,  cfg.perceiver_dim).to(device)

print("\nAdapters created:")
print("  VisionAdapter:", vision_adapter)
print("  AudioAdapter:", audio_adapter)
# --------------------------------------------
# 5.2 – Perceiver building blocks
# --------------------------------------------

class FeedForward(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PerceiverLayer(nn.Module):
    """
    One Perceiver layer:
      1) Cross-attention: latents query encoder tokens
      2) Self-attention on latents
      3) MLP on latents
    """
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.dim = dim

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.self_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True,
        )

        self.ln_latents_1 = nn.LayerNorm(dim)
        self.ln_tokens    = nn.LayerNorm(dim)
        self.ln_latents_2 = nn.LayerNorm(dim)
        self.ln_latents_3 = nn.LayerNorm(dim)

        self.mlp = FeedForward(dim, mlp_ratio=mlp_ratio)

    def forward(
        self,
        latents: torch.Tensor,   # (B, L, D)
        tokens: torch.Tensor,    # (B, T, D)
        token_mask: torch.Tensor | None = None,  # (B, T) bool, 1=valid
    ) -> torch.Tensor:
        """
        token_mask: bool mask, True for valid tokens. Will be converted to key_padding_mask.
        """
        B, L, D = latents.shape
        _, T, _ = tokens.shape

        # LayerNorm
        q = self.ln_latents_1(latents)   # (B, L, D)
        kv = self.ln_tokens(tokens)      # (B, T, D)

        # key_padding_mask: True for *ignored* positions
        key_padding_mask = None
        if token_mask is not None:
            # token_mask: True=valid → invert
            key_padding_mask = ~token_mask.bool()   # (B, T)

        # 1) Cross-attention: latents query the encoder tokens
        attn_out, _ = self.cross_attn(
            query=q,
            key=kv,
            value=kv,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        latents = latents + attn_out

        # 2) Self-attention on latents
        q2 = self.ln_latents_2(latents)
        self_attn_out, _ = self.self_attn(
            query=q2,
            key=q2,
            value=q2,
            need_weights=False,
        )
        latents = latents + self_attn_out

        # 3) MLP on latents
        latents = latents + self.mlp(self.ln_latents_3(latents))

        return latents



class PerceiverResampler(nn.Module):
    """
    Latent array Z ∈ R^{L × D}, cross-attends to encoder tokens X ∈ R^{B × T × D}
    to produce a fixed number of latent tokens per example.
    """
    def __init__(
        self,
        dim: int,
        num_latents: int,
        num_layers: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_latents = num_latents

        # Learned latent array (L, D)
        self.latents = nn.Parameter(torch.randn(num_latents, dim) / math.sqrt(dim))

        # Stack of Perceiver layers
        self.layers = nn.ModuleList([
            PerceiverLayer(dim, num_heads=num_heads, mlp_ratio=mlp_ratio)
            for _ in range(num_layers)
        ])

    def forward(
        self,
        tokens: torch.Tensor,         # (B, T, D)
        token_mask: torch.Tensor | None = None,  # (B, T) bool
    ) -> torch.Tensor:
        B, T, D = tokens.shape
        assert D == self.dim, f"Expected dim={self.dim}, got {D}"

        # Expand latent array to batch: (B, L, D)
        latents = self.latents.unsqueeze(0).expand(B, -1, -1)

        for layer in self.layers:
            latents = layer(latents, tokens, token_mask)

        return latents  # (B, L, D)


perceiver = PerceiverResampler(
    dim=cfg.perceiver_dim,
    num_latents=cfg.num_latents,
    num_layers=cfg.num_perceiver_layers,
    num_heads=cfg.num_attn_heads,
    mlp_ratio=cfg.mlp_ratio,
).to(device)

print("\nPerceiverResampler created:")
print(perceiver)



# --------------------------------------------
# 5.3 – Projector: Perceiver → Qwen hidden space
# --------------------------------------------

projector = nn.Linear(cfg.perceiver_dim, cfg.llm_hidden_size).to(device)
print("\nProjector created:")
print("  projector:", projector)



# --------------------------------------------
# 5.4 – Quick shape sanity check with fake batch
# --------------------------------------------

with torch.no_grad():
    B = 2
    # Fake vision sequence: (B, T_v, d_vision)
    T_v = 32
    fake_vision = torch.randn(B, T_v, cfg.encoder_dim_vision, device=device, dtype=default_dtype)
    fake_mask   = torch.ones(B, T_v, dtype=torch.bool, device=device)

    # 1) Adapt to perceiver_dim
    v_tokens = vision_adapter(fake_vision)           # (B, T_v, D_perc)

    # 2) Perceiver latents
    latents = perceiver(v_tokens, fake_mask)         # (B, L, D_perc)

    # 3) Project to Qwen hidden dim
    z_llm = projector(latents)                       # (B, L, D_llm)

print("\nSanity check:")
print("  v_tokens shape:", v_tokens.shape)
print("  latents shape:", latents.shape)
print("  z_llm shape:", z_llm.shape)
print("Done Part 5.")

print("\n== Sanity check dims ==")
print("encoder_dim_vision:", cfg.encoder_dim_vision, type(cfg.encoder_dim_vision))
print("encoder_dim_audio:", cfg.encoder_dim_audio, type(cfg.encoder_dim_audio))
print("perceiver_dim:", cfg.perceiver_dim, type(cfg.perceiver_dim))
print("llm_hidden_size:", cfg.llm_hidden_size, type(cfg.llm_hidden_size))

### Part-6
# ============================================
# Part 6 – Collate, Matryoshka loss, Forward Step
# ============================================

from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence


# --------------------------------------------
# 6.1 – Collate functions for vision & audio
# --------------------------------------------

def collate_features_with_text(batch):
    """
    Generic collate:
        batch: list of dicts with
            "features": (T_i, D_enc)
            "text": str
            (optionally "duration")
    Returns:
        encoder_feats: (B, T_max, D_enc)
        encoder_mask:  (B, T_max) bool
        texts: list[str]
        durations: list[float] | None
    """
    feats = [torch.as_tensor(ex["features"], dtype=default_dtype) for ex in batch]  # list[(T_i, D_enc)]
    lengths = [f.size(0) for f in feats]

    # Pad to max length
    encoder_feats = pad_sequence(feats, batch_first=True)  # (B, T_max, D_enc)

    B, T_max, _ = encoder_feats.shape
    encoder_mask = torch.zeros(B, T_max, dtype=torch.bool)
    for i, L in enumerate(lengths):
        encoder_mask[i, :L] = True

    texts = [ex["text"] for ex in batch]
    durations = [ex.get("duration", None) for ex in batch]

    return {
        "encoder_feats": encoder_feats,    # (B, T_max, D_enc)
        "encoder_mask": encoder_mask,      # (B, T_max)
        "texts": texts,
        "durations": durations,
    }

vision_dataset = PixmoVisionDataset(
    pixmo_subset,
    vision_model=vision_model,
    vision_processor=vision_processor,
)

print("Vision dataset ready (HF image-based if available).")
sample_v = vision_dataset[0]
print("  features shape:", sample_v["features"].shape)
print("  text snippet:", sample_v["text"][:120], "...")

vision_loader = DataLoader(
    vision_dataset,
    batch_size=cfg.batch_size_vision,
    shuffle=True,
    collate_fn=collate_features_with_text,
)

# Vision & audio loaders (you’ll use these in Part 7 for training)
vision_loader = DataLoader(
    vision_dataset,
    batch_size=cfg.batch_size_vision,
    shuffle=True,
    collate_fn=collate_features_with_text,
)

audio_loader = DataLoader(
    audio_dataset,
    batch_size=cfg.batch_size_audio,
    shuffle=True,
    collate_fn=collate_features_with_text,
)

print("Vision loader & audio loader ready.")

# --------------------------------------------
# 6.2 – Matryoshka (MRL) contrastive loss
# --------------------------------------------

# def matryoshka_contrastive_loss(
#     z_mod: torch.Tensor,    # (B, D)
#     z_txt: torch.Tensor,    # (B, D)
#     trunc_dims: tuple[int, ...],
#     temperature: float = 0.07,
# ) -> torch.Tensor:
#     """
#     Matryoshka-style symmetric InfoNCE at multiple truncation dims.

#     For each d in trunc_dims:
#       - truncate embeddings to first d dims
#       - L2-normalize
#       - compute similarity matrix
#       - compute symmetric cross-entropy (mod→text and text→mod)
#     Then average across all dims.
#     """
#     assert z_mod.shape == z_txt.shape
#     B, D = z_mod.shape
#     max_d = max(trunc_dims)
#     assert max_d <= D, f"Max trunc dim {max_d} exceeds embedding dim {D}"

#     losses = []
#     targets = torch.arange(B, device=z_mod.device)

#     for d in trunc_dims:
#         zm = F.normalize(z_mod[:, :d], dim=-1)  # (B, d)
#         zt = F.normalize(z_txt[:, :d], dim=-1)  # (B, d)

#         logits = zm @ zt.T / temperature        # (B, B)
#         loss_m2t = F.cross_entropy(logits, targets)
#         loss_t2m = F.cross_entropy(logits.T, targets)

#         losses.append(0.5 * (loss_m2t + loss_t2m))

#     return sum(losses) / len(losses)



# --------------------------------------------
# 6.2 – Matryoshka (MRL) contrastive loss
# --------------------------------------------

def matryoshka_contrastive_loss(
    z_mod: torch.Tensor,    # (B, D)
    z_txt: torch.Tensor,    # (B, D)
    trunc_dims: tuple[int, ...],
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Matryoshka-style symmetric InfoNCE at multiple truncation dims.

    For each d in trunc_dims:
      - truncate embeddings to first d dims
      - L2-normalize
      - compute similarity matrix
      - compute symmetric cross-entropy (mod→text and text→mod)
    Then average across all dims.
    """
    assert z_mod.shape == z_txt.shape, "z_mod and z_txt must have same shape"
    B, D = z_mod.shape

    # Ensure both embeddings share the same dtype (important under bf16 mixed precision)
    if z_mod.dtype != z_txt.dtype:
        z_txt = z_txt.to(z_mod.dtype)

    # Sanity check truncation dims
    max_d = max(trunc_dims)
    assert max_d <= D, f"Max trunc dim {max_d} exceeds embedding dim {D}"

    # Make temperature a tensor on the right device/dtype
    temp = torch.as_tensor(temperature, device=z_mod.device, dtype=z_mod.dtype)

    losses = []
    targets = torch.arange(B, device=z_mod.device)

    for d in trunc_dims:
        # 1) Truncate and L2-normalize
        zm = F.normalize(z_mod[:, :d], dim=-1)  # (B, d)
        zt = F.normalize(z_txt[:, :d], dim=-1)  # (B, d)

        # 2) Similarity matrix
        logits = zm @ zt.T / temp               # (B, B)

        # 3) Symmetric cross-entropy
        loss_m2t = F.cross_entropy(logits, targets)
        loss_t2m = F.cross_entropy(logits.T, targets)

        losses.append(0.5 * (loss_m2t + loss_t2m))

    return sum(losses) / len(losses)


# --------------------------------------------
# 6.3 – Helpers for global text & modality embeddings
# --------------------------------------------

# def pooled_text_embedding(texts: list[str], max_length: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
#     """
#     Returns:
#         h_text: (B, D_llm) pooled text embeddings
#         text_tok_info: dict with token_embs, input_ids, attention_mask
#     """
#     tok_out = encode_text_with_qwen(texts, max_length=max_length)  # uses qwen_model embedding layer
#     token_embs = tok_out["token_embs"]          # (B, L, D_llm)
#     attn_mask = tok_out["attention_mask"]      # (B, L)

#     mask = attn_mask.unsqueeze(-1).to(token_embs.device)  # (B, L, 1)
        
#     # masked mean-pooling over tokens
#     mask = attn_mask.unsqueeze(-1)             # (B, L, 1)
#     denom = mask.sum(dim=1).clamp_min(1)       # (B, 1)
#     h_text = (token_embs * mask).sum(dim=1) / denom  # (B, D_llm)


#     return h_text, tok_out


def pooled_text_embedding(
    texts: list[str],
    max_length: int = 64,
) -> tuple[torch.Tensor, dict]:
    """
    Returns:
        h_text: (B, D_llm) pooled text embeddings
        tok_out: dict with token_embs, input_ids, attention_mask
    """
    tok_out = encode_text_with_qwen(texts, max_length=max_length)
    token_embs = tok_out["token_embs"]          # (B, L, D_llm) on GPU
    attn_mask  = tok_out["attention_mask"]      # (B, L) on same device

    # masked mean-pooling over tokens
    mask  = attn_mask.unsqueeze(-1)             # (B, L, 1)
    denom = mask.sum(dim=1).clamp_min(1)        # (B, 1)
    h_text = (token_embs * mask).sum(dim=1) / denom  # (B, D_llm)

    return h_text, tok_out


def pooled_modality_embedding(latent_tokens_llm: torch.Tensor) -> torch.Tensor:
    """
    latent_tokens_llm: (B, L, D_llm) from projector(perceiver(...))
    Returns:
        h_mod: (B, D_llm)
    """
    return latent_tokens_llm.mean(dim=1)  # simple mean over latents


# --------------------------------------------
# 6.4 – Unified alignment forward step (vision or audio)
# --------------------------------------------

# def forward_alignment_step(
#     batch: dict,
#     modality: str = "vision",   # "vision" or "audio"
# ) -> tuple[torch.Tensor, dict]:
#     """
#     One step of alignment loss for a batch.

#     batch keys from collate_features_with_text:
#         - encoder_feats: (B, T, D_enc)
#         - encoder_mask:  (B, T) bool
#         - texts: list[str]

#     modality:
#         "vision" → use vision_adapter
#         "audio"  → use audio_adapter
#     """
#     encoder_feats = batch["encoder_feats"].to(device)   # (B, T, D_enc)
#     encoder_mask  = batch["encoder_mask"].to(device)    # (B, T)
#     texts         = batch["texts"]                      # list[str]

#     # 1) Modality adapter → Perceiver dim
#     if modality == "vision":
#         tokens = vision_adapter(encoder_feats)          # (B, T, D_perc)
#     elif modality == "audio":
#         tokens = audio_adapter(encoder_feats)           # (B, T, D_perc)
#     else:
#         raise ValueError(f"Unknown modality: {modality}")

#     # 2) Perceiver resampler → latent tokens
#     latents = perceiver(tokens, encoder_mask)           # (B, L, D_perc)

#     # 3) Project to Qwen hidden space
#     z_llm = projector(latents)                          # (B, L, D_llm)

#     # 4) Global modality embedding (for MRL)
#     h_mod = pooled_modality_embedding(z_llm)            # (B, D_llm)

#     # 5) Global text embedding from Qwen
#     h_txt, tok_info = pooled_text_embedding(texts, max_length=64)  # (B, D_llm)

#     # 6) Matryoshka contrastive loss
#     mrl_loss = matryoshka_contrastive_loss(
#         h_mod,
#         h_txt,
#         trunc_dims=cfg.mrl_dims,
#         temperature=cfg.mrl_temperature,
#     )

#     # For now we focus on alignment-only POC → total_loss = mrl_loss
#     total_loss = mrl_loss

#     metrics = {
#         "loss":        float(total_loss.detach().cpu()),
#         "mrl_loss":    float(mrl_loss.detach().cpu()),
#         "modality":    modality,
#         "batch_size":  int(h_mod.size(0)),
#     }

#     return total_loss, metrics


# --------------------------------------------
# 6.4 – Unified alignment forward step (Fixed)
# --------------------------------------------

# def forward_alignment_step(
#     batch: dict,
#     accelerator,
#     modality: str = "vision",   # "vision" or "audio"
# ) -> tuple[torch.Tensor, dict]:
#     """
#     One step of alignment loss for a batch.
#     """
#     encoder_feats = batch["encoder_feats"].to(device)   # (B, T, D_enc)
#     encoder_mask  = batch["encoder_mask"].to(device)    # (B, T)
#     texts         = batch["texts"]                      # list[str]

#     # 1) Modality adapter -> Perceiver dim
#     if modality == "vision":
#         tokens = vision_adapter(encoder_feats)          # (B, T, D_perc)
#     elif modality == "audio":
#         tokens = audio_adapter(encoder_feats)           # (B, T, D_perc)
#     else:
#         raise ValueError(f"Unknown modality: {modality}")

#     # 2) Perceiver resampler -> latent tokens
#     latents = perceiver(tokens, encoder_mask)           # (B, L, D_perc)

#     # 3) Project to Qwen hidden space
#     z_llm_local = projector(latents)                          # (B, L, D_llm)
    
#     # Safety check: Ensure projector output matches config
#     assert z_llm.shape[-1] == cfg.llm_hidden_size, \
#         f"Projector output {z_llm.shape[-1]} != Config {cfg.llm_hidden_size}"

#     # 2. Get Local Embeddings
#     h_mod_local = pooled_modality_embedding(z_llm_local)
#     h_txt_local, _ = pooled_text_embedding(texts, max_length=64)
    
    
#     # 4) Global modality embedding (Mean Pooling over latents)
#     h_mod = pooled_modality_embedding(z_llm)            # (B, D_llm)

#     # 5) Global text embedding from Qwen (Pre-computed or on-the-fly)
#     # Note: encode_text_with_qwen returns raw embeddings, not LM outputs, 
#     # which is correct for alignment.
#     h_txt, tok_info = pooled_text_embedding(texts, max_length=64)  # (B, D_llm)

#     h_mod_global = accelerator.gather(h_mod_local) 
#     h_txt_global = accelerator.gather(h_txt_local)
    
#     # 6) Matryoshka contrastive loss
#     # We pass the corrected cfg.mrl_dims here (e.g., 128, 256, 512, 3584)
#     mrl_loss = matryoshka_contrastive_loss(
#         h_mod_global,
#         h_txt_global,
#         trunc_dims=cfg.mrl_dims,
#         temperature=cfg.mrl_temperature,
#     )

#     metrics = {
#         "loss":        float(mrl_loss.detach().cpu()),
#         "mrl_loss":    float(mrl_loss.detach().cpu()),
#         "modality":    modality,
#         "batch_size":  int(h_mod.size(0)),
#     }

#     return mrl_loss, metrics



def forward_alignment_step(
    batch: dict,
    accelerator,
    modality: str = "vision",   # "vision" or "audio"
) -> tuple[torch.Tensor, dict]:
    """
    One step of alignment loss for a batch.

    batch keys:
        - encoder_feats: (B, T, D_enc)
        - encoder_mask:  (B, T) bool
        - texts: list[str]
    """
    device = accelerator.device

    encoder_feats = batch["encoder_feats"].to(device)   # (B, T, D_enc)
    encoder_mask  = batch["encoder_mask"].to(device)    # (B, T)
    texts         = batch["texts"]                      # list[str]

    # 1) Modality adapter → Perceiver dim
    if modality == "vision":
        tokens = vision_adapter(encoder_feats)          # (B, T, D_perc)
    elif modality == "audio":
        tokens = audio_adapter(encoder_feats)           # (B, T, D_perc)
    else:
        raise ValueError(f"Unknown modality: {modality}")

    # 2) Perceiver resampler → latent tokens
    latents = perceiver(tokens, encoder_mask)           # (B, L, D_perc)

    # 3) Project to Qwen hidden space
    z_llm_local = projector(latents)                    # (B, L, D_llm)
    assert z_llm_local.shape[-1] == cfg.llm_hidden_size, \
        f"Projector output {z_llm_local.shape[-1]} != cfg.llm_hidden_size {cfg.llm_hidden_size}"

    # 4) Local pooled embeddings
    h_mod_local = pooled_modality_embedding(z_llm_local)       # (B, D_llm)
    h_txt_local, _ = pooled_text_embedding(texts, max_length=64)  # (B, D_llm)

    # 5) Gather across all processes → global batch for contrastive loss
    h_mod_global = accelerator.gather(h_mod_local)
    h_txt_global = accelerator.gather(h_txt_local)

    # 6) Matryoshka contrastive loss
    mrl_loss = matryoshka_contrastive_loss(
        h_mod_global,
        h_txt_global,
        trunc_dims=cfg.mrl_dims,
        temperature=cfg.mrl_temperature,
    )

    metrics = {
        "loss":        float(mrl_loss.detach().cpu()),
        "mrl_loss":    float(mrl_loss.detach().cpu()),
        "modality":    modality,
        "batch_size":  int(h_mod_local.size(0)),
    }

    return mrl_loss, metrics

print("\nPart 6 ready: collate, MRL, and forward_alignment_step defined.")

### Part-7
# ============================================
# Part 7 – Training loops (vision & audio alignment)
# ============================================

from torch.optim import AdamW
from tqdm.auto import tqdm


# --------------------------------------------
# 7.0 – Collect trainable parameters
# --------------------------------------------

# We ONLY train:
#   - vision_adapter
#   - audio_adapter
#   - perceiver
#   - projector
# Qwen, CLIP, and Whisper are frozen.

trainable_modules = nn.ModuleList([
    vision_adapter,
    audio_adapter,
    perceiver,
    projector,
])

for name, p in trainable_modules.named_parameters():
    if p.requires_grad:
        print("Trainable:", name, p.shape)

optimizer = AdamW(
    [p for p in trainable_modules.parameters() if p.requires_grad],
    lr=cfg.learning_rate,
    weight_decay=cfg.weight_decay,
)

print("\nOptimizer ready with", sum(p.numel() for p in trainable_modules.parameters() if p.requires_grad), "trainable params.")



wandb.watch(trainable_modules, log="all", log_freq=50)
# --------------------------------------------
# 7.1 – Generic training epoch for one modality
# --------------------------------------------
# def train_one_epoch(
#     dataloader,
#     modality,
#     max_steps,
#     optimizer,      # Passed in
#     scheduler,      # Passed in
#     accelerator,
#     log_prefix=""
# ):
#     trainable_modules.train()
#     running_loss = 0.0
#     num_batches = 0
#     import time
#     start_time = time.time()

    
#     # We assume dataloader is already prepared by accelerator
#     # pbar = tqdm(dataloader, total=max_steps, desc=f"{log_prefix}train-{modality}", leave=False)
#     pbar = tqdm(
#     dataloader,
#     total=max_steps,
#     desc=f"{log_prefix}train-{modality}",
#     leave=True,
#     dynamic_ncols=True,
#     )
#     for step, batch in enumerate(pbar, start=1):
#         if step > max_steps:
#             break

#         optimizer.zero_grad(set_to_none=True)

#         # Forward step
#         loss, metrics = forward_alignment_step(batch, accelerator, modality=modality)
        
#         # Backward step (Accelerate handles the scaling/unscaling)
#         accelerator.backward(loss)

#         if cfg.max_grad_norm is not None:
#             if accelerator.sync_gradients:
#                 accelerator.clip_grad_norm_(trainable_modules.parameters(), cfg.max_grad_norm)

#         optimizer.step()
#         if scheduler:
#             scheduler.step()

#         running_loss += metrics["loss"]
#         num_batches += 1
#         avg_loss = running_loss / num_batches

#         # W&B logging (only on main process)
#         if accelerator.is_main_process:
#             wandb.log({
#                 f"{modality}/train/loss": metrics["loss"],
#                 f"{modality}/train/avg_loss": avg_loss,
#                 f"{modality}/train/mrl_loss": metrics["mrl_loss"],
#             })
            
#             if step % cfg.log_every_steps == 0:
#                 pbar.set_postfix({"loss": f"{metrics['loss']:.4f}"})
#         if accelerator.is_main_process and step % 50 == 0:
#             elapsed = time.time() - start_time
#             steps_done = step
#             steps_left = max_steps - steps_done
#             eta = steps_left * (elapsed / steps_done)
#             print(f"[ETA] {modality}: {eta/60:.2f} min remaining")

#     return running_loss / max(1, num_batches)

import time

def train_one_epoch(
    dataloader,
    modality,
    max_steps,
    optimizer,      # Passed in
    scheduler,      # Passed in
    accelerator,
    log_prefix=""
):
    trainable_modules.train()

    # --- timing state ---
    start_time = time.time()
    running_loss = 0.0
    num_batches = 0

    # tqdm with ETA-friendly settings
    pbar = tqdm(
        dataloader,
        total=max_steps,
        desc=f"{log_prefix}train-{modality}",
        leave=True,
        dynamic_ncols=True,
    )

    for step, batch in enumerate(pbar, start=1):
        if step > max_steps:
            break

        iter_start = time.time()

        optimizer.zero_grad(set_to_none=True)

        # Forward step
        loss, metrics = forward_alignment_step(batch, accelerator, modality=modality)
        
        # Backward step (Accelerate handles scaling/unscaling)
        accelerator.backward(loss)

        if cfg.max_grad_norm is not None and accelerator.sync_gradients:
            accelerator.clip_grad_norm_(trainable_modules.parameters(), cfg.max_grad_norm)

        optimizer.step()
        if scheduler:
            scheduler.step()

        # ---- update running stats ----
        running_loss += metrics["loss"]
        num_batches += 1
        avg_loss = running_loss / num_batches

        iter_time = time.time() - iter_start
        elapsed = time.time() - start_time
        steps_done = num_batches
        steps_left = max_steps - steps_done
        avg_step_time = elapsed / max(1, steps_done)
        eta_sec = steps_left * avg_step_time

        if accelerator.is_main_process:
            # Update tqdm bar
            if step % cfg.log_every_steps == 0:
                pbar.set_postfix({
                    "loss": f"{metrics['loss']:.4f}",
                    "eta_min": f"{eta_sec/60:.1f}",
                })

            # W&B logging
            wandb.log({
                f"{modality}/train/loss": metrics["loss"],
                f"{modality}/train/avg_loss": avg_loss,
                f"{modality}/train/mrl_loss": metrics["mrl_loss"],
                f"{modality}/train/step_time_sec": iter_time,
                f"{modality}/train/avg_step_time_sec": avg_step_time,
                f"{modality}/train/eta_sec": eta_sec,
                f"{modality}/train/eta_min": eta_sec / 60.0,
                f"{modality}/train/steps_done": steps_done,
                f"{modality}/train/steps_left": steps_left,
            })

    return running_loss / max(1, num_batches)



# --------------------------------------------
# 7.2 – Simple retrieval eval (sanity check)
# --------------------------------------------

@torch.no_grad()
def eval_retrieval(
    dataset,
    modality: str,
    num_samples: int = 64,
):
    """
    Very small retrieval sanity check:
      - take num_samples examples
      - compute modality & text embeddings
      - compute similarity matrix
      - report Recall@1 (how often correct text is most similar)

    Works for both vision_dataset and audio_dataset.
    """
    trainable_modules.eval()

    # Build a tiny batch with collate
    from math import ceil
    B = min(num_samples, len(dataset))
    # Manual batching using DataLoader with our collate
    tmp_loader = DataLoader(
        dataset,
        batch_size=B,
        shuffle=True,
        collate_fn=collate_features_with_text,
    )
    batch = next(iter(tmp_loader))

    # Forward until we get h_mod and h_txt (without loss)
    encoder_feats = batch["encoder_feats"].to(device)
    encoder_mask  = batch["encoder_mask"].to(device)
    texts         = batch["texts"]

    if modality == "vision":
        tokens = vision_adapter(encoder_feats)
    elif modality == "audio":
        tokens = audio_adapter(encoder_feats)
    else:
        raise ValueError(f"Unknown modality: {modality}")

    latents = perceiver(tokens, encoder_mask)
    z_llm   = projector(latents)

    h_mod = pooled_modality_embedding(z_llm)      # (B, D_llm)
    h_txt, _ = pooled_text_embedding(texts)      # (B, D_llm)

    # Normalize
    h_mod = F.normalize(h_mod, dim=-1)
    h_txt = F.normalize(h_txt, dim=-1)

    # Similarity matrix (B, B)
    sims = h_mod @ h_txt.T

    # For each modality embedding, check if its diagonal text is top-1
    ranks = sims.argsort(dim=-1, descending=True)
    correct_top1 = (ranks[:, 0] == torch.arange(B, device=ranks.device)).float().mean().item()

    print(f"[Eval {modality}] Retrieval Recall@1 on {B} samples: {correct_top1:.3f}")
    return correct_top1


from transformers import get_cosine_schedule_with_warmup
def training_function():
    # 1. Initialize Accelerator FIRST
    # We must create it here so each process gets its own instance.
    accelerator = Accelerator(mixed_precision="bf16", log_with="wandb")
    device = accelerator.device

    # --- move all modules to device BEFORE training ---
    perceiver.to(device)
    projector.to(device)
    # pooled_modality_embedding.to(device)
    qwen_model.to(device)   # <-- IMPORTANT

    # Now we can safely use it
    print(f"Process {accelerator.process_index} starting...")

    # 2. Setup Optimizer (Must be created fresh inside the function)
    optimizer = AdamW(
        [p for p in trainable_modules.parameters() if p.requires_grad],
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    # 3. Setup Scheduler
    total_steps = (cfg.max_train_steps_vision + cfg.max_train_steps_audio) * cfg.num_rounds
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(0.1 * total_steps), 
        num_training_steps=total_steps
    )

    # 4. Prepare everything ONCE
    # We reference the global models/loaders, but Accelerate wraps them for this specific process.
    # global vision_adapter, audio_adapter, perceiver, projector, trainable_modules
    (
        vision_adapter_acc, 
        audio_adapter_acc, 
        perceiver_acc, 
        projector_acc, 
        optimizer_acc, 
        vision_loader_acc, 
        audio_loader_acc,
        scheduler_acc
    ) = accelerator.prepare(
        vision_adapter, 
        audio_adapter, 
        perceiver, 
        projector, 
        optimizer, 
        vision_loader, 
        audio_loader,
        scheduler
    )

    # 5. The Training Loop
    for round_idx in range(cfg.num_rounds):
        if accelerator.is_main_process:
            print(f"\n========== Round {round_idx+1}/{cfg.num_rounds} ==========")

        # --- Vision Training ---
        train_one_epoch(
            dataloader=vision_loader_acc,
            modality="vision",
            max_steps=cfg.max_train_steps_vision,
            optimizer=optimizer_acc,
            scheduler=scheduler_acc,
            accelerator=accelerator,
            log_prefix=f"round{round_idx+1}-"
        )
        
        # Wait for all GPUs to finish before evaluation
        accelerator.wait_for_everyone()
        
        # Run eval only on the main process to avoid duplicate logs/prints
        if accelerator.is_main_process:
            eval_retrieval(vision_dataset, modality="vision", num_samples=32)

        # --- Audio Training ---
        train_one_epoch(
            dataloader=audio_loader_acc,
            modality="audio",
            max_steps=cfg.max_train_steps_audio,
            optimizer=optimizer_acc,
            scheduler=scheduler_acc,
            accelerator=accelerator,
            log_prefix=f"round{round_idx+1}-"
        )
        
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            eval_retrieval(audio_dataset, modality="audio", num_samples=32)

    if accelerator.is_main_process:
        print("Training Finished!")


if __name__ == "__main__":
    training_function()
    









# training_function()
# import torch.multiprocessing as mp
# mp.set_start_method("spawn", force=True)
# import os
# os.environ["MASTER_PORT"] = "29999"   # choose any free port
# from accelerate import notebook_launcher

# num_processes = Number of GPUs you have
# notebook_launcher(training_function, num_processes=torch.cuda.device_count())

# # --------------------------------------------
# # 7.3 – Run a small POC training loop
# # --------------------------------------------

# # You can adjust these to be very small for a first run:
# vision_steps = getattr(cfg, "max_train_steps_vision", 100)
# audio_steps  = getattr(cfg, "max_train_steps_audio", 100)

# num_rounds = 1  # or >1 if you want to alternate vision/audio multiple times

# for round_idx in range(num_rounds):
#     print(f"\n========== Training Round {round_idx+1}/{num_rounds} ==========")

#     # ---- Vision–text alignment ----
#     print("\n--- Vision–Text alignment ---")
#     train_one_epoch(
#         dataloader=vision_loader,
#         modality="vision",
#         max_steps=vision_steps,
#         log_prefix=f"round{round_idx+1}-",
#     )
#     eval_retrieval(vision_dataset, modality="vision", num_samples=32)

#     # ---- Audio–text alignment ----
#     print("\n--- Audio–Text alignment ---")
#     train_one_epoch(
#         dataloader=audio_loader,
#         modality="audio",
#         max_steps=audio_steps,
#         log_prefix=f"round{round_idx+1}-",
#     )
#     eval_retrieval(audio_dataset, modality="audio", num_samples=32)

# print("\nTraining POC finished.")

# 














