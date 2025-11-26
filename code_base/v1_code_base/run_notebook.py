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
# Part 0.1 – Global config
# ============================================

@dataclass
class Config:
    # --- Model names ---
    # Vision encoder (you can swap to your PixMo vision backbone later)
    vision_model_name: str = "openai/clip-vit-base-patch32"

    # Audio encoder (Whisper)
    audio_model_name: str = "openai/whisper-base"
    audio_sample_rate: int = 16000

    # Text encoder/decoder
    llm_model_name: str = "Qwen/Qwen2.5-7B-Instruct"

    # --- Dimensions (will be filled after loading models) ---
    encoder_dim_vision: Optional[int] = None
    encoder_dim_audio: Optional[int] = None
    perceiver_dim: int = 512          # unified bottleneck dim
    llm_hidden_size: Optional[int] = None

    num_latents: int = 64             # Perceiver latent length

    # --- Matryoshka loss (MRL) ---
    use_mrl: bool = True
    mrl_dims: Tuple[int, ...] = (128, 256, 512)
    mrl_temperature: float = 0.07
    mrl_weight: float = 0.1

    # --- Training (we'll tune later) ---
    batch_size_vision: int = 16
    batch_size_audio: int = 16
    max_train_steps_vision: int = 200
    max_train_steps_audio: int = 200
    learning_rate: float = 1e-4
    weight_decay: float = 0.01

        # --- Data paths & limits ---
    # Vision feature cache (you can reuse PixMo / CLIP features here)
    vision_features_root: Path = Path("./features_vision")

    # Audio feature cache for LibriSpeech (Whisper embeddings)
    audio_features_root: Path = Path("./features_audio_librispeech")

    # LibriSpeech POC limits
    librispeech_max_samples: int = 3000   # total train subset for alignment
    max_audio_duration_s: float = 12.0    # filter very long clips

    # --- Misc ---
    seed: int = 42
    log_every_steps: int = 20
    save_dir: Path = Path("./runs_perceiver_mrl_qwen")

cfg = Config()
set_seed(cfg.seed)

cfg.save_dir.mkdir(parents=True, exist_ok=True)
cfg.vision_features_root.mkdir(parents=True, exist_ok=True)
cfg.audio_features_root.mkdir(parents=True, exist_ok=True)

print("Config:", asdict(cfg))


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



# ============================================
# Part 1 – Load models: vision, audio, text (Qwen2.5-7B)
# ============================================

# ------------------------------
# 1.1 Vision encoder (CLIP-style)
# ------------------------------
# For now we use CLIP as a simple vision encoder.
# Later you can swap this for your PixMo vision encoder or precomputed features.

from transformers import CLIPModel, CLIPImageProcessor

print("\nLoading vision encoder:", cfg.vision_model_name)
vision_processor = CLIPImageProcessor.from_pretrained(cfg.vision_model_name)
# Load full CLIP model to avoid config issues with CLIPVisionModel.from_pretrained
full_clip = CLIPModel.from_pretrained(
    cfg.vision_model_name,
    torch_dtype=default_dtype,
    device_map=None,
).to(device)
vision_model = full_clip.vision_model
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
    device_map="auto",
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

# ============================================
# Part 2 – Quick text embedding helper (for later MRL)
# ============================================

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
    enc = qwen_tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(device)

    # (B, L, D)
    token_embs = qwen_model.get_input_embeddings()(enc.input_ids)

    return {
        "input_ids": enc.input_ids,
        "attention_mask": enc.attention_mask,
        "token_embs": token_embs,
    }

print("Text embedding helper ready.")


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

    if dur <= cfg.max_audio_duration_s:
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




from torchvision import transforms
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
    def __init__(self, hf_dataset, vision_model, vision_processor, max_retries: int = 5):
        self.ds = hf_dataset
        self.vision_model = vision_model
        self.vision_processor = vision_processor
        self.max_retries = max_retries
        self.augment = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        ])

    def __len__(self):
        return len(self.ds)

    def _load_image_from_url(self, url: str) -> Image.Image:
        resp = requests.get(url, timeout=10)
        # do NOT let this propagate; we'll catch in __getitem__
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content)).convert("RGB")
        return img

    def _encode_image(self, img: Image.Image):
        proc = self.vision_processor(images=img, return_tensors="pt")
        pixel_values = proc["pixel_values"].to(device)

        with torch.no_grad():
            out = self.vision_model(pixel_values=pixel_values)
            # (1, T, d_vision)
            feats = out.last_hidden_state.squeeze(0).to("cpu")  # (T, d_vision)
        return feats

    def _get_example(self, idx: int):
        ex = self.ds[idx]
        caption = ex[txt_col]

        if HAS_IMAGE_COL:
            # HF has already downloaded/cached images; this is usually a PIL.Image
            img = ex[img_col]
            if not isinstance(img, Image.Image):
                img = img.convert("RGB")
        else:
            url = ex[img_col]
            img = self._load_image_from_url(url)
        # Apply augmentation
        if self.vision_model.training:
            img = self.augment(img)

        feats = self._encode_image(img)
        return {
            "features": feats,
            "text": caption,
        }

    def __getitem__(self, idx: int):
        """
        Try up to max_retries times with different indices if something fails
        (HTTP error, decoding error, etc).
        """
        n = len(self.ds)
        attempt = 0
        cur_idx = idx

        while attempt < self.max_retries:
            try:
                return self._get_example(cur_idx)
            except Exception as e:
                # print(f"[PixmoVisionDataset] Failed idx={cur_idx}, attempt={attempt+1}, err={e}")
                attempt += 1
                cur_idx = (cur_idx + 1) % n

        # Final fallback: try random indices
        for _ in range(self.max_retries):
            j = random.randint(0, n - 1)
            try:
                return self._get_example(j)
            except Exception:
                continue

        raise RuntimeError("PixmoVisionDataset: could not load any valid images after multiple retries.")




# ============================================
# Part 4 – Audio features dataset (LibriSpeech + Whisper)
# ============================================

from torch.utils.data import Dataset

# We assume:
#  - `filtered` has been built in Part 3 (streaming LibriSpeech)
#  - Each entry: {"waveform": np.ndarray, "sampling_rate": int, "duration": float, "text": str}
print("\nBuilding LibriSpeech audio–text dataset from filtered streaming subset...")
print("Filtered LibriSpeech examples:", len(filtered))


from torchaudio import transforms as T_audio
def whisper_encode_sequence(wav: np.ndarray, sr: int):
    """
    wav: 1D numpy array (time,)
    sr:  sampling rate (expected 16k)
    Returns:
        feats: Tensor(T_enc, d_audio) on CPU (float16)
    """
    # WhisperProcessor: raw waveform -> log-Mel spectrogram features
    inputs = audio_processor(
        wav,
        sampling_rate=sr,
        return_tensors="pt",
    )
    input_features = inputs["input_features"].to(device)  # (1, T_mel, 80)
    # Apply SpecAugment if training (assuming global audio_model.training check or similar)
    # Since this function is global, we check audio_model.training
    if audio_model.training:
        freq_mask = T_audio.FrequencyMasking(freq_mask_param=15)
        time_mask = T_audio.TimeMasking(time_mask_param=35)
        input_features = freq_mask(input_features)
        input_features = time_mask(input_features)

    with torch.no_grad():
        enc_out = audio_model.encoder(input_features)
        hidden = enc_out.last_hidden_state  # (1, T_enc, d_audio)

    feats = hidden.squeeze(0).to(torch.float16).cpu()  # (T_enc, d_audio)
    return feats


class LibriSpeechAudioDataset(Dataset):
    """
    Dataset over the in-memory filtered LibriSpeech examples.
    Returns:
        {
          "features": Tensor(T_enc, d_audio),
          "text": str,
          "duration": float
        }
    """
    def __init__(self, examples, max_len: int | None = None):
        self.examples = examples
        if max_len is not None and max_len < len(examples):
            # Optionally cut down further for faster experiments
            self.examples = examples[:max_len]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int):
        ex = self.examples[idx]
        wav = ex["waveform"]
        sr = ex["sampling_rate"]
        text = ex["text"]
        dur = ex["duration"]

        feats = whisper_encode_sequence(wav, sr)  # (T_enc, d_audio)

        return {
            "features": feats,
            "text": text,
            "duration": dur,
        }


audio_max = getattr(cfg, "librispeech_max_samples", len(filtered))
audio_dataset = LibriSpeechAudioDataset(filtered, max_len=audio_max)

print("Audio dataset ready. Example:")
sample_a = audio_dataset[0]
print("  features shape:", sample_a["features"].shape)
print("  duration:", round(sample_a["duration"], 2), "s")
print("  text:", sample_a["text"])


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

class AttentionPooling(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim))
        self.scale = dim ** -0.5
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # x: (B, L, D)
        B, L, D = x.shape
        q = self.query.expand(B, -1, -1) # (B, 1, D)
        scores = torch.matmul(q, x.transpose(1, 2)) * self.scale # (B, 1, L)
        if mask is not None:
             # mask is (B, L) bool, True=valid
             # we want to mask INVALID positions (False) with -inf
             # scores is (B, 1, L)
             m = mask.unsqueeze(1) # (B, 1, L)
             scores = scores.masked_fill(~m, float('-inf'))
        attn = F.softmax(scores, dim=-1) # (B, 1, L)
        out = torch.matmul(attn, x) # (B, 1, D)
        return out.squeeze(1) # (B, D)

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
pooler = AttentionPooling(cfg.llm_hidden_size).to(device)
print("Pooler created:", pooler)
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
    assert z_mod.shape == z_txt.shape
    B, D = z_mod.shape
    max_d = max(trunc_dims)
    assert max_d <= D, f"Max trunc dim {max_d} exceeds embedding dim {D}"

    losses = []
    targets = torch.arange(B, device=z_mod.device)

    for d in trunc_dims:
        zm = F.normalize(z_mod[:, :d], dim=-1)  # (B, d)
        zt = F.normalize(z_txt[:, :d], dim=-1)  # (B, d)

        logits = zm @ zt.T / temperature        # (B, B)
        loss_m2t = F.cross_entropy(logits, targets)
        loss_t2m = F.cross_entropy(logits.T, targets)

        losses.append(0.5 * (loss_m2t + loss_t2m))

    return sum(losses) / len(losses)



# --------------------------------------------
# 6.3 – Helpers for global text & modality embeddings
# --------------------------------------------

def pooled_text_embedding(texts: list[str], max_length: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
        h_text: (B, D_llm) pooled text embeddings
        text_tok_info: dict with token_embs, input_ids, attention_mask
    """
    tok_out = encode_text_with_qwen(texts, max_length=max_length)  # uses qwen_model embedding layer
    token_embs = tok_out["token_embs"]          # (B, L, D_llm)
    attn_mask = tok_out["attention_mask"]      # (B, L)

    # masked mean-pooling over tokens
    mask = attn_mask.unsqueeze(-1)             # (B, L, 1)
    denom = mask.sum(dim=1).clamp_min(1)       # (B, 1)
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
    # 4) Global modality embedding (Attention Pooling)
    # We don't have a mask for latents (they are fixed size), so mask=None
    h_mod = pooler(z_llm)            # (B, D_llm)

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

def forward_alignment_step(
    batch: dict,
    modality: str = "vision",   # "vision" or "audio"
) -> tuple[torch.Tensor, dict]:
    """
    One step of alignment loss for a batch.
    """
    encoder_feats = batch["encoder_feats"].to(device)   # (B, T, D_enc)
    encoder_mask  = batch["encoder_mask"].to(device)    # (B, T)
    texts         = batch["texts"]                      # list[str]

    # 1) Modality adapter -> Perceiver dim
    if modality == "vision":
        tokens = vision_adapter(encoder_feats)          # (B, T, D_perc)
    elif modality == "audio":
        tokens = audio_adapter(encoder_feats)           # (B, T, D_perc)
    else:
        raise ValueError(f"Unknown modality: {modality}")

    # 2) Perceiver resampler -> latent tokens
    latents = perceiver(tokens, encoder_mask)           # (B, L, D_perc)

    # 3) Project to Qwen hidden space
    z_llm = projector(latents)                          # (B, L, D_llm)
    
    # Safety check: Ensure projector output matches config
    assert z_llm.shape[-1] == cfg.llm_hidden_size, \
        f"Projector output {z_llm.shape[-1]} != Config {cfg.llm_hidden_size}"

    # 4) Global modality embedding (Mean Pooling over latents)
    # 4) Global modality embedding (Attention Pooling)
    # We don't have a mask for latents (they are fixed size), so mask=None
    h_mod = pooler(z_llm)            # (B, D_llm)

    # 5) Global text embedding from Qwen (Pre-computed or on-the-fly)
    # Note: encode_text_with_qwen returns raw embeddings, not LM outputs, 
    # which is correct for alignment.
    h_txt, tok_info = pooled_text_embedding(texts, max_length=64)  # (B, D_llm)

    # 6) Matryoshka contrastive loss
    # We pass the corrected cfg.mrl_dims here (e.g., 128, 256, 512, 3584)
    mrl_loss = matryoshka_contrastive_loss(
        h_mod,
        h_txt,
        trunc_dims=cfg.mrl_dims,
        temperature=cfg.mrl_temperature,
    )

    metrics = {
        "loss":        float(mrl_loss.detach().cpu()),
        "mrl_loss":    float(mrl_loss.detach().cpu()),
        "modality":    modality,
        "batch_size":  int(h_mod.size(0)),
    }

    return mrl_loss, metrics
print("\nPart 6 ready: collate, MRL, and forward_alignment_step defined.")


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
    pooler,
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

def train_one_epoch(
    dataloader: DataLoader,
    modality: str,
    max_steps: int,
    log_prefix: str = "",
):
    trainable_modules.train()
    running_loss = 0.0
    num_batches = 0
    grad_accum_steps = getattr(cfg, 'grad_accum_steps', 4)

    pbar = tqdm(dataloader, total=max_steps, desc=f"{log_prefix}train-{modality}", leave=False)

    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(pbar, start=1):
        if step > max_steps:
            break

        loss, metrics = forward_alignment_step(batch, modality=modality)
        loss = loss / grad_accum_steps
        loss.backward()

        if step % grad_accum_steps == 0:
            if hasattr(cfg, "max_grad_norm") and cfg.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(trainable_modules.parameters(), cfg.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        running_loss += metrics["loss"]
        num_batches += 1
        avg_loss = running_loss / num_batches

        # ✅ W&B logging
        wandb.log(
            {
                f"{modality}/train/loss": metrics["loss"],
                f"{modality}/train/avg_loss": avg_loss,
                f"{modality}/train/mrl_loss": metrics["mrl_loss"],
                f"{modality}/train/batch_size": metrics["batch_size"],
            }
        )

        if step % cfg.log_every_steps == 0 or step == 1:
            pbar.set_postfix({
                "loss": f"{metrics['loss']:.4f}",
                "avg_loss": f"{avg_loss:.4f}",
            })

    avg_epoch_loss = running_loss / max(1, num_batches)
    print(f"[{log_prefix} {modality}] avg loss: {avg_epoch_loss:.4f}")
    # ✅ epoch-level log
    wandb.log({f"{modality}/train/epoch_loss": avg_epoch_loss})
    return avg_epoch_loss



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



# Scheduler
total_steps = (vision_steps + audio_steps) * num_rounds
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)

# --------------------------------------------
# 7.3 – Run a small POC training loop
# --------------------------------------------

# You can adjust these to be very small for a first run:
vision_steps = getattr(cfg, "max_train_steps_vision", 100)
audio_steps  = getattr(cfg, "max_train_steps_audio", 100)

num_rounds = 1  # or >1 if you want to alternate vision/audio multiple times

for round_idx in range(num_rounds):
    print(f"\n========== Training Round {round_idx+1}/{num_rounds} ==========")

    # ---- Vision–text alignment ----
    print("\n--- Vision–Text alignment ---")
    train_one_epoch(
        dataloader=vision_loader,
        modality="vision",
        max_steps=vision_steps,
        log_prefix=f"round{round_idx+1}-",
    )
    eval_retrieval(vision_dataset, modality="vision", num_samples=32)
    scheduler.step()
    print(f"LR: {scheduler.get_last_lr()[0]:.6f}")

    # ---- Audio–text alignment ----
    print("\n--- Audio–Text alignment ---")
    train_one_epoch(
        dataloader=audio_loader,
        modality="audio",
        max_steps=audio_steps,
        log_prefix=f"round{round_idx+1}-",
    )
    eval_retrieval(audio_dataset, modality="audio", num_samples=32)
    scheduler.step()
    print(f"LR: {scheduler.get_last_lr()[0]:.6f}")

print("\nTraining POC finished.")




































