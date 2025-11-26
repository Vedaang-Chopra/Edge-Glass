"""
================================================================================
MULTIMODAL ALIGNMENT WITH PERCEIVER RESAMPLER
================================================================================

A complete implementation for aligning vision, audio, and text modalities
into a unified embedding space, with optional LLM integration.

Author: Extended from previous alignment work
Date: 2024

================================================================================
ARCHITECTURE OVERVIEW
================================================================================

The system has THREE main phases:

┌─────────────────────────────────────────────────────────────────────────────┐
│                        PHASE 1: FEATURE EXTRACTION                          │
│                           (Frozen Encoders)                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   IMAGE ──► CLIP ViT ──► [CLS, patch1, patch2, ..., patch49] (50×768)      │
│                                                                             │
│   AUDIO ──► Whisper Encoder ──► [frame1, frame2, ..., frameT] (T×512)      │
│                                                                             │
│   TEXT ──► Sentence-BERT ──► [tok1, tok2, ..., tokL] (L×384)               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      PHASE 2: MODALITY ADAPTATION                           │
│                         (Trainable Adapters)                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Each modality has its own adapter to project to common dimension:         │
│                                                                             │
│   Vision Adapter: 768 → 512 (perceiver_dim)                                │
│   Audio Adapter:  512 → 512 (perceiver_dim)                                │
│   Text Adapter:   384 → 512 (perceiver_dim)                                │
│                                                                             │
│   Output: All modalities now have shape (B, T_modality, 512)               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      PHASE 3: PERCEIVER RESAMPLER                           │
│                    (Trainable, Shared Across Modalities)                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   PURPOSE: Compress variable-length sequences to fixed-size latents         │
│                                                                             │
│   Input:  (B, T_variable, 512)  ← different T for each modality            │
│   Output: (B, K_fixed, 512)     ← same K (e.g., 64) for all modalities     │
│                                                                             │
│   MECHANISM:                                                                │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  Learned Latent Queries: (K, D) = (64, 512)                         │  │
│   │                                                                     │  │
│   │  For each layer:                                                    │  │
│   │    1. Cross-Attention: Latents attend to input tokens               │  │
│   │       Q = Latents, K = V = Input Tokens                             │  │
│   │       Latents = Latents + CrossAttn(Q, K, V)                        │  │
│   │                                                                     │  │
│   │    2. Self-Attention: Latents attend to themselves                  │  │
│   │       Latents = Latents + SelfAttn(Latents)                         │  │
│   │                                                                     │  │
│   │    3. Feed-Forward Network                                          │  │
│   │       Latents = Latents + FFN(Latents)                              │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   WHY PERCEIVER?                                                            │
│   • Fixed output size regardless of input length                            │
│   • Memory efficient (O(K×T) instead of O(T²))                             │
│   • Learned queries can focus on task-relevant information                  │
│   • Natural bottleneck for information compression                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       PHASE 4: ALIGNMENT PROJECTION                         │
│                            (Trainable)                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Option A: For Retrieval/Classification (Contrastive Learning)             │
│   ─────────────────────────────────────────────────────────────             │
│   Latents (B, 64, 512) → MeanPool → (B, 512) → Normalize → z_aligned       │
│                                                                             │
│   Loss: InfoNCE / MRL Contrastive Loss                                      │
│   • Bring matching pairs closer                                             │
│   • Push non-matching pairs apart                                           │
│                                                                             │
│   Option B: For LLM Integration (Generation)                                │
│   ─────────────────────────────────────────────────────────────             │
│   Latents (B, 64, 512) → Linear → (B, 64, D_llm) → LLM Prefix Tokens       │
│                                                                             │
│   These become "soft prompts" that the LLM conditions on                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘


================================================================================
TRAINING STRATEGY
================================================================================

PHASE 1 TRAINING: Multimodal Alignment
──────────────────────────────────────
Objective: Align all modalities in shared embedding space

Trainable:
  ✓ Vision Adapter
  ✓ Audio Adapter  
  ✓ Text Adapter (optional, can use frozen text encoder)
  ✓ Perceiver Resampler
  ✓ Alignment Projector

Frozen:
  ✗ CLIP Vision Encoder
  ✗ Whisper Audio Encoder
  ✗ Text Encoder

Loss Function: Matryoshka Representation Learning (MRL) + CLIP Loss
  
  L_total = λ_mrl × L_MRL + λ_clip × L_CLIP
  
  where L_MRL = Σ_r L_contrastive(z[:, :r], z[:, :r])  # Multiple radii
        L_CLIP = L_contrastive(z_full, z_full)         # Full dimension


PHASE 2 TRAINING: LLM Integration
─────────────────────────────────
Objective: Enable LLM to understand multimodal inputs

Trainable:
  ✓ LLM Projector (512 → D_llm)
  ✓ (Optional) LoRA adapters on LLM

Frozen:
  ✗ Everything from Phase 1
  ✗ LLM base weights

Loss Function: Next-token prediction (Language Modeling)
  
  L_LM = -Σ log P(token_i | prefix_tokens, token_1:i-1)


================================================================================
DATA FLOW EXAMPLE
================================================================================

Input: An image of a dog playing fetch

1. CLIP Encoder:
   Image (224×224×3) → Patches → ViT → (1, 50, 768)
   
2. Vision Adapter:
   (1, 50, 768) → Linear → (1, 50, 512)
   
3. Perceiver Resampler:
   Latents (1, 64, 512) initialized randomly
   
   Layer 1:
     CrossAttn: Latents query image patches
     SelfAttn:  Latents refine themselves
     FFN:       Non-linear transformation
   
   Layer 2-6: Same process, progressively refining
   
   Output: (1, 64, 512) - compressed representation
   
4a. For Retrieval:
    MeanPool: (1, 64, 512) → (1, 512)
    Normalize: L2 norm
    Compare with text embeddings using cosine similarity
    
4b. For Generation:
    Project: (1, 64, 512) → (1, 64, 3584)  # Qwen hidden size
    Concatenate with text prompt embeddings
    Generate response autoregressively

================================================================================
"""

import os
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import numpy as np
from tqdm.auto import tqdm


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class MultimodalAlignmentConfig:
    """
    Complete configuration for multimodal alignment with Perceiver.
    
    Architecture Parameters:
    ─────────────────────────
    - d_vision: Output dimension of vision encoder (CLIP ViT-B/32 = 768)
    - d_audio: Output dimension of audio encoder (Whisper-base = 512)
    - d_text: Output dimension of text encoder (MiniLM = 384)
    - perceiver_dim: Internal dimension for Perceiver (512)
    - num_latents: Number of learned latent queries (64)
    - num_perceiver_layers: Depth of Perceiver (2-6)
    - num_attn_heads: Attention heads in Perceiver (8)
    
    Training Parameters:
    ────────────────────
    - mrl_dims: Matryoshka dimensions for nested contrastive learning
    - temperature: Softmax temperature for contrastive loss
    - learning_rate: AdamW learning rate
    - weight_decay: L2 regularization
    
    LLM Integration:
    ────────────────
    - llm_hidden_size: Hidden dimension of target LLM
    - num_prefix_tokens: How many latents to use as LLM prefix
    """
    
    # === Model Names ===
    vision_model_name: str = "openai/clip-vit-base-patch32"
    audio_model_name: str = "openai/whisper-base"
    text_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    
    # === Encoder Dimensions ===
    d_vision: int = 768      # CLIP ViT-B/32 hidden size
    d_audio: int = 512       # Whisper-base encoder hidden size
    d_text: int = 384        # MiniLM hidden size
    
    # === Perceiver Architecture ===
    perceiver_dim: int = 512          # Internal dimension
    num_latents: int = 64             # Number of learned queries
    num_perceiver_layers: int = 4     # Depth (2-6 typical)
    num_attn_heads: int = 8           # Attention heads
    perceiver_mlp_ratio: float = 4.0  # FFN expansion ratio
    perceiver_dropout: float = 0.1    # Dropout rate
    
    # === Alignment ===
    d_align: int = 512                        # Final alignment dimension
    mrl_dims: Tuple[int, ...] = (64, 128, 256, 512)  # Matryoshka radii
    
    # === LLM Projection ===
    llm_hidden_size: int = 1536       # Qwen2.5-1.5B hidden size
    num_prefix_tokens: int = 64       # Latents used as LLM prefix
    
    # === Training ===
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    num_epochs: int = 10
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    
    # === Contrastive Loss ===
    temperature: float = 0.07
    mrl_weight: float = 1.0
    clip_weight: float = 0.5
    
    # === Misc ===
    seed: int = 42
    device: str = "cuda"
    dtype: str = "float32"


# ============================================================================
# PERCEIVER RESAMPLER COMPONENTS
# ============================================================================

class FeedForward(nn.Module):
    """
    Feed-Forward Network (FFN) with GELU activation.
    
    Architecture: Linear → GELU → Dropout → Linear → Dropout
    
    The expansion ratio (mlp_ratio) controls the hidden dimension:
        hidden_dim = dim × mlp_ratio
    
    Typical values: mlp_ratio = 4.0 (following Transformer convention)
    """
    
    def __init__(self, dim: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PerceiverAttention(nn.Module):
    """
    Multi-Head Attention for Perceiver with Pre-LayerNorm.
    
    This is a standard multi-head attention but with:
    1. Pre-LayerNorm on queries and keys/values
    2. Support for cross-attention (Q from latents, KV from inputs)
    3. Support for self-attention (Q, K, V all from same source)
    """
    
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.ln_q = nn.LayerNorm(dim)
        self.ln_kv = nn.LayerNorm(dim)
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        q: torch.Tensor,           # Queries: (B, N_q, D)
        kv: torch.Tensor,          # Keys/Values: (B, N_kv, D)
        mask: Optional[torch.Tensor] = None,  # Key padding mask: (B, N_kv)
    ) -> torch.Tensor:
        """
        Perform multi-head attention.
        
        For cross-attention: q = latents, kv = input tokens
        For self-attention:  q = kv = latents
        """
        B, N_q, D = q.shape
        N_kv = kv.shape[1]
        
        # Pre-LayerNorm
        q = self.ln_q(q)
        kv = self.ln_kv(kv)
        
        # Project to Q, K, V
        Q = self.q_proj(q).view(B, N_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(kv).view(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(kv).view(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores: (B, H, N_q, N_kv)
        attn = (Q @ K.transpose(-2, -1)) * self.scale
        
        # Apply mask if provided (for padding)
        if mask is not None:
            # mask: (B, N_kv) → (B, 1, 1, N_kv)
            mask = mask.unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(~mask.bool(), float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = (attn @ V).transpose(1, 2).reshape(B, N_q, D)
        out = self.out_proj(out)
        
        return out


class PerceiverLayer(nn.Module):
    """
    Single Perceiver Layer.
    
    Architecture:
    ─────────────
    1. Cross-Attention: Latents attend to input tokens
       - Allows latents to "read" information from the input
       - Q = Latents, K = V = Input tokens
       
    2. Self-Attention: Latents attend to themselves
       - Allows latents to share information with each other
       - Q = K = V = Latents
       
    3. Feed-Forward Network
       - Non-linear transformation of each latent independently
    
    All operations use residual connections:
        output = input + Attention(input)
    """
    
    def __init__(
        self, 
        dim: int, 
        num_heads: int = 8, 
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # Cross-attention: latents query input tokens
        self.cross_attn = PerceiverAttention(dim, num_heads, dropout)
        
        # Self-attention: latents attend to each other
        self.self_attn = PerceiverAttention(dim, num_heads, dropout)
        
        # Feed-forward network
        self.ffn = FeedForward(dim, mlp_ratio, dropout)
        self.ln_ffn = nn.LayerNorm(dim)
    
    def forward(
        self,
        latents: torch.Tensor,     # (B, K, D) - learned queries
        tokens: torch.Tensor,      # (B, T, D) - input tokens
        token_mask: Optional[torch.Tensor] = None,  # (B, T) - padding mask
    ) -> torch.Tensor:
        """
        Process one Perceiver layer.
        
        Args:
            latents: Learned latent queries (B, K, D)
            tokens: Input tokens from encoder (B, T, D)
            token_mask: Boolean mask, True for valid tokens (B, T)
        
        Returns:
            Updated latents (B, K, D)
        """
        # 1. Cross-attention: latents read from input tokens
        latents = latents + self.cross_attn(latents, tokens, token_mask)
        
        # 2. Self-attention: latents communicate with each other
        latents = latents + self.self_attn(latents, latents)
        
        # 3. FFN: non-linear transformation
        latents = latents + self.ffn(self.ln_ffn(latents))
        
        return latents


class PerceiverResampler(nn.Module):
    """
    Perceiver Resampler for Variable-Length Sequence Compression.
    
    This is the core component that enables multimodal alignment:
    
    PURPOSE:
    ────────
    • Convert variable-length encoder outputs to fixed-size representations
    • Image: 50 patches → 64 latents
    • Audio: 1500 frames → 64 latents  
    • Text: variable tokens → 64 latents
    
    MECHANISM:
    ──────────
    Uses learned "latent queries" that cross-attend to input tokens.
    Think of it as: "What are the 64 most important aspects of this input?"
    
    The latent queries are learned during training and become specialized
    for extracting task-relevant information.
    
    WHY THIS DESIGN:
    ────────────────
    1. EFFICIENCY: O(K×T) attention instead of O(T²)
       - K = 64 latents, T = input length
       - For long audio (T=1500), this is 23× more efficient
       
    2. FIXED OUTPUT: Regardless of input length, output is always (B, K, D)
       - Easy to compare across modalities
       - Easy to feed to downstream models
       
    3. INFORMATION BOTTLENECK: Forces compression of information
       - Latents must capture the essence of the input
       - Removes redundant/irrelevant details
       
    4. LEARNABLE QUERIES: Queries specialize during training
       - Some may focus on objects, others on actions, etc.
       - Emergent specialization improves representations
    """
    
    def __init__(
        self,
        dim: int,                    # Latent dimension
        num_latents: int = 64,       # Number of learned queries
        num_layers: int = 4,         # Depth of Perceiver
        num_heads: int = 8,          # Attention heads
        mlp_ratio: float = 4.0,      # FFN expansion
        dropout: float = 0.1,        # Dropout rate
    ):
        super().__init__()
        self.dim = dim
        self.num_latents = num_latents
        
        # Learned latent queries - these are the "questions" we ask the input
        # Initialized with small random values (scaled by 1/sqrt(dim))
        self.latents = nn.Parameter(
            torch.randn(num_latents, dim) * (dim ** -0.5)
        )
        
        # Stack of Perceiver layers
        self.layers = nn.ModuleList([
            PerceiverLayer(dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.ln_out = nn.LayerNorm(dim)
    
    def forward(
        self,
        tokens: torch.Tensor,        # (B, T, D) - input tokens
        token_mask: Optional[torch.Tensor] = None,  # (B, T) - padding mask
    ) -> torch.Tensor:
        """
        Resample variable-length input to fixed-size latents.
        
        Args:
            tokens: Input tokens from adapter (B, T, D)
            token_mask: Boolean mask for valid tokens (B, T)
        
        Returns:
            latents: Fixed-size representation (B, K, D)
        """
        B = tokens.shape[0]
        
        # Expand latents for batch: (K, D) → (B, K, D)
        latents = self.latents.unsqueeze(0).expand(B, -1, -1)
        
        # Process through Perceiver layers
        for layer in self.layers:
            latents = layer(latents, tokens, token_mask)
        
        # Final normalization
        latents = self.ln_out(latents)
        
        return latents


# ============================================================================
# MODALITY ADAPTERS
# ============================================================================

class ModalityAdapter(nn.Module):
    """
    Simple Linear Adapter for projecting encoder outputs.
    
    Maps encoder dimension to Perceiver dimension:
        Vision: 768 → 512
        Audio:  512 → 512
        Text:   384 → 512
    """
    
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class MLPAdapter(nn.Module):
    """
    MLP Adapter with LayerNorm and non-linearity.
    
    More expressive than linear adapter:
        LayerNorm → Linear → GELU → Dropout → Linear
    
    Use this when you need more transformation capacity.
    """
    
    def __init__(
        self, 
        in_dim: int, 
        out_dim: int, 
        hidden_factor: float = 2.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        hidden_dim = int(in_dim * hidden_factor)
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ============================================================================
# PROJECTION HEADS
# ============================================================================

class AlignmentProjector(nn.Module):
    """
    Projects Perceiver latents to alignment space for contrastive learning.
    
    For retrieval tasks:
        Latents (B, K, D) → Pool → (B, D) → Project → (B, d_align)
    
    Pooling options:
        - 'mean': Average all latents
        - 'first': Use first latent (like [CLS] token)
        - 'attention': Learned attention pooling
    """
    
    def __init__(
        self, 
        input_dim: int, 
        output_dim: int,
        pooling: str = 'mean',
    ):
        super().__init__()
        self.pooling = pooling
        
        if pooling == 'attention':
            self.attn_pool = nn.Sequential(
                nn.Linear(input_dim, 1),
                nn.Softmax(dim=1),
            )
        
        self.proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, output_dim),
        )
    
    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Project latents to alignment space.
        
        Args:
            latents: (B, K, D) from Perceiver
        
        Returns:
            z: (B, d_align) aligned embedding
        """
        if self.pooling == 'mean':
            pooled = latents.mean(dim=1)  # (B, D)
        elif self.pooling == 'first':
            pooled = latents[:, 0]  # (B, D)
        elif self.pooling == 'attention':
            weights = self.attn_pool(latents)  # (B, K, 1)
            pooled = (latents * weights).sum(dim=1)  # (B, D)
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")
        
        return self.proj(pooled)


class LLMProjector(nn.Module):
    """
    Projects Perceiver latents to LLM embedding space.
    
    For generation tasks:
        Latents (B, K, D) → Linear → (B, K, D_llm)
    
    These become prefix tokens that the LLM conditions on.
    """
    
    def __init__(self, input_dim: int, llm_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, llm_dim),
        )
    
    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Project latents to LLM space.
        
        Args:
            latents: (B, K, D) from Perceiver
        
        Returns:
            prefix: (B, K, D_llm) prefix tokens for LLM
        """
        return self.proj(latents)


# ============================================================================
# COMPLETE MULTIMODAL ALIGNMENT MODEL
# ============================================================================

class MultimodalAlignmentModel(nn.Module):
    """
    Complete Multimodal Alignment Model with Perceiver Resampler.
    
    ARCHITECTURE SUMMARY:
    ─────────────────────
    
    ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
    │   Vision    │     │    Audio    │     │    Text     │
    │   Encoder   │     │   Encoder   │     │   Encoder   │
    │  (Frozen)   │     │  (Frozen)   │     │  (Frozen)   │
    └──────┬──────┘     └──────┬──────┘     └──────┬──────┘
           │                   │                   │
           ▼                   ▼                   ▼
    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
    │   Vision     │    │    Audio     │    │    Text      │
    │   Adapter    │    │   Adapter    │    │   Adapter    │
    │ (Trainable)  │    │ (Trainable)  │    │ (Trainable)  │
    └──────┬───────┘    └──────┬───────┘    └──────┬───────┘
           │                   │                   │
           └───────────────────┼───────────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │     Perceiver       │
                    │     Resampler       │
                    │    (Trainable)      │
                    │                     │
                    │  Variable → Fixed   │
                    │    (T, D) → (K, D)  │
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
              ▼                ▼                ▼
    ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
    │   Alignment     │ │      LLM        │ │    Latent       │
    │   Projector     │ │   Projector     │ │   Inspection    │
    │  (For Retrieval)│ │(For Generation) │ │  (For Debug)    │
    └─────────────────┘ └─────────────────┘ └─────────────────┘
    
    
    USAGE:
    ──────
    
    # Initialize
    config = MultimodalAlignmentConfig()
    model = MultimodalAlignmentModel(config)
    
    # Forward pass (vision example)
    vision_features = clip_encoder(images)  # (B, 50, 768)
    z_vision = model.encode_vision(vision_features)  # (B, d_align)
    
    # For retrieval
    z_vision = model.encode_vision(vision_features)
    z_text = model.encode_text(text_features)
    similarity = z_vision @ z_text.T  # Cosine similarity
    
    # For generation
    prefix_tokens = model.project_to_llm(vision_features, modality='vision')
    # Use prefix_tokens with LLM.generate()
    """
    
    def __init__(self, config: MultimodalAlignmentConfig):
        super().__init__()
        self.config = config
        
        # === Modality Adapters ===
        # Project each modality to common Perceiver dimension
        self.vision_adapter = MLPAdapter(
            config.d_vision, 
            config.perceiver_dim,
            hidden_factor=2.0,
            dropout=config.perceiver_dropout,
        )
        
        self.audio_adapter = MLPAdapter(
            config.d_audio,
            config.perceiver_dim,
            hidden_factor=2.0,
            dropout=config.perceiver_dropout,
        )
        
        self.text_adapter = MLPAdapter(
            config.d_text,
            config.perceiver_dim,
            hidden_factor=2.0,
            dropout=config.perceiver_dropout,
        )
        
        # === Shared Perceiver Resampler ===
        # Same Perceiver processes all modalities
        self.perceiver = PerceiverResampler(
            dim=config.perceiver_dim,
            num_latents=config.num_latents,
            num_layers=config.num_perceiver_layers,
            num_heads=config.num_attn_heads,
            mlp_ratio=config.perceiver_mlp_ratio,
            dropout=config.perceiver_dropout,
        )
        
        # === Projection Heads ===
        # For contrastive alignment (retrieval)
        self.alignment_projector = AlignmentProjector(
            config.perceiver_dim,
            config.d_align,
            pooling='mean',
        )
        
        # For LLM integration (generation)
        self.llm_projector = LLMProjector(
            config.perceiver_dim,
            config.llm_hidden_size,
        )
    
    def _encode_modality(
        self,
        features: torch.Tensor,
        adapter: nn.Module,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Internal method to encode any modality through adapter + perceiver.
        
        Args:
            features: Encoder output (B, T, D_encoder)
            adapter: Modality-specific adapter
            mask: Padding mask (B, T)
        
        Returns:
            latents: Perceiver output (B, K, D)
            z_aligned: Alignment embedding (B, d_align)
        """
        # Project to Perceiver dimension
        tokens = adapter(features)  # (B, T, perceiver_dim)
        
        # Pass through Perceiver
        latents = self.perceiver(tokens, mask)  # (B, K, perceiver_dim)
        
        # Project to alignment space
        z_aligned = self.alignment_projector(latents)  # (B, d_align)
        
        return latents, z_aligned
    
    def encode_vision(
        self,
        features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_latents: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Encode vision features to alignment space.
        
        Args:
            features: CLIP features (B, T, 768)
            mask: Valid token mask (B, T)
            return_latents: If True, also return Perceiver latents
        
        Returns:
            z: Aligned embedding (B, d_align)
            latents: (optional) Perceiver latents (B, K, D)
        """
        latents, z = self._encode_modality(features, self.vision_adapter, mask)
        
        if return_latents:
            return z, latents
        return z
    
    def encode_audio(
        self,
        features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_latents: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Encode audio features to alignment space.
        
        Args:
            features: Whisper features (B, T, 512)
            mask: Valid token mask (B, T)
            return_latents: If True, also return Perceiver latents
        
        Returns:
            z: Aligned embedding (B, d_align)
            latents: (optional) Perceiver latents (B, K, D)
        """
        latents, z = self._encode_modality(features, self.audio_adapter, mask)
        
        if return_latents:
            return z, latents
        return z
    
    def encode_text(
        self,
        features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_latents: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Encode text features to alignment space.
        
        Args:
            features: Text encoder features (B, T, 384)
            mask: Valid token mask (B, T)
            return_latents: If True, also return Perceiver latents
        
        Returns:
            z: Aligned embedding (B, d_align)
            latents: (optional) Perceiver latents (B, K, D)
        """
        latents, z = self._encode_modality(features, self.text_adapter, mask)
        
        if return_latents:
            return z, latents
        return z
    
    def project_to_llm(
        self,
        features: torch.Tensor,
        modality: str,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Project modality features to LLM embedding space for generation.
        
        Args:
            features: Encoder features (B, T, D)
            modality: 'vision', 'audio', or 'text'
            mask: Valid token mask (B, T)
        
        Returns:
            prefix_tokens: LLM prefix embeddings (B, K, D_llm)
        """
        # Select adapter
        if modality == 'vision':
            adapter = self.vision_adapter
        elif modality == 'audio':
            adapter = self.audio_adapter
        elif modality == 'text':
            adapter = self.text_adapter
        else:
            raise ValueError(f"Unknown modality: {modality}")
        
        # Get latents
        tokens = adapter(features)
        latents = self.perceiver(tokens, mask)
        
        # Project to LLM space
        prefix_tokens = self.llm_projector(latents)
        
        return prefix_tokens
    
    def forward(
        self,
        vision_features: Optional[torch.Tensor] = None,
        audio_features: Optional[torch.Tensor] = None,
        text_features: Optional[torch.Tensor] = None,
        vision_mask: Optional[torch.Tensor] = None,
        audio_mask: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for all provided modalities.
        
        Returns dict with aligned embeddings for each modality.
        """
        outputs = {}
        
        if vision_features is not None:
            outputs['z_vision'] = self.encode_vision(vision_features, vision_mask)
        
        if audio_features is not None:
            outputs['z_audio'] = self.encode_audio(audio_features, audio_mask)
        
        if text_features is not None:
            outputs['z_text'] = self.encode_text(text_features, text_mask)
        
        return outputs


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

def contrastive_loss(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Symmetric InfoNCE contrastive loss (CLIP-style).
    
    Brings matching pairs (diagonal) together, pushes non-matching apart.
    
    Args:
        z_a: Embeddings from modality A (B, D)
        z_b: Embeddings from modality B (B, D)
        temperature: Softmax temperature (lower = sharper)
    
    Returns:
        loss: Scalar loss value
    """
    # Normalize
    z_a = F.normalize(z_a, dim=-1)
    z_b = F.normalize(z_b, dim=-1)
    
    # Similarity matrix
    logits = z_a @ z_b.T / temperature  # (B, B)
    
    # Labels: diagonal entries are positive pairs
    labels = torch.arange(z_a.size(0), device=z_a.device)
    
    # Symmetric loss
    loss_a2b = F.cross_entropy(logits, labels)
    loss_b2a = F.cross_entropy(logits.T, labels)
    
    return (loss_a2b + loss_b2a) / 2


def matryoshka_loss(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    dims: Tuple[int, ...] = (64, 128, 256, 512),
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Matryoshka Representation Learning (MRL) loss.
    
    Applies contrastive loss at multiple embedding dimensions.
    This encourages the model to pack important information in early dimensions.
    
    BENEFITS:
    ─────────
    • Flexible inference: Use fewer dimensions for faster retrieval
    • Better gradients: Multiple objectives provide richer signal
    • Implicit curriculum: Smaller dimensions are harder, provide challenge
    
    Args:
        z_a, z_b: Full-dimension embeddings (B, D)
        dims: Tuple of dimensions to evaluate at
        temperature: Contrastive temperature
    
    Returns:
        loss: Weighted average of losses at each dimension
    """
    total_loss = 0.0
    num_dims = 0
    
    for dim in dims:
        if dim > z_a.size(-1):
            continue
        
        # Truncate to first `dim` dimensions
        z_a_trunc = z_a[:, :dim]
        z_b_trunc = z_b[:, :dim]
        
        # Compute contrastive loss at this dimension
        loss = contrastive_loss(z_a_trunc, z_b_trunc, temperature)
        total_loss += loss
        num_dims += 1
    
    return total_loss / num_dims if num_dims > 0 else total_loss


# ============================================================================
# TRAINING UTILITIES
# ============================================================================

def build_optimizer(
    model: MultimodalAlignmentModel,
    config: MultimodalAlignmentConfig,
) -> torch.optim.Optimizer:
    """Build AdamW optimizer with weight decay."""
    
    # Separate parameters that should/shouldn't have weight decay
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'bias' in name or 'ln' in name or 'layernorm' in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    optimizer = torch.optim.AdamW([
        {'params': decay_params, 'weight_decay': config.weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0},
    ], lr=config.learning_rate)
    
    return optimizer


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count trainable and total parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total,
        'trainable': trainable,
        'frozen': total - trainable,
    }


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("MULTIMODAL ALIGNMENT WITH PERCEIVER RESAMPLER")
    print("="*70)
    
    # Configuration
    config = MultimodalAlignmentConfig(
        perceiver_dim=512,
        num_latents=64,
        num_perceiver_layers=4,
        d_align=512,
    )
    
    # Create model
    model = MultimodalAlignmentModel(config)
    
    # Print architecture
    print("\n📐 Architecture:")
    params = count_parameters(model)
    print(f"   Total parameters: {params['total']:,}")
    print(f"   Trainable: {params['trainable']:,}")
    
    # Test forward pass
    print("\n🧪 Testing forward pass...")
    
    # Simulate encoder outputs
    batch_size = 4
    vision_feats = torch.randn(batch_size, 50, config.d_vision)   # CLIP: 50 patches
    audio_feats = torch.randn(batch_size, 1500, config.d_audio)   # Whisper: ~1500 frames
    text_feats = torch.randn(batch_size, 32, config.d_text)       # Text: 32 tokens
    
    # Encode each modality
    z_vision = model.encode_vision(vision_feats)
    z_audio = model.encode_audio(audio_feats)
    z_text = model.encode_text(text_feats)
    
    print(f"   Vision embedding: {z_vision.shape}")  # (4, 512)
    print(f"   Audio embedding: {z_audio.shape}")    # (4, 512)
    print(f"   Text embedding: {z_text.shape}")      # (4, 512)
    
    # Test LLM projection
    llm_prefix = model.project_to_llm(vision_feats, 'vision')
    print(f"   LLM prefix: {llm_prefix.shape}")      # (4, 64, 1536)
    
    # Test loss computation
    print("\n📉 Testing loss computation...")
    
    loss_clip = contrastive_loss(z_vision, z_text)
    loss_mrl = matryoshka_loss(z_vision, z_text, dims=config.mrl_dims)
    
    print(f"   CLIP loss: {loss_clip.item():.4f}")
    print(f"   MRL loss: {loss_mrl.item():.4f}")
    
    print("\n✅ All tests passed!")
    print("="*70)
