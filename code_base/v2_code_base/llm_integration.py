"""
llm_integration.py - Phase 2: Connect aligned embeddings to LLM decoder

This module provides:
- LLM wrapper for generation
- Prefix-based multimodal input
- Simple fine-tuning setup
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
)

from core import VisionTextAligner, AlignmentConfig, get_device


# ============================================================
# Configuration
# ============================================================

@dataclass
class LLMConfig:
    """Configuration for LLM integration."""
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    num_prefix_tokens: int = 8  # Number of learned prefix tokens
    freeze_llm: bool = True     # Phase 2a: freeze LLM, train projector only


# ============================================================
# LLM Wrapper
# ============================================================

class LLMDecoder(nn.Module):
    """
    Wrapper around a causal LLM for multimodal generation.
    
    The aligned vision embeddings are projected to the LLM's embedding space
    and prepended as a "soft prompt" before the text input.
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        freeze: bool = True,
    ):
        super().__init__()
        self.device = device or get_device()
        self.dtype = dtype
        
        print(f"[LLMDecoder] Loading {model_name}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=None,  # Manual placement
        )
        self.model.to(self.device)
        
        if freeze:
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad = False
        
        self.hidden_size = self.model.config.hidden_size
        print(f"[LLMDecoder] hidden_size={self.hidden_size}, frozen={freeze}")
    
    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get token embeddings from input IDs."""
        return self.model.get_input_embeddings()(input_ids)
    
    @torch.no_grad()
    def generate(
        self,
        prefix_embeds: Optional[torch.Tensor] = None,
        prompt: str = "",
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """
        Generate text with optional prefix embeddings.
        
        Args:
            prefix_embeds: (1, num_prefix, hidden_size) vision prefix
            prompt: Text prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
        
        Returns:
            Generated text string
        """
        # Tokenize prompt
        tokens = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)
        
        if prefix_embeds is None:
            # Text-only generation
            outputs = self.model.generate(
                **tokens,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        else:
            # Multimodal generation with prefix
            text_embeds = self.get_input_embeddings(tokens["input_ids"])
            
            # Concatenate prefix + text embeddings
            inputs_embeds = torch.cat([prefix_embeds, text_embeds], dim=1)
            
            # Update attention mask
            prefix_len = prefix_embeds.size(1)
            prefix_mask = torch.ones(1, prefix_len, device=self.device, dtype=tokens["attention_mask"].dtype)
            attention_mask = torch.cat([prefix_mask, tokens["attention_mask"]], dim=1)
            
            outputs = self.model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        # Decode
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated


# ============================================================
# Vision-to-LLM Projector
# ============================================================

class VisionToLLMProjector(nn.Module):
    """
    Projects aligned vision embeddings to LLM embedding space.
    
    Takes the aligned vision embedding (d_align) and projects it to
    a sequence of tokens in LLM space (num_tokens × d_llm).
    """
    
    def __init__(
        self,
        d_align: int,
        d_llm: int,
        num_tokens: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_align = d_align
        self.d_llm = d_llm
        self.num_tokens = num_tokens
        
        # Project to multiple tokens
        self.projector = nn.Sequential(
            nn.LayerNorm(d_align),
            nn.Linear(d_align, d_llm * num_tokens),
            nn.GELU(),
            nn.Dropout(dropout),
        )
    
    def forward(self, z_vision: torch.Tensor) -> torch.Tensor:
        """
        Project vision embedding to LLM prefix tokens.
        
        Args:
            z_vision: (B, d_align) aligned vision embedding
        
        Returns:
            (B, num_tokens, d_llm) prefix embeddings
        """
        B = z_vision.size(0)
        out = self.projector(z_vision)  # (B, d_llm * num_tokens)
        return out.view(B, self.num_tokens, self.d_llm)


# ============================================================
# Full Multimodal Model (Phase 2)
# ============================================================

class MultimodalLLM(nn.Module):
    """
    Complete multimodal model: Vision → Alignment → LLM.
    
    Phase 2a (freeze_llm=True):
        - Vision encoder frozen
        - Alignment adapters frozen (from Phase 1)
        - Vision-to-LLM projector trainable
        - LLM frozen
    
    Phase 2b (freeze_llm=False):
        - Everything trainable (LoRA recommended for LLM)
    """
    
    def __init__(
        self,
        aligner: VisionTextAligner,
        llm_config: LLMConfig,
    ):
        super().__init__()
        
        self.aligner = aligner
        self.llm_config = llm_config
        
        # Freeze aligner (use Phase 1 weights)
        for p in self.aligner.parameters():
            p.requires_grad = False
        
        # LLM decoder
        self.llm = LLMDecoder(
            model_name=llm_config.model_name,
            device=aligner.cfg.device,
            dtype=aligner.cfg.dtype,
            freeze=llm_config.freeze_llm,
        )
        
        # Vision-to-LLM projector (trainable)
        self.projector = VisionToLLMProjector(
            d_align=aligner.cfg.d_align,
            d_llm=self.llm.hidden_size,
            num_tokens=llm_config.num_prefix_tokens,
        ).to(aligner.cfg.device, dtype=aligner.cfg.dtype)
        
        print(f"[MultimodalLLM] Projector: {aligner.cfg.d_align} → {llm_config.num_prefix_tokens} × {self.llm.hidden_size}")
    
    def encode_vision_for_llm(self, images) -> torch.Tensor:
        """
        Encode images to LLM prefix tokens.
        
        Args:
            images: List of PIL images or tensor
        
        Returns:
            (B, num_tokens, d_llm) prefix embeddings
        """
        with torch.no_grad():
            z_vision = self.aligner.encode_vision(images)  # (B, d_align)
        
        prefix = self.projector(z_vision)  # (B, num_tokens, d_llm)
        return prefix
    
    @torch.no_grad()
    def generate(
        self,
        images,
        prompt: str = "Describe this image:",
        max_new_tokens: int = 256,
        temperature: float = 0.7,
    ) -> str:
        """
        Generate description for an image.
        
        Args:
            images: Single image or list of images
            prompt: Text prompt
            max_new_tokens: Max tokens to generate
            temperature: Sampling temperature
        
        Returns:
            Generated text
        """
        # Ensure list
        if not isinstance(images, list):
            images = [images]
        
        # Get vision prefix
        prefix = self.encode_vision_for_llm(images)  # (B, num_tokens, d_llm)
        
        # Generate (batch size 1 for now)
        return self.llm.generate(
            prefix_embeds=prefix[:1],
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
    
    def get_trainable_params(self):
        """Get trainable parameters (projector only in Phase 2a)."""
        return [p for p in self.projector.parameters() if p.requires_grad]
    
    def forward(
        self,
        images,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Forward pass for training.
        
        Args:
            images: List of PIL images
            input_ids: (B, T) token IDs
            attention_mask: (B, T) attention mask
            labels: (B, T) labels for language modeling loss
        
        Returns:
            dict with loss and logits
        """
        B = input_ids.size(0)
        device = input_ids.device
        
        # Get vision prefix
        prefix = self.encode_vision_for_llm(images)  # (B, num_tokens, d_llm)
        num_prefix = prefix.size(1)
        
        # Get text embeddings
        text_embeds = self.llm.get_input_embeddings(input_ids)  # (B, T, d_llm)
        
        # Concatenate
        inputs_embeds = torch.cat([prefix, text_embeds], dim=1)  # (B, num_prefix + T, d_llm)
        
        # Extend attention mask
        prefix_mask = torch.ones(B, num_prefix, device=device, dtype=attention_mask.dtype)
        full_attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        
        # Extend labels (ignore prefix tokens in loss)
        if labels is not None:
            prefix_labels = torch.full((B, num_prefix), -100, device=device, dtype=labels.dtype)
            full_labels = torch.cat([prefix_labels, labels], dim=1)
        else:
            full_labels = None
        
        # Forward through LLM
        outputs = self.llm.model(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention_mask,
            labels=full_labels,
        )
        
        return {
            "loss": outputs.loss if outputs.loss is not None else None,
            "logits": outputs.logits,
        }


# ============================================================
# Phase 2 Training Utilities
# ============================================================

def train_multimodal_step(
    model: MultimodalLLM,
    batch: Dict[str, Any],
    optimizer: torch.optim.Optimizer,
) -> Dict[str, float]:
    """
    Single training step for multimodal model.
    
    Expected batch:
        {
            "images": List of PIL images,
            "input_ids": (B, T) tensor,
            "attention_mask": (B, T) tensor,
            "labels": (B, T) tensor,
        }
    """
    optimizer.zero_grad()
    
    outputs = model(
        images=batch["images"],
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
    )
    
    loss = outputs["loss"]
    loss.backward()
    
    torch.nn.utils.clip_grad_norm_(model.get_trainable_params(), 1.0)
    optimizer.step()
    
    return {"loss": loss.item()}


def create_caption_labels(
    tokenizer,
    captions: List[str],
    max_length: int = 256,
    device: torch.device = None,
) -> Dict[str, torch.Tensor]:
    """
    Create input_ids, attention_mask, and labels for caption training.
    """
    tokens = tokenizer(
        captions,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    
    input_ids = tokens["input_ids"]
    attention_mask = tokens["attention_mask"]
    
    # Labels are same as input_ids (autoregressive), padding is -100
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    
    if device:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }
