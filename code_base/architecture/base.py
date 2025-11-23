"""
base.py

Common base classes for model configs, encoders, and decoder-only LLMs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM


@dataclass
class ModelConfig:
    """
    Lightweight config for a single model (encoder or decoder).

    This is separate from your global Config/YAML. You can create it
    from there by passing model_name, device, dtype.
    """
    model_name: str
    device: Optional[str] = None
    dtype: torch.dtype = torch.float16

    def __post_init__(self):
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"


class BaseEncoder(nn.Module):
    """
    Abstract base class for encoders (image, audio, text).
    They all share cfg, device, dtype and implement `encode(...)`.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.dtype = cfg.dtype

    def encode(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement `encode`.")


class DecoderLLM(nn.Module):
    """
    Generic wrapper around a decoder-only LLM (e.g., Llama, Qwen, Phi).

    Responsibilities:
      - Load tokenizer + model
      - Ensure pad token exists
      - Set output_hidden_states=True (for MRL or analysis)
      - Provide `generate` method
    """

    def __init__(
        self,
        cfg: ModelConfig,
        device_map: str | Dict[str, int] = "auto",
        use_fast_tokenizer: bool = True,
    ):
        super().__init__()
        self.cfg = cfg

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_name,
            use_fast=use_fast_tokenizer,
        )

        # Ensure pad_token exists
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Model
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            torch_dtype=cfg.dtype,
            device_map=device_map,
        )

        # Enable hidden states for MRL / analysis
        if hasattr(self.model.config, "output_hidden_states"):
            self.model.config.output_hidden_states = True

        # Freeze (Phase 1: frozen LLM)
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **gen_kwargs: Any,
    ) -> str:
        device = next(self.model.parameters()).device

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
        ).to(device)

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=self.tokenizer.pad_token_id,
            **gen_kwargs,
        )

        # Strip the prompt tokens from the front
        gen_ids = output_ids[0, inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        return text


class BaseTaskDecoder:
    """
    Small helper base class that wraps DecoderLLM for task-specific prompting.

    You subclass this and implement `build_prompt`, then call instances directly:
        answer = my_decoder(input_text)
    """

    def __init__(self, llm: DecoderLLM):
        self.llm = llm

    def build_prompt(self, *args, **kwargs) -> str:
        raise NotImplementedError

    def __call__(self, *args, **kwargs) -> str:
        prompt = self.build_prompt(*args, **kwargs)
        return self.llm.generate(prompt)
