"""
models_oop.py

Generic OOP wrappers for:
- Image encoder (CLIP / SigLIP style)
- Audio encoder (Whisper style)
- Decoder-only LLM
- Simple task-specific decoders (text/audio/video)

You import these in your main IPYNB and just instantiate the classes.
"""

from dataclasses import dataclass
from typing import List, Optional, Union, Dict, Any

import torch
import torch.nn as nn

from transformers import (
    AutoProcessor,
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
)
from transformers import WhisperProcessor, WhisperModel

# If you use PIL images:
try:
    from PIL import Image
except ImportError:
    Image = Any  # type: ignore


# ==============================
#  Common config & base classes
# ==============================

@dataclass
class ModelConfig:
    model_name: str
    device: Optional[str] = None
    dtype: torch.dtype = torch.float16

    def __post_init__(self):
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"


class BaseEncoder(nn.Module):
    """
    Abstract base class for encoders.
    All encoders should implement `encode(...)` and return a tensor.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.device = cfg.device
        self.dtype = cfg.dtype

    def encode(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement `encode`.")


# ==============================
#  Image Encoder (CLIP / SigLIP)
# ==============================

class ImageEncoder(BaseEncoder):
    """
    Generic image encoder using Hugging Face vision-text models
    such as CLIP / SigLIP.

    Usage:
        cfg = ModelConfig("openai/clip-vit-base-patch32")
        img_enc = ImageEncoder(cfg)
        emb = img_enc.encode_pil(list_of_pil_images)
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)

        # AutoProcessor works for CLIP/SigLIP etc.
        self.processor = AutoProcessor.from_pretrained(cfg.model_name)
        self.model = AutoModel.from_pretrained(
            cfg.model_name, torch_dtype=cfg.dtype
        ).to(cfg.device)

        self.model.eval()

    @torch.no_grad()
    def encode_pil(self, images: List[Image.Image]) -> torch.Tensor:
        """
        Encode a batch of PIL images into embeddings.
        """
        inputs = self.processor(
            images=images,
            return_tensors="pt"
        ).to(self.device)

        outputs = self.model(**inputs)

        # For CLIP-like models:
        if hasattr(outputs, "image_embeds"):
            img_emb = outputs.image_embeds  # (B, D)
        else:
            # Generic fallback: mean-pool last_hidden_state
            img_emb = outputs.last_hidden_state.mean(dim=1)

        return img_emb

    def encode(self, images: List[Image.Image]) -> torch.Tensor:
        return self.encode_pil(images)


# ==============================
#  Audio Encoder (Whisper)
# ==============================

class AudioEncoder(BaseEncoder):
    """
    Generic audio encoder using Whisper as an encoder.

    Usage:
        cfg = ModelConfig("openai/whisper-base")
        aud_enc = AudioEncoder(cfg)
        emb = aud_enc.encode_waveform(waveform, sr=16000)
    """

    def __init__(self, cfg: ModelConfig, target_sr: int = 16000):
        super().__init__(cfg)

        self.processor: WhisperProcessor = WhisperProcessor.from_pretrained(
            cfg.model_name
        )
        self.model: WhisperModel = WhisperModel.from_pretrained(
            cfg.model_name, torch_dtype=cfg.dtype
        ).to(cfg.device)

        self.model.eval()
        self.target_sr = target_sr

    @torch.no_grad()
    def encode_waveform(
        self,
        waveform: Union[torch.Tensor, "np.ndarray"],
        sr: int,
    ) -> torch.Tensor:
        """
        Encode a single waveform (mono) into an embedding.

        waveform: shape (T,) or (1, T), in float32/float64
        sr: original sampling rate of `waveform`
        """
        import numpy as np
        import torchaudio

        if isinstance(waveform, np.ndarray):
            waveform = torch.from_numpy(waveform)

        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)  # (1, T)

        # Resample if needed
        if sr != self.target_sr:
            waveform = torchaudio.functional.resample(
                waveform, orig_freq=sr, new_freq=self.target_sr
            )

        inputs = self.processor(
            waveform.squeeze(0).numpy(),
            sampling_rate=self.target_sr,
            return_tensors="pt"
        ).to(self.device)

        outputs = self.model(**inputs)

        # Example: mean pool encoder hidden states as embedding
        # (B, T, D) -> (B, D)
        hidden = outputs.last_hidden_state
        audio_emb = hidden.mean(dim=1)

        return audio_emb

    def encode(self, waveform, sr: int) -> torch.Tensor:
        return self.encode_waveform(waveform, sr)


# ==============================
#  Decoder-only LLM wrapper
# ==============================

class DecoderLLM(nn.Module):
    """
    Generic wrapper around a decoder-only LLM (e.g., Llama, Phi, Qwen).

    It exposes:
      - `generate(prompt, **gen_kwargs)`
    You can share this instance between different decoder task classes.
    """

    def __init__(
        self,
        cfg: ModelConfig,
        device_map: Union[str, Dict[str, int]] = "auto",
        use_fast_tokenizer: bool = True,
    ):
        super().__init__()

        self.cfg = cfg
        self.device = cfg.device
        self.dtype = cfg.dtype

        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_name,
            use_fast=use_fast_tokenizer
        )

        # Some models need a pad token
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            torch_dtype=cfg.dtype,
            device_map=device_map,
        )

        self.model.eval()

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **gen_kwargs,
    ) -> str:
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt"
        ).to(self.model.device)

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=self.tokenizer.pad_token_id,
            **gen_kwargs,
        )

        # Remove the prompt portion
        generated_ids = output_ids[0, inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True
        )
        return text


# ==============================
#  Task-style decoders
# ==============================

class BaseTaskDecoder:
    """
    A small helper base class that wraps DecoderLLM for specific tasks.

    You can subclass for text-only, audio-text, video-text prompts etc.
    """

    def __init__(self, llm: DecoderLLM):
        self.llm = llm

    def build_prompt(self, *args, **kwargs) -> str:
        """
        Override this to build the actual prompt from inputs.
        """
        raise NotImplementedError

    def __call__(self, *args, **kwargs) -> str:
        prompt = self.build_prompt(*args, **kwargs)
        return self.llm.generate(prompt)


class TextDecoder(BaseTaskDecoder):
    """
    Text-only decoding – generic text-to-text.
    """

    def build_prompt(self, text: str, instruction: Optional[str] = None) -> str:
        if instruction is None:
            instruction = "You are a helpful assistant."
        prompt = (
            f"{instruction}\n\n"
            f"Input:\n{text}\n\n"
            f"Answer:"
        )
        return prompt


class AudioDecoder(BaseTaskDecoder):
    """
    Audio -> text decoder.

    Here we assume you already encoded audio into some representation
    (e.g., Whisper transcript or embedding). For now we keep it simple:
    pass the transcript (or description) as text to the LLM.
    """

    def build_prompt(
        self,
        audio_text: str,
        instruction: Optional[str] = "You are an assistant that summarizes and explains audio transcripts.",
    ) -> str:
        prompt = (
            f"{instruction}\n\n"
            f"Transcript:\n{audio_text}\n\n"
            f"Summary and explanation:"
        )
        return prompt


class VideoDecoder(BaseTaskDecoder):
    """
    Video -> text decoder.

    For now you can feed in a description of frames, or
    a list of frame captions stitched together.
    Later you can replace this with your Perceiver-resampler
    + projector pipeline.
    """

    def build_prompt(
        self,
        video_description: str,
        instruction: Optional[str] = "You are an assistant that describes and explains videos.",
    ) -> str:
        prompt = (
            f"{instruction}\n\n"
            f"Video description:\n{video_description}\n\n"
            f"Detailed explanation:"
        )
        return prompt


