"""
text_decoder.py

Task-specific decoders on top of a generic decoder-only LLM.
"""

from typing import Optional, Dict

import torch

from .base import ModelConfig, DecoderLLM, BaseTaskDecoder


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
    Audio -> text decoder (takes an audio transcript string).
    """

    def build_prompt(
        self,
        audio_text: str,
        instruction: Optional[str] = (
            "You are an assistant that summarizes and explains audio transcripts."
        ),
    ) -> str:
        prompt = (
            f"{instruction}\n\n"
            f"Transcript:\n{audio_text}\n\n"
            f"Summary and explanation:"
        )
        return prompt


class VideoDecoder(BaseTaskDecoder):
    """
    Video -> text decoder (takes a textual description of the video).
    """

    def build_prompt(
        self,
        video_description: str,
        instruction: Optional[str] = (
            "You are an assistant that describes and explains videos."
        ),
    ) -> str:
        prompt = (
            f"{instruction}\n\n"
            f"Video description:\n{video_description}\n\n"
            f"Detailed explanation:"
        )
        return prompt


def load_decoder_llm(
    model_name: str,
    device: Optional[str] = None,
    dtype: torch.dtype = torch.float16,
    device_map: str = "auto",
    add_image_special_tokens: bool = False,
    image_token_strings: Optional[Dict[str, str]] = None,
) -> DecoderLLM:
    """
    Convenience loader for a decoder-only LLM.

    Example:

        llm = load_decoder_llm(
            cfg.models.llm_model_name,
            device=str(cfg.torch_device),
            dtype=cfg.torch_dtype,
            device_map="auto",
            add_image_special_tokens=True,
        )
    """
    cfg = ModelConfig(model_name=model_name, device=device, dtype=dtype)
    return DecoderLLM(
        cfg,
        device_map=device_map,
        add_image_special_tokens=add_image_special_tokens,
        image_token_strings=image_token_strings,
    )
