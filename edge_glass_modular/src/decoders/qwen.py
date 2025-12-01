"""Qwen decoder with LoRA support."""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
from typing import Optional, List


class QwenDecoder(nn.Module):
    """Qwen LLM decoder with optional LoRA fine-tuning.

    Supports:
    - 8-bit/4-bit quantization for memory efficiency
    - LoRA parameter-efficient fine-tuning
    - Multimodal prefix tokens from aligned encoders

    Args:
        model_name: HuggingFace model name (e.g., 'Qwen/Qwen2.5-7B-Instruct')
        load_in_8bit: Use 8-bit quantization
        load_in_4bit: Use 4-bit quantization
        use_lora: Enable LoRA fine-tuning
        lora_r: LoRA rank
        lora_alpha: LoRA alpha (scaling factor)
        lora_dropout: LoRA dropout
        lora_target_modules: Which modules to apply LoRA to
        device_map: Device mapping strategy ('auto', 'balanced', etc.)
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        load_in_8bit: bool = True,
        load_in_4bit: bool = False,
        use_lora: bool = True,
        lora_r: int = 32,
        lora_alpha: int = 64,
        lora_dropout: float = 0.1,
        lora_target_modules: List[str] = None,
        device_map: str = "auto",
    ):
        super().__init__()

        self.model_name = model_name
        self.use_lora = use_lora

        # Quantization config
        quantization_config = None
        if load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif load_in_8bit:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map=device_map,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, padding_side="left"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Get model config
        self.config = self.model.config
        self.hidden_dim = self.config.hidden_size
        self.vocab_size = self.config.vocab_size

        # Apply LoRA if requested
        if use_lora:
            if lora_target_modules is None:
                lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=lora_target_modules,
                bias="none",
            )

            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        prefix_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ):
        """Forward pass through decoder.

        Args:
            input_ids: Token IDs (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
            prefix_embeds: Optional multimodal prefix embeddings (batch_size, num_prefix, hidden_dim)
            labels: Optional labels for language modeling loss

        Returns:
            ModelOutput with logits and optional loss
        """
        # Get input embeddings
        inputs_embeds = self.model.get_input_embeddings()(input_ids)

        # Prepend prefix if provided
        if prefix_embeds is not None:
            batch_size, num_prefix, _ = prefix_embeds.shape

            # Concatenate prefix + text embeddings
            inputs_embeds = torch.cat([prefix_embeds, inputs_embeds], dim=1)

            # Extend attention mask for prefix
            if attention_mask is not None:
                prefix_mask = torch.ones(
                    batch_size, num_prefix, dtype=attention_mask.dtype, device=attention_mask.device
                )
                attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)

            # Adjust labels if provided (ignore prefix in loss)
            if labels is not None:
                prefix_labels = torch.full(
                    (batch_size, num_prefix),
                    fill_value=-100,  # Ignore index
                    dtype=labels.dtype,
                    device=labels.device,
                )
                labels = torch.cat([prefix_labels, labels], dim=1)

        # Forward through model
        outputs = self.model(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels
        )

        return outputs

    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        prefix_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        **kwargs,
    ):
        """Generate text with optional multimodal prefix.

        Args:
            input_ids: Token IDs for text prompt (optional if only using prefix)
            prefix_embeds: Multimodal prefix embeddings (batch_size, num_prefix, hidden_dim)
            attention_mask: Attention mask
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            do_sample: Whether to use sampling
            **kwargs: Additional generation arguments

        Returns:
            Generated token IDs
        """
        # Prepare inputs
        if input_ids is None and prefix_embeds is not None:
            # Only prefix, create dummy input_ids
            batch_size = prefix_embeds.shape[0]
            input_ids = torch.tensor(
                [[self.tokenizer.pad_token_id]], device=prefix_embeds.device
            ).expand(batch_size, 1)

        # Get input embeddings
        inputs_embeds = self.model.get_input_embeddings()(input_ids)

        # Prepend prefix if provided
        if prefix_embeds is not None:
            batch_size, num_prefix, _ = prefix_embeds.shape
            inputs_embeds = torch.cat([prefix_embeds, inputs_embeds], dim=1)

            # Extend attention mask
            if attention_mask is not None:
                prefix_mask = torch.ones(
                    batch_size, num_prefix, dtype=attention_mask.dtype, device=attention_mask.device
                )
                attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
            else:
                attention_mask = torch.ones(
                    batch_size,
                    num_prefix + input_ids.shape[1],
                    dtype=torch.long,
                    device=prefix_embeds.device,
                )

        # Generate
        outputs = self.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **kwargs,
        )

        return outputs

    def encode_text(self, texts: List[str], max_length: int = 512):
        """Tokenize text inputs.

        Args:
            texts: List of text strings
            max_length: Maximum sequence length

        Returns:
            Dictionary with input_ids and attention_mask
        """
        return self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length
        )

    @property
    def num_parameters(self) -> int:
        """Return number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def num_total_parameters(self) -> int:
        """Return total number of parameters."""
        return sum(p.numel() for p in self.parameters())
