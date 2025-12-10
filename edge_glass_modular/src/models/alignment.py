"""Multimodal alignment model."""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from dataclasses import dataclass

from encoders import VisionEncoder, AudioEncoder, TextEncoder
from decoders import QwenDecoder, TRMDecoder
from config import ExperimentConfig
from .fusion import MultimodalFusion
from .projector import MultimodalProjector
from .losses import AlignmentLoss, TriModalAlignmentLoss


@dataclass
class AlignmentModelOutput:
    """Output from alignment model."""

    loss: Optional[torch.Tensor] = None
    losses: Optional[Dict[str, torch.Tensor]] = None
    vision_emb: Optional[torch.Tensor] = None
    audio_emb: Optional[torch.Tensor] = None
    text_emb: Optional[torch.Tensor] = None
    fused_emb: Optional[torch.Tensor] = None
    lm_loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None


class MultimodalAlignmentModel(nn.Module):
    """Multimodal alignment model with optional instruction tuning.

    This is the main model class that combines:
    - Encoders (vision, audio, text)
    - Optional fusion module
    - Optional decoder for instruction tuning
    - Alignment losses

    Args:
        config: ExperimentConfig with all model settings
    """

    def __init__(self, config: ExperimentConfig):
        super().__init__()

        self.config = config

        # Initialize encoders
        self.vision_encoder = None
        self.audio_encoder = None
        self.text_encoder = None

        if config.vision_encoder is not None:
            self.vision_encoder = VisionEncoder(
                model_name=config.vision_encoder.model_name,
                projection_dim=config.vision_encoder.projection_dim,
                freeze=config.vision_encoder.freeze,
                use_perceiver=config.vision_encoder.use_perceiver,
                perceiver_num_latents=config.vision_encoder.perceiver_num_latents,
                perceiver_latent_dim=config.vision_encoder.perceiver_latent_dim,
                perceiver_num_layers=config.vision_encoder.perceiver_num_layers,
                perceiver_num_heads=config.vision_encoder.perceiver_num_heads,
                use_mrl=config.vision_encoder.use_mrl,
                mrl_dimensions=config.vision_encoder.mrl_dimensions,
            )

        if config.audio_encoder is not None:
            self.audio_encoder = AudioEncoder(
                model_name=config.audio_encoder.model_name,
                projection_dim=config.audio_encoder.projection_dim,
                freeze=config.audio_encoder.freeze,
                use_perceiver=config.audio_encoder.use_perceiver,
                perceiver_num_latents=config.audio_encoder.perceiver_num_latents,
                perceiver_latent_dim=config.audio_encoder.perceiver_latent_dim,
                perceiver_num_layers=config.audio_encoder.perceiver_num_layers,
                perceiver_num_heads=config.audio_encoder.perceiver_num_heads,
                use_mrl=config.audio_encoder.use_mrl,
                mrl_dimensions=config.audio_encoder.mrl_dimensions,
            )

        if config.text_encoder is not None:
            self.text_encoder = TextEncoder(
                model_name=config.text_encoder.model_name,
                projection_dim=config.text_encoder.projection_dim,
                freeze=config.text_encoder.freeze,
                use_mrl=config.text_encoder.use_mrl,
                mrl_dimensions=config.text_encoder.mrl_dimensions,
            )

        # Initialize fusion if multiple modalities
        self.fusion = None
        active_encoders = sum([
            self.vision_encoder is not None,
            self.audio_encoder is not None,
            self.text_encoder is not None,
        ])

        if active_encoders > 1 and config.fusion is not None:
            modality_dims = {}
            if self.vision_encoder is not None:
                modality_dims["vision"] = config.vision_encoder.projection_dim
            if self.audio_encoder is not None:
                modality_dims["audio"] = config.audio_encoder.projection_dim
            if self.text_encoder is not None:
                modality_dims["text"] = config.text_encoder.projection_dim

            self.fusion = MultimodalFusion(
                modality_dims=modality_dims,
                fusion_dim=config.fusion.fusion_dim,
                strategy=config.fusion.strategy,
                num_layers=config.fusion.num_fusion_layers,
                num_heads=config.fusion.num_heads,
                dropout=config.fusion.dropout,
            )

        # Initialize decoder if needed
        self.decoder = None
        self.llm_projector = None

        if config.decoder is not None:
            if config.decoder.type == "qwen":
                self.decoder = QwenDecoder(
                    model_name=config.decoder.model_name,
                    load_in_8bit=config.decoder.load_in_8bit,
                    load_in_4bit=config.decoder.load_in_4bit,
                    use_lora=config.decoder.use_lora,
                    lora_r=config.decoder.lora_r,
                    lora_alpha=config.decoder.lora_alpha,
                    lora_dropout=config.decoder.lora_dropout,
                    lora_target_modules=config.decoder.lora_target_modules,
                    device_map=config.decoder.device_map,
                )
            elif config.decoder.type == "trm":
                from decoders.trm import TRMConfig
                trm_config = TRMConfig(
                    vocab_size=config.decoder.trm_vocab_size,
                    hidden_dim=config.decoder.trm_hidden_dim,
                    num_layers=config.decoder.trm_num_layers,
                    num_heads=config.decoder.trm_num_heads,
                    intermediate_dim=config.decoder.trm_intermediate_dim,
                    max_seq_len=config.decoder.trm_max_seq_len,
                    dropout=config.decoder.trm_dropout,
                    layer_norm_eps=config.decoder.trm_layer_norm_eps,
                    use_rope=config.decoder.trm_use_rope,
                    rope_theta=config.decoder.trm_rope_theta,
                )
                self.decoder = TRMDecoder(trm_config)

            # Create projector to LLM space
            input_dim = config.fusion.fusion_dim if self.fusion is not None else config.vision_encoder.projection_dim
            self.llm_projector = MultimodalProjector(
                input_dim=input_dim,
                output_dim=self.decoder.hidden_dim,
                num_tokens=8,
            )

        # Initialize loss functions
        if active_encoders == 3:
            # Tri-modal loss
            self.alignment_loss_fn = TriModalAlignmentLoss(
                vision_text_weight=1.0,
                audio_text_weight=1.0,
                vision_audio_weight=0.5,
                mrl_weight=config.optimization.mrl_loss_weight,
                temperature=0.07,
            )
        else:
            # Bi-modal loss
            self.alignment_loss_fn = AlignmentLoss(
                contrastive_weight=config.optimization.contrastive_loss_weight,
                mrl_weight=config.optimization.mrl_loss_weight,
                temperature=0.07,
            )

    def forward(
        self,
        images: Optional[torch.Tensor] = None,
        audio_features: Optional[torch.Tensor] = None,
        texts: Optional[Any] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_embeddings: bool = False,
    ) -> AlignmentModelOutput:
        """Forward pass through alignment model.

        Args:
            images: Input images (batch_size, 3, H, W)
            audio_features: Input audio features (batch_size, n_mels, time)
            texts: Text strings or embeddings
            input_ids: Token IDs for decoder (batch_size, seq_len)
            attention_mask: Attention mask for decoder
            labels: Labels for language modeling
            return_embeddings: Whether to return intermediate embeddings

        Returns:
            AlignmentModelOutput with losses and embeddings
        """
        # Encode modalities
        vision_output = None
        audio_output = None
        text_output = None

        if images is not None and self.vision_encoder is not None:
            vision_output = self.vision_encoder(images)

        if audio_features is not None and self.audio_encoder is not None:
            audio_output = self.audio_encoder(audio_features)

        if texts is not None and self.text_encoder is not None:
            text_output = self.text_encoder(texts)

        # Get pooled embeddings
        vision_emb = vision_output.pooled if vision_output is not None else None
        audio_emb = audio_output.pooled if audio_output is not None else None
        text_emb = text_output.pooled if text_output is not None else None

        # Compute alignment losses
        alignment_losses = None
        if self.training or return_embeddings:
            if hasattr(self.alignment_loss_fn, "forward"):
                # Tri-modal case
                if vision_emb is not None and audio_emb is not None and text_emb is not None:
                    alignment_losses = self.alignment_loss_fn(
                        vision_emb=vision_emb,
                        audio_emb=audio_emb,
                        text_emb=text_emb,
                        vision_mrl=vision_output.mrl_embeddings if vision_output else None,
                        audio_mrl=audio_output.mrl_embeddings if audio_output else None,
                        text_mrl=text_output.mrl_embeddings if text_output else None,
                    )
                # Bi-modal cases
                elif vision_emb is not None and text_emb is not None:
                    alignment_losses = self.alignment_loss_fn(
                        embeddings_a=vision_emb,
                        embeddings_b=text_emb,
                        embeddings_a_mrl=vision_output.mrl_embeddings if vision_output else None,
                        embeddings_b_mrl=text_output.mrl_embeddings if text_output else None,
                    )
                elif audio_emb is not None and text_emb is not None:
                    alignment_losses = self.alignment_loss_fn(
                        embeddings_a=audio_emb,
                        embeddings_b=text_emb,
                        embeddings_a_mrl=audio_output.mrl_embeddings if audio_output else None,
                        embeddings_b_mrl=text_output.mrl_embeddings if text_output else None,
                    )

        # Fuse modalities if fusion module exists
        fused_emb = None
        if self.fusion is not None:
            modality_embeddings = {}
            if vision_emb is not None:
                modality_embeddings["vision"] = vision_emb
            if audio_emb is not None:
                modality_embeddings["audio"] = audio_emb
            if text_emb is not None:
                modality_embeddings["text"] = text_emb

            if len(modality_embeddings) > 1:
                fused_emb = self.fusion(modality_embeddings)

        # Decoder forward pass if needed
        lm_loss = None
        logits = None
        if self.decoder is not None and input_ids is not None:
            # Use fused embedding or single modality
            if fused_emb is not None:
                prefix_emb = fused_emb
            elif vision_emb is not None:
                prefix_emb = vision_emb
            elif audio_emb is not None:
                prefix_emb = audio_emb
            else:
                prefix_emb = None

            # Project to LLM space
            if prefix_emb is not None:
                prefix_tokens = self.llm_projector(prefix_emb)
            else:
                prefix_tokens = None

            # Forward through decoder
            if hasattr(self.decoder, "model"):
                # Qwen decoder
                decoder_output = self.decoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    prefix_embeds=prefix_tokens,
                    labels=labels,
                )
                lm_loss = decoder_output.loss if hasattr(decoder_output, "loss") else None
                logits = decoder_output.logits if hasattr(decoder_output, "logits") else None
            else:
                # TRM decoder
                decoder_output = self.decoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    prefix_embeds=prefix_tokens,
                    labels=labels,
                )
                lm_loss = decoder_output.get("loss")
                logits = decoder_output.get("logits")

        # Combine losses
        # total_loss = None
        # if self.training:
        #     if alignment_losses is not None:
        #         total_loss = alignment_losses.get("total_loss", 0)

        #     if lm_loss is not None:
        #         lm_weight = self.config.optimization.lm_loss_weight
        #         if total_loss is not None:
        #             total_loss = total_loss + lm_weight * lm_loss
        #         else:
        #             total_loss = lm_weight * lm_loss
        total_loss = None
        if alignment_losses is not None:
            total_loss = alignment_losses.get("total_loss", 0)

        if lm_loss is not None:
            lm_weight = self.config.optimization.lm_loss_weight
            if total_loss is not None:
                total_loss = total_loss + lm_weight * lm_loss
            else:
                total_loss = lm_weight * lm_loss


        return AlignmentModelOutput(
            loss=total_loss,
            losses=alignment_losses,
            vision_emb=vision_emb if return_embeddings else None,
            audio_emb=audio_emb if return_embeddings else None,
            text_emb=text_emb if return_embeddings else None,
            fused_emb=fused_emb if return_embeddings else None,
            lm_loss=lm_loss,
            logits=logits,
        )

    @torch.no_grad()
    def generate(
        self,
        images: Optional[torch.Tensor] = None,
        audio_features: Optional[torch.Tensor] = None,
        prompt: Optional[str] = None,
        max_new_tokens: int = 256,
        **generation_kwargs,
    ):
        """Generate text from multimodal inputs.

        Args:
            images: Input images
            audio_features: Input audio
            prompt: Text prompt
            max_new_tokens: Maximum tokens to generate
            **generation_kwargs: Additional generation parameters

        Returns:
            Generated text
        """
        if self.decoder is None:
            raise ValueError("Decoder not initialized. Cannot generate.")

        # Encode modalities
        vision_emb = None
        audio_emb = None

        if images is not None and self.vision_encoder is not None:
            vision_output = self.vision_encoder(images)
            vision_emb = vision_output.pooled

        if audio_features is not None and self.audio_encoder is not None:
            audio_output = self.audio_encoder(audio_features)
            audio_emb = audio_output.pooled

        # Fuse if needed
        if self.fusion is not None:
            modality_embeddings = {}
            if vision_emb is not None:
                modality_embeddings["vision"] = vision_emb
            if audio_emb is not None:
                modality_embeddings["audio"] = audio_emb

            if len(modality_embeddings) > 1:
                prefix_emb = self.fusion(modality_embeddings)
            else:
                prefix_emb = vision_emb if vision_emb is not None else audio_emb
        else:
            prefix_emb = vision_emb if vision_emb is not None else audio_emb

        # Project to LLM space
        if prefix_emb is not None:
            prefix_tokens = self.llm_projector(prefix_emb)
        else:
            prefix_tokens = None

        # Encode prompt
        input_ids = None
        if prompt is not None and hasattr(self.decoder, "encode_text"):
            encoded = self.decoder.encode_text([prompt])
            input_ids = encoded["input_ids"].to(prefix_tokens.device if prefix_tokens is not None else "cuda")

        # Generate
        if hasattr(self.decoder, "generate"):
            output_ids = self.decoder.generate(
                input_ids=input_ids,
                prefix_embeds=prefix_tokens,
                max_new_tokens=max_new_tokens,
                **generation_kwargs,
            )

            # Decode
            if hasattr(self.decoder, "tokenizer"):
                generated_text = self.decoder.tokenizer.batch_decode(
                    output_ids, skip_special_tokens=True
                )
                return generated_text

        return output_ids

    @property
    def num_trainable_parameters(self) -> int:
        """Return number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def num_total_parameters(self) -> int:
        """Return total number of parameters."""
        return sum(p.numel() for p in self.parameters())

    def print_parameter_counts(self):
        """Print parameter counts for each component."""
        print(f"{'Component':<30} {'Trainable':>15} {'Total':>15}")
        print("-" * 60)

        if self.vision_encoder is not None:
            print(f"{'Vision Encoder':<30} {self.vision_encoder.num_parameters:>15,} "
                  f"{self.vision_encoder.num_total_parameters:>15,}")

        if self.audio_encoder is not None:
            print(f"{'Audio Encoder':<30} {self.audio_encoder.num_parameters:>15,} "
                  f"{self.audio_encoder.num_total_parameters:>15,}")

        if self.text_encoder is not None:
            print(f"{'Text Encoder':<30} {self.text_encoder.num_parameters:>15,} "
                  f"{self.text_encoder.num_total_parameters:>15,}")

        if self.fusion is not None:
            fusion_trainable = sum(p.numel() for p in self.fusion.parameters() if p.requires_grad)
            fusion_total = sum(p.numel() for p in self.fusion.parameters())
            print(f"{'Fusion':<30} {fusion_trainable:>15,} {fusion_total:>15,}")

        if self.decoder is not None:
            decoder_trainable = sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)
            decoder_total = sum(p.numel() for p in self.decoder.parameters())
            print(f"{'Decoder':<30} {decoder_trainable:>15,} {decoder_total:>15,}")

        print("-" * 60)
        print(f"{'TOTAL':<30} {self.num_trainable_parameters:>15,} {self.num_total_parameters:>15,}")

    def to(self, *args, **kwargs):
        """Safely move modules to a device/dtype.

        Quantized decoders (8/4-bit with device_map) cannot be moved with the
        standard ``nn.Module.to`` because their parameters live on meta tensors
        during dispatch. We temporarily drop the decoder from the module tree
        so encoders/fusion/projector can move while leaving the decoder on its
        configured devices.
        """
        decoder = self.decoder

        should_skip_decoder = False
        if decoder is not None:
            decoder_model = getattr(decoder, "model", None)
            if decoder_model is not None:
                should_skip_decoder = any([
                    getattr(decoder_model, "is_loaded_in_8bit", False),
                    getattr(decoder_model, "is_loaded_in_4bit", False),
                    getattr(decoder_model, "hf_device_map", None) is not None,
                    getattr(decoder_model, "device_map", None) is not None,
                ])

        if should_skip_decoder:
            self.decoder = None
            output = super(MultimodalAlignmentModel, self).to(*args, **kwargs)
            self.decoder = decoder
            return output

        return super(MultimodalAlignmentModel, self).to(*args, **kwargs)
