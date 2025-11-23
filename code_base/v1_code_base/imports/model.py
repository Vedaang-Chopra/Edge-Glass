import torch
import torch.nn as nn
import torch.nn.functional as F

from imports.encoders import VisionEncoder, AudioEncoder
from imports.perceiver import PerceiverLatentEncoder, ProjectorMLP


class MultiModalAlignmentModel(nn.Module):
    """
    Vision + Audio alignment model with optional shared Perceiver.

    - VisionEncoder → (Linear adapter) → shared Perceiver? → ProjectorMLP → z_align
    - AudioEncoder  → (Linear adapter) → shared Perceiver? → ProjectorMLP → z_align

    If use_perceiver = True:
        encoder_feats -> adapter -> Perceiver -> ProjectorMLP (token-level) -> pooled
    If use_perceiver = False:
        encoder_feats -> adapter -> masked mean over time -> ProjectorMLP -> pooled
    """

    def __init__(
        self,
        d_shared: int = 512,       # Perceiver input dim (after adapters)
        d_latent: int = 512,       # Perceiver latent dim
        d_align: int = 1024,       # final alignment dim
        num_latents: int = 64,
        num_layers: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        use_perceiver: bool = True,
        vision_model_name: str = "facebook/dinov2-base",
        audio_model_name: str = "openai/whisper-base",
        dtype: torch.dtype = torch.float16,
        device: torch.device | None = None,
    ):
        super().__init__()

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.dtype = dtype

        self.d_shared = d_shared
        self.d_latent = d_latent
        self.d_align = d_align
        self.use_perceiver = use_perceiver

        # ---- Encoders (frozen) ----
        self.vision_encoder = VisionEncoder(
            model_name=vision_model_name,
            device=device,
            dtype=dtype,
        )
        self.audio_encoder = AudioEncoder(
            model_name=audio_model_name,
            device=device,
            dtype=dtype,
        )

        # ---- Modality-specific adapters to shared Perceiver input dim ----
        self.vision_adapter: nn.Linear | None = None   # created lazily
        self.audio_adapter: nn.Linear | None = None    # created lazily

        # ---- Shared Perceiver (optional) ----
        if use_perceiver:
            self.perceiver = PerceiverLatentEncoder(
                num_latents=num_latents,
                d_latent=d_latent,
                d_input=d_shared,   # after adapters
                num_layers=num_layers,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            ).to(device)
            projector_in_dim = d_latent
        else:
            self.perceiver = None
            projector_in_dim = d_shared

        # ---- Shared alignment projector ----
        self.projector = ProjectorMLP(
            d_in=projector_in_dim,
            d_out=d_align,
            hidden_factor=2.0,
            dropout=dropout,
        ).to(device)

    # ---------------------------
    # Internal helpers
    # ---------------------------

    def _ensure_vision_adapter(self, feat_dim: int):
        if self.vision_adapter is None:
            self.vision_adapter = nn.Linear(feat_dim, self.d_shared).to(self.device, dtype=self.dtype)

    def _ensure_audio_adapter(self, feat_dim: int):
        if self.audio_adapter is None:
            self.audio_adapter = nn.Linear(feat_dim, self.d_shared).to(self.device, dtype=self.dtype)

    def _pool_masked_mean(self, feats: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        feats: (B, T, D)
        mask: (B, T) bool, True = valid
        returns: (B, D)
        """
        mask_f = mask.float()
        sum_feats = (feats * mask_f.unsqueeze(-1)).sum(dim=1)          # (B, D)
        denom = mask_f.sum(dim=1, keepdim=True).clamp(min=1e-6)        # (B, 1)
        return sum_feats / denom

    # ---------------------------
    # Public encode APIs
    # ---------------------------

    @torch.no_grad()
    def encode_vision(self, images):
        """
        images: list[PIL.Image] or preprocessed tensor
        Returns:
            {
              "tokens":  (B, L, d_align) or None if use_perceiver=False,
              "pooled":  (B, d_align),
              "mask":    (B, T_patches) bool,
            }
        """
        enc_out = self.vision_encoder.encode_images(images)
        feats, mask = enc_out["feats"].to(self.device), enc_out["mask"]  # (B, T_v, D_v)

        self._ensure_vision_adapter(feats.size(-1))
        feats_sh = self.vision_adapter(feats)  # (B, T_v, d_shared)

        if self.use_perceiver:
            latents = self.perceiver(feats_sh, encoder_mask=mask)   # (B, L, d_latent)
            tokens = self.projector(latents)                        # (B, L, d_align)
            pooled = tokens.mean(dim=1)                             # (B, d_align)
        else:
            # No Perceiver: pool over time, then projector
            pooled_in = self._pool_masked_mean(feats_sh, mask)      # (B, d_shared)
            pooled = self.projector(pooled_in)                      # (B, d_align)
            tokens = None

        return {
            "tokens": tokens,
            "pooled": pooled,
            "mask": mask,
        }

    @torch.no_grad()
    def encode_audio(self, waveforms, sample_rates):
        """
        waveforms: Tensor (B, T) / (B, 1, T) or list[Tensor]
        sample_rates: int or list[int]
        Returns:
            {
              "tokens":  (B, L, d_align) or None if use_perceiver=False,
              "pooled":  (B, d_align),
              "mask":    (B, T_audio) bool,
            }
        """
        enc_out = self.audio_encoder.encode_waveforms(waveforms, sample_rates)
        feats, mask = enc_out["feats"].to(self.device), enc_out["mask"]  # (B, T_a, D_a)

        self._ensure_audio_adapter(feats.size(-1))
        feats_sh = self.audio_adapter(feats)  # (B, T_a, d_shared)

        if self.use_perceiver:
            latents = self.perceiver(feats_sh, encoder_mask=mask)   # (B, L, d_latent)
            tokens = self.projector(latents)                        # (B, L, d_align)
            pooled = tokens.mean(dim=1)                             # (B, d_align)
        else:
            pooled_in = self._pool_masked_mean(feats_sh, mask)      # (B, d_shared)
            pooled = self.projector(pooled_in)                      # (B, d_align)
            tokens = None

        return {
            "tokens": tokens,
            "pooled": pooled,
            "mask": mask,
        }

    # ---------------------------
    # Optional: joint forward for training
    # ---------------------------

    def forward(self, images, waveforms, sample_rates):
        """
        Convenience forward that encodes both modalities and returns aligned pooled embeddings.
        """
        v = self.encode_vision(images)
        a = self.encode_audio(waveforms, sample_rates)
        return {
            "vision": v,   # v["pooled"] → (B, d_align)
            "audio": a,    # a["pooled"] → (B, d_align)
        }
