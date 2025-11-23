"""
audio_encoder.py

OOP wrapper for an audio encoder (Whisper-style) with extra capabilities
for Phase-0/1 POCs:

- Flexible feature strategies:
    * "auto"          → default to last_hidden_state + pooling
    * "last_hidden"   → use outputs.last_hidden_state
    * "layers_concat" → concatenate selected hidden layers, then pool

- Configurable pooling:
    * "mean"  → mean over time dimension
    * "cls"   → first token/frame (rough CLS-like)
    * "none"  → return token-level features (no pooling)

Usage (simple, same as before):

    cfg = ModelConfig("openai/whisper-base")
    enc = AudioEncoder(cfg)
    emb = enc.encode_waveform(waveform, sr=16000)           # (1, D)

Advanced usage:

    # concat 2nd and 2nd-to-last hidden layer, then mean-pool over time:
    enc = AudioEncoder(
        cfg,
        feature_strategy="layers_concat",
        layer_indices=[2, -2],
        pool="mean",
    )

    emb = enc.encode_waveform(waveform, sr=16000)           # (1, D_concat)

    # get token-level features (no pooling):
    tokens = enc.encode_waveform(waveform, sr=16000, pooled=False)  # (1, T, D or D_concat)
"""

from typing import Optional, Union, List, Literal

import numpy as np
import torch
import torchaudio
from transformers import WhisperProcessor, WhisperModel

from .base import ModelConfig, BaseEncoder


AudioFeatureStrategy = Literal[
    "auto",
    "last_hidden",
    "layers_concat",
]

PoolStrategy = Literal[
    "mean",
    "cls",
    "none",
]


class AudioEncoder(BaseEncoder):
    """
    Generic audio encoder using Whisper as an encoder.

    The encoder is frozen (Phase 1: frozen audio encoder) and returns
    embeddings suitable for downstream alignment / Perceiver / projector.

    Attributes
    ----------
    feature_strategy : AudioFeatureStrategy
        How to derive features from model outputs.
    layer_indices : List[int]
        Layer indices (into hidden_states) used when feature_strategy=="layers_concat".
    pool : PoolStrategy
        Pooling strategy over the time dimension.
    """

    def __init__(
        self,
        cfg: ModelConfig,
        target_sr: int = 16000,
        feature_strategy: AudioFeatureStrategy = "auto",
        layer_indices: Optional[List[int]] = None,
        pool: PoolStrategy = "mean",
    ):
        """
        Parameters
        ----------
        cfg : ModelConfig
            Per-model configuration (name, device, dtype).
        target_sr : int, default 16000
            Target sampling rate to resample to.
        feature_strategy : {"auto", "last_hidden", "layers_concat"}
            Feature extraction strategy (see docstring above).
        layer_indices : list of int, optional
            Indices of hidden layers to use when concatenating hidden states.
            For example: [2, -2] (2nd layer and 2nd-to-last layer).
        pool : {"mean", "cls", "none"}
            Pooling strategy across time frames.
        """
        super().__init__(cfg)

        self.processor: WhisperProcessor = WhisperProcessor.from_pretrained(
            cfg.model_name
        )
        self.model: WhisperModel = WhisperModel.from_pretrained(
            cfg.model_name, torch_dtype=cfg.dtype
        ).to(self.device)

        # Enable hidden states for analysis / future MRL
        if hasattr(self.model.config, "output_hidden_states"):
            self.model.config.output_hidden_states = True

        # Freeze (Phase 1: frozen encoder)
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

        self.target_sr = target_sr

        self.feature_strategy: AudioFeatureStrategy = feature_strategy
        self.layer_indices: List[int] = layer_indices or []
        self.pool: PoolStrategy = pool

    # ---------------------------
    # Internal helpers
    # ---------------------------

    def _pool_sequence(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pool over the time dimension according to self.pool.

        x: (B, T, D)
        returns: (B, D) if pool != "none", else (B, T, D)
        """
        if self.pool == "mean":
            return x.mean(dim=1)
        elif self.pool == "cls":
            # Take first time step as CLS-like representation
            return x[:, 0, :]
        elif self.pool == "none":
            return x
        else:
            raise ValueError(f"Unknown pool strategy: {self.pool}")

    def _extract_features(self, outputs, pooled: bool) -> torch.Tensor:
        """
        Given Whisper model outputs, apply the configured feature strategy and pooling.
        """
        # 1) AUTO: default to last_hidden + pooling
        if self.feature_strategy == "auto":
            hidden = outputs.last_hidden_state  # (B, T, D)
            if pooled:
                return self._pool_sequence(hidden)
            else:
                return hidden

        # 2) last_hidden explicitly
        if self.feature_strategy == "last_hidden":
            hidden = outputs.last_hidden_state  # (B, T, D)
            if pooled:
                return self._pool_sequence(hidden)
            else:
                return hidden

        # 3) layers_concat: concat chosen hidden layers along channel dim
        if self.feature_strategy == "layers_concat":
            if not hasattr(outputs, "hidden_states") or outputs.hidden_states is None:
                raise ValueError(
                    "feature_strategy='layers_concat' requires outputs.hidden_states=True."
                )
            if not self.layer_indices:
                raise ValueError(
                    "feature_strategy='layers_concat' but no layer_indices provided."
                )

            hidden_states = outputs.hidden_states  # tuple of (layer, B, T, D)
            layers_to_cat = []
            for idx in self.layer_indices:
                layers_to_cat.append(hidden_states[idx])  # (B, T, D_i)

            concat_hidden = torch.cat(layers_to_cat, dim=-1)  # (B, T, sum(D_i))

            if pooled:
                return self._pool_sequence(concat_hidden)
            else:
                return concat_hidden

        raise ValueError(f"Unknown feature strategy: {self.feature_strategy}")

    # ---------------------------
    # Public API
    # ---------------------------

    @torch.no_grad()
    def encode_waveform(
        self,
        waveform: Union[torch.Tensor, np.ndarray],
        sr: int,
        pooled: bool = True,
    ) -> torch.Tensor:
        """
        Encode a single waveform into an embedding or sequence of embeddings.

        Parameters
        ----------
        waveform : np.ndarray or torch.Tensor
            Shape (T,) or (1, T) – mono waveform.
        sr : int
            Original sampling rate of waveform.
        pooled : bool, default True
            If True, returns (1, D or D_concat) embedding.
            If False, returns (1, T, D or D_concat) token-level features.

        Returns
        -------
        torch.Tensor
            Audio features according to feature_strategy & pooling.
        """
        if isinstance(waveform, np.ndarray):
            waveform = torch.from_numpy(waveform)

        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)  # (1, T)

        if sr != self.target_sr:
            waveform = torchaudio.functional.resample(
                waveform, orig_freq=sr, new_freq=self.target_sr
            )

        # WhisperProcessor expects a 1D numpy array for a single example
        inputs = self.processor(
            waveform.squeeze(0).numpy(),
            sampling_rate=self.target_sr,
            return_tensors="pt",
        ).to(self.device)

        outputs = self.model(**inputs)

        feats = self._extract_features(outputs, pooled=pooled)
        return feats

    def encode(
        self,
        waveform: Union[torch.Tensor, np.ndarray],
        sr: int,
        pooled: bool = True,
    ) -> torch.Tensor:
        """
        Thin wrapper over encode_waveform to satisfy BaseEncoder interface.

        This keeps the simple usage:

            emb = enc.encode(wav, sr=16000)

        and also allows:

            tokens = enc.encode(wav, sr=16000, pooled=False)
        """
        return self.encode_waveform(waveform, sr=sr, pooled=pooled)


def load_audio_encoder(
    model_name: str,
    device: Optional[str] = None,
    dtype: torch.dtype = torch.float16,
    target_sr: int = 16000,
    feature_strategy: AudioFeatureStrategy = "auto",
    layer_indices: Optional[List[int]] = None,
    pool: PoolStrategy = "mean",
) -> AudioEncoder:
    """
    Convenience loader for scripts / notebooks.

    Typical usage from your main IPYNB:

        from models.audio_encoder import load_audio_encoder

        audio_enc = load_audio_encoder(
            cfg.models.audio_model_name,
            device=str(cfg.torch_device),
            dtype=cfg.torch_dtype,
            target_sr=16000,
            feature_strategy="layers_concat",
            layer_indices=[2, -2],
            pool="mean",
        )
    """
    cfg = ModelConfig(model_name=model_name, device=device, dtype=dtype)
    return AudioEncoder(
        cfg,
        target_sr=target_sr,
        feature_strategy=feature_strategy,
        layer_indices=layer_indices,
        pool=pool,
    )
