"""Text encoder module using Sentence Transformers."""

import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass
from typing import Optional, List, Union

from .mrl import MatryoshkaProjection


@dataclass
class TextEncoderOutput:
    """Output from text encoder."""

    pooled: torch.Tensor  # (batch_size, projection_dim)
    sequence: Optional[torch.Tensor] = None  # (batch_size, seq_len, projection_dim)
    mrl_embeddings: Optional[dict] = None  # {dim: embedding} for MRL


class TextEncoder(nn.Module):
    """Text encoder with optional MRL.

    Architecture:
        Text strings
          ↓
        Sentence-BERT (frozen)
          ↓ (B, hidden_dim)
        Linear Projection
          ↓ (B, projection_dim)
        [Optional] MRL Projection
          ↓ (B, projection_dim) + MRL outputs

    Args:
        model_name: HuggingFace/SentenceTransformers model name
        projection_dim: Output dimension for projected embeddings
        freeze: Whether to freeze the base encoder
        use_mrl: Whether to use Matryoshka Representation Learning
        mrl_dimensions: List of MRL dimensions
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        projection_dim: int = 4096,  # Updated to 4096 to match vision encoder
        freeze: bool = True,
        use_mrl: bool = False,
        mrl_dimensions: list = None,
    ):
        super().__init__()

        self.model_name = model_name
        self.projection_dim = projection_dim
        self.use_mrl = use_mrl

        # Load Sentence-BERT encoder
        self.encoder = SentenceTransformer(model_name)

        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Get hidden dimension
        self.hidden_dim = self.encoder.get_sentence_embedding_dimension()

        # Projection layer
        self.projector = nn.Linear(self.hidden_dim, projection_dim)

        # MRL projection with updated dimensions for 4096
        if use_mrl:
            if mrl_dimensions is None:
                # Same MRL dimensions as vision: 4096 -> 2048 -> 1024 -> 512 -> 256 -> 128
                mrl_dimensions = [2048, 1024, 512, 256, 128]
            self.mrl = MatryoshkaProjection(
                input_dim=projection_dim, mrl_dimensions=mrl_dimensions
            )
        else:
            self.mrl = None

    def forward(
        self, texts: Union[List[str], torch.Tensor], return_sequence: bool = False
    ) -> TextEncoderOutput:
        """Encode texts to embeddings.

        Args:
            texts: Either list of text strings OR pre-tokenized input_ids tensor
            return_sequence: Whether to return sequence embeddings (not supported for SentenceTransformers)

        Returns:
            TextEncoderOutput with pooled embeddings
        """
        if isinstance(texts, list):
            # Encode with Sentence-BERT
            with torch.no_grad() if not self.training else torch.enable_grad():
                embeddings = self.encoder.encode(
                    texts,
                    convert_to_tensor=True,
                    show_progress_bar=False,
                    batch_size=len(texts),
                )
        else:
            # Assume already encoded
            embeddings = texts

        # Project
        projected = self.projector(embeddings)  # (B, projection_dim)

        # L2 normalize BEFORE MRL
        pooled = nn.functional.normalize(projected, p=2, dim=-1)

        # Apply MRL if enabled (MRL expects normalized input)
        mrl_embeddings = None
        if self.use_mrl:
            mrl_embeddings = self.mrl(pooled)

        # Note: Sequence embeddings not supported for sentence transformers
        # They only return sentence-level embeddings
        sequence_output = None

        return TextEncoderOutput(
            pooled=pooled, sequence=sequence_output, mrl_embeddings=mrl_embeddings
        )

    def preprocess(self, texts: List[str]):
        """Preprocess is handled internally by SentenceTransformer.

        Args:
            texts: List of text strings

        Returns:
            Same list (no preprocessing needed)
        """
        return texts

    @property
    def num_parameters(self) -> int:
        """Return number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def num_total_parameters(self) -> int:
        """Return total number of parameters."""
        return sum(p.numel() for p in self.parameters())
