"""Projection heads for multimodal alignment and LLM integration."""

import torch
import torch.nn as nn
from typing import Optional


class ProjectionHead(nn.Module):
    """MLP projection head for embeddings.

    Simple 2-layer MLP with GELU activation and optional dropout.

    Args:
        input_dim: Input dimension
        output_dim: Output dimension
        hidden_dim: Hidden layer dimension (default: 2x output_dim)
        dropout: Dropout probability
        use_layer_norm: Whether to use layer normalization
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
        use_layer_norm: bool = True,
    ):
        super().__init__()

        if hidden_dim is None:
            hidden_dim = output_dim * 2

        layers = []

        if use_layer_norm:
            layers.append(nn.LayerNorm(input_dim))

        layers.extend([
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        ])

        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project input embeddings.

        Args:
            x: Input tensor (..., input_dim)

        Returns:
            Projected tensor (..., output_dim)
        """
        return self.net(x)


class MultimodalProjector(nn.Module):
    """Projector for multimodal embeddings to LLM space.

    Projects aligned multimodal embeddings into the LLM's hidden space,
    optionally expanding to multiple soft prompt tokens.

    Args:
        input_dim: Dimension of input embeddings
        output_dim: Dimension of LLM hidden states
        num_tokens: Number of soft prompt tokens to generate per input
        use_layer_norm: Whether to use layer normalization
        dropout: Dropout probability
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_tokens: int = 8,
        use_layer_norm: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_tokens = num_tokens

        # Project to expanded space
        expanded_dim = output_dim * num_tokens

        self.projector = ProjectionHead(
            input_dim=input_dim,
            output_dim=expanded_dim,
            hidden_dim=expanded_dim,
            dropout=dropout,
            use_layer_norm=use_layer_norm,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project embeddings to LLM token space.

        Args:
            x: Input embeddings (batch_size, input_dim) or (batch_size, seq_len, input_dim)

        Returns:
            Projected tokens (batch_size, num_tokens, output_dim) or
                            (batch_size, seq_len, num_tokens, output_dim)
        """
        # Project
        projected = self.projector(x)  # (..., num_tokens * output_dim)

        # Reshape to tokens
        if x.dim() == 2:
            # Single embedding per sample
            batch_size = x.shape[0]
            projected = projected.view(batch_size, self.num_tokens, self.output_dim)
        else:
            # Sequence of embeddings
            batch_size, seq_len = x.shape[:2]
            projected = projected.view(batch_size, seq_len, self.num_tokens, self.output_dim)

        return projected


class VisionToLLMProjector(nn.Module):
    """Projection from vision embeddings to LLM space.

    Handles both pooled embeddings and sequences with optional Perceiver compression.

    Args:
        vision_dim: Dimension of vision embeddings
        llm_dim: Dimension of LLM hidden states
        num_tokens: Number of soft prompt tokens
        use_sequence: Whether to use sequence embeddings or pooled
        dropout: Dropout probability
    """

    def __init__(
        self,
        vision_dim: int,
        llm_dim: int,
        num_tokens: int = 8,
        use_sequence: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.use_sequence = use_sequence
        self.num_tokens = num_tokens

        if use_sequence:
            # Project each sequence element to LLM space
            self.projector = ProjectionHead(
                input_dim=vision_dim,
                output_dim=llm_dim,
                dropout=dropout,
            )
        else:
            # Expand pooled embedding to multiple tokens
            self.projector = MultimodalProjector(
                input_dim=vision_dim,
                output_dim=llm_dim,
                num_tokens=num_tokens,
                dropout=dropout,
            )

    def forward(self, vision_output) -> torch.Tensor:
        """Project vision embeddings to LLM space.

        Args:
            vision_output: VisionEncoderOutput with pooled and/or sequence embeddings

        Returns:
            LLM prefix tokens (batch_size, num_tokens, llm_dim)
        """
        if self.use_sequence and vision_output.sequence is not None:
            # Use sequence embeddings
            return self.projector(vision_output.sequence)
        else:
            # Use pooled embedding
            return self.projector(vision_output.pooled)
