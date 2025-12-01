"""Tiny Recursive Model (TRM) decoder implementation.

Based on the TRM architecture that uses recursive reasoning with tiny networks
instead of deep transformers.
"""

import torch
import torch.nn as nn
import math
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class TRMConfig:
    """Configuration for TRM decoder."""

    vocab_size: int = 32000
    hidden_dim: int = 512
    num_layers: int = 6
    num_heads: int = 8
    intermediate_dim: int = 2048
    max_seq_len: int = 2048
    dropout: float = 0.1
    layer_norm_eps: float = 1e-6
    use_rope: bool = True
    rope_theta: float = 10000.0


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # RMS normalization
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        x_normed = x / rms
        return self.weight * x_normed


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)."""

    def __init__(self, dim: int, max_seq_len: int = 2048, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta

        # Precompute frequencies
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Precompute sin/cos for max sequence length
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)  # (max_seq_len, dim//2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (max_seq_len, dim)
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())

    def forward(self, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return cos and sin embeddings for sequence length."""
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def apply_rotary_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings to queries and keys.

    Args:
        q: Queries (batch_size, num_heads, seq_len, head_dim)
        k: Keys (batch_size, num_heads, seq_len, head_dim)
        cos: Cosine embeddings (seq_len, head_dim)
        sin: Sine embeddings (seq_len, head_dim)

    Returns:
        Rotated queries and keys
    """
    # Reshape for rotation
    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    # Apply rotation
    q_embed = (q * cos.unsqueeze(0).unsqueeze(0)) + (rotate_half(q) * sin.unsqueeze(0).unsqueeze(0))
    k_embed = (k * cos.unsqueeze(0).unsqueeze(0)) + (rotate_half(k) * sin.unsqueeze(0).unsqueeze(0))

    return q_embed, k_embed


class TRMAttention(nn.Module):
    """Multi-head self-attention with optional RoPE."""

    def __init__(self, config: TRMConfig):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_dim // config.num_heads
        self.use_rope = config.use_rope

        assert self.hidden_dim % self.num_heads == 0

        self.q_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.o_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)

        self.dropout = nn.Dropout(config.dropout)

        if self.use_rope:
            self.rotary_emb = RotaryEmbedding(
                self.head_dim, max_seq_len=config.max_seq_len, theta=config.rope_theta
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        # Project to Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE if enabled
        if self.use_rope:
            cos, sin = self.rotary_emb(seq_len)
            q, k = apply_rotary_emb(q, k, cos, sin)

        # Compute attention scores
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply causal mask
        if attention_mask is None:
            # Create causal mask
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=hidden_states.device), diagonal=1
            ).bool()
            attn_weights = attn_weights.masked_fill(causal_mask, float("-inf"))
        else:
            attn_weights = attn_weights + attention_mask

        # Softmax
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)

        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_dim)

        # Output projection
        output = self.o_proj(attn_output)

        return output


class TRMMLP(nn.Module):
    """MLP feed-forward network."""

    def __init__(self, config: TRMConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_dim, config.intermediate_dim)
        self.fc2 = nn.Linear(config.intermediate_dim, config.hidden_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TRMLayer(nn.Module):
    """Single TRM transformer layer."""

    def __init__(self, config: TRMConfig):
        super().__init__()
        self.attention = TRMAttention(config)
        self.mlp = TRMMLP(config)
        self.norm1 = RMSNorm(config.hidden_dim, eps=config.layer_norm_eps)
        self.norm2 = RMSNorm(config.hidden_dim, eps=config.layer_norm_eps)

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Self-attention with residual
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.attention(hidden_states, attention_mask)
        hidden_states = residual + hidden_states

        # MLP with residual
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class TRMDecoder(nn.Module):
    """Tiny Recursive Model decoder for efficient language modeling.

    This is a lightweight transformer decoder with:
    - RMSNorm for normalization
    - RoPE for positional encoding
    - Fewer layers than standard models (typically 6-8)

    Args:
        config: TRM configuration
    """

    def __init__(self, config: TRMConfig):
        super().__init__()
        self.config = config

        # Token embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_dim)

        # Transformer layers
        self.layers = nn.ModuleList([TRMLayer(config) for _ in range(config.num_layers)])

        # Final norm
        self.norm = RMSNorm(config.hidden_dim, eps=config.layer_norm_eps)

        # LM head
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

        # Tie weights with embedding
        self.lm_head.weight = self.embed_tokens.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        prefix_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ):
        """Forward pass.

        Args:
            input_ids: Token IDs (batch_size, seq_len)
            inputs_embeds: Alternative to input_ids (batch_size, seq_len, hidden_dim)
            attention_mask: Attention mask
            prefix_embeds: Multimodal prefix (batch_size, num_prefix, hidden_dim)
            labels: Labels for language modeling loss

        Returns:
            Dictionary with logits and optional loss
        """
        # Get embeddings
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # Prepend prefix if provided
        if prefix_embeds is not None:
            batch_size, num_prefix, _ = prefix_embeds.shape
            inputs_embeds = torch.cat([prefix_embeds, inputs_embeds], dim=1)

            # Adjust labels if provided
            if labels is not None:
                prefix_labels = torch.full(
                    (batch_size, num_prefix),
                    fill_value=-100,
                    dtype=labels.dtype,
                    device=labels.device,
                )
                labels = torch.cat([prefix_labels, labels], dim=1)

        hidden_states = inputs_embeds

        # Pass through transformer layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)

        # Final norm
        hidden_states = self.norm(hidden_states)

        # LM head
        logits = self.lm_head(hidden_states)

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Shift logits and labels for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Flatten tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        return {"logits": logits, "loss": loss}

    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        prefix_embeds: Optional[torch.Tensor] = None,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        do_sample: bool = True,
    ):
        """Generate text autoregressively.

        Args:
            input_ids: Initial token IDs (batch_size, init_len)
            prefix_embeds: Multimodal prefix
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling
            do_sample: Use sampling or greedy decoding

        Returns:
            Generated token IDs
        """
        if input_ids is None and prefix_embeds is None:
            raise ValueError("Either input_ids or prefix_embeds must be provided")

        # Initialize with input_ids or dummy token
        if input_ids is None:
            batch_size = prefix_embeds.shape[0]
            input_ids = torch.zeros(batch_size, 1, dtype=torch.long, device=prefix_embeds.device)

        batch_size, current_len = input_ids.shape

        for _ in range(max_new_tokens):
            # Forward pass
            outputs = self.forward(input_ids=input_ids, prefix_embeds=prefix_embeds)
            logits = outputs["logits"]

            # Get logits for last position
            next_token_logits = logits[:, -1, :]  # (batch_size, vocab_size)

            # Apply temperature
            next_token_logits = next_token_logits / temperature

            # Sampling or greedy
            if do_sample:
                # Top-k filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[
                        0
                    ][..., -1, None]
                    next_token_logits[indices_to_remove] = float("-inf")

                # Top-p filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(
                        torch.softmax(sorted_logits, dim=-1), dim=-1
                    )
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    next_token_logits[indices_to_remove] = float("-inf")

                # Sample
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)

            # Only pass prefix on first iteration
            prefix_embeds = None

        return input_ids

    @property
    def num_parameters(self) -> int:
        """Return number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def hidden_dim(self) -> int:
        """Return hidden dimension for projector compatibility."""
        return self.config.hidden_dim
