"""
TRM Decoder LLM & VLM Implementation

This module implements the Tiny Recursive Model (TRM) concept from the paper
"Less is More: Recursive Reasoning with Tiny Networks" adapted for:
1. Decoder-only Language Models
2. Vision-Language Models

Key TRM Concepts:
- Recursive reasoning with tiny networks (2 layers)
- Deep supervision at each recursion step
- Single network for both z and y updates
- Progressive answer refinement

Author: Adapted from TRM paper (Jolicoeur-Martineau, 2025)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict


# ============================================================================
# Configuration Classes
# ============================================================================

@dataclass
class TRMDecoderConfig:
    """Configuration for TRM Decoder LLM"""
    vocab_size: int = 32000
    hidden_size: int = 256
    num_heads: int = 4
    num_layers: int = 2  # TRM uses tiny networks (2 layers as per paper)
    max_seq_len: int = 512
    dropout: float = 0.1
    
    # TRM specific parameters
    n_recursions: int = 6  # Number of latent reasoning steps (n in paper)
    t_cycles: int = 3  # Number of deep supervision cycles (T in paper)
    
    # Architecture
    expansion: float = 2.67  # SwiGLU expansion factor
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    
    def __post_init__(self):
        self.head_dim = self.hidden_size // self.num_heads
        self.intermediate_size = int(self.hidden_size * self.expansion)


@dataclass
class VisionEncoderConfig:
    """Configuration for Vision Encoder"""
    img_size: int = 224
    patch_size: int = 16
    in_channels: int = 3
    embed_dim: int = 256
    num_heads: int = 4
    num_layers: int = 4
    dropout: float = 0.1
    n_recursions: int = 4
    
    def __post_init__(self):
        self.n_patches = (self.img_size // self.patch_size) ** 2
        self.head_dim = self.embed_dim // self.num_heads


@dataclass
class TRMVLMConfig:
    """Configuration for TRM Vision-Language Model"""
    # Vision config
    img_size: int = 224
    patch_size: int = 16
    vision_embed_dim: int = 256
    vision_num_heads: int = 4
    vision_num_layers: int = 2
    vision_n_recursions: int = 4
    
    # Language config
    vocab_size: int = 32000
    lang_hidden_size: int = 256
    lang_num_heads: int = 4
    lang_num_layers: int = 2
    max_seq_len: int = 256
    lang_n_recursions: int = 4
    lang_t_cycles: int = 3
    
    # Alignment config
    projection_dim: int = 256
    
    # Training config
    dropout: float = 0.1
    temperature: float = 0.07


# ============================================================================
# Core Building Blocks
# ============================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)"""
    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self._build_cache(max_seq_len)
    
    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer('cos_cached', emb.cos())
        self.register_buffer('sin_cached', emb.sin())
    
    def forward(self, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, 
    k: torch.Tensor, 
    cos: torch.Tensor, 
    sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings to queries and keys."""
    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class SwiGLU(nn.Module):
    """SwiGLU activation function with linear projections"""
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with RoPE"""
    def __init__(self, config: TRMDecoderConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(
        self, 
        hidden_states: torch.Tensor, 
        cos: torch.Tensor, 
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, L, _ = hidden_states.shape
        
        q = self.q_proj(hidden_states).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
        
        # Causal mask
        if attention_mask is None:
            causal_mask = torch.triu(
                torch.ones(L, L, device=hidden_states.device), diagonal=1
            ).bool()
            attn_weights = attn_weights.masked_fill(
                causal_mask.unsqueeze(0).unsqueeze(0), float('-inf')
            )
        else:
            attn_weights = attn_weights + attention_mask
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, self.hidden_size)
        
        return self.o_proj(attn_output)


# ============================================================================
# TRM Blocks and Modules
# ============================================================================

class TRMBlock(nn.Module):
    """
    A single TRM transformer block with Post-Norm architecture.
    """
    def __init__(self, config: TRMDecoderConfig):
        super().__init__()
        self.config = config
        self.self_attn = CausalSelfAttention(config)
        self.mlp = SwiGLU(config.hidden_size, config.intermediate_size)
        self.attn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
    
    def forward(
        self, 
        hidden_states: torch.Tensor,
        cos: torch.Tensor, 
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Self-attention with post-norm
        residual = hidden_states
        hidden_states = self.self_attn(hidden_states, cos, sin, attention_mask)
        hidden_states = self.attn_norm(residual + hidden_states)
        
        # MLP with post-norm
        residual = hidden_states
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.mlp_norm(residual + hidden_states)
        
        return hidden_states


class TRMReasoningModule(nn.Module):
    """
    The core TRM reasoning module that applies recursive reasoning.
    """
    def __init__(self, config: TRMDecoderConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([
            TRMBlock(config) for _ in range(config.num_layers)
        ])
    
    def forward(
        self, 
        hidden_states: torch.Tensor, 
        input_injection: torch.Tensor,
        cos: torch.Tensor, 
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Input injection (add)
        hidden_states = hidden_states + input_injection
        
        # Pass through transformer layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, cos, sin, attention_mask)
        
        return hidden_states


# ============================================================================
# TRM Decoder LLM
# ============================================================================

class TRMDecoderLLM(nn.Module):
    """
    TRM-based Decoder-Only Language Model.
    
    Key Features:
    1. Recursive reasoning with a tiny network
    2. Deep supervision at each recursion step
    3. Single network for both z and y updates
    """
    def __init__(self, config: TRMDecoderConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.embed_scale = math.sqrt(config.hidden_size)
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Rotary embeddings
        self.rotary_emb = RotaryEmbedding(
            dim=config.head_dim,
            max_seq_len=config.max_seq_len,
            base=config.rope_theta
        )
        
        # TRM Reasoning Module (shared across recursions)
        self.reasoning = TRMReasoningModule(config)
        
        # Output projection (LM head)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Initial state for z (learnable)
        self.z_init = nn.Parameter(torch.randn(config.hidden_size) * 0.02)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with truncated normal distribution"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.trunc_normal_(module.weight, std=0.02)
    
    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get token embeddings scaled by sqrt(hidden_size)"""
        return self.embed_scale * self.token_embedding(input_ids)
    
    def forward(
        self, 
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_all_logits: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with TRM recursive reasoning.
        """
        B, L = input_ids.shape
        device = input_ids.device
        
        # Get input embeddings (this is 'x' in TRM)
        x_emb = self.get_input_embeddings(input_ids)
        x_emb = self.dropout(x_emb)
        
        # Get rotary embeddings
        cos, sin = self.rotary_emb(L)
        
        # Initialize y (answer draft) as copy of x
        y_emb = x_emb.clone()
        
        # Initialize z (reasoning state)
        z = self.z_init.unsqueeze(0).unsqueeze(0).expand(B, L, -1)
        
        all_logits = []
        
        # TRM Recursive Reasoning Loop
        for t in range(self.config.t_cycles):
            use_grad = (t == self.config.t_cycles - 1)
            
            with torch.set_grad_enabled(use_grad or self.training):
                # n recursions to update z
                for n in range(self.config.n_recursions):
                    input_injection = x_emb + y_emb
                    z = self.reasoning(z, input_injection, cos, sin, attention_mask)
                
                # Update y given z
                z_for_y = self.reasoning(z, torch.zeros_like(z), cos, sin, attention_mask)
                
                # Compute logits
                logits = self.lm_head(z_for_y)
                all_logits.append(logits)
                
                # Soft update of y_emb
                probs = F.softmax(logits, dim=-1)
                y_emb = probs @ self.token_embedding.weight
                y_emb = self.embed_scale * y_emb
            
            # Detach for next cycle
            z = z.detach()
            y_emb = y_emb.detach()
        
        final_logits = all_logits[-1]
        outputs = {'logits': final_logits}
        
        if return_all_logits:
            outputs['all_logits'] = all_logits
        
        # Compute loss with deep supervision
        if labels is not None:
            shift_labels = labels[:, 1:].contiguous()
            
            total_loss = 0.0
            for cycle_logits in all_logits:
                shift_logits = cycle_logits[:, :-1, :].contiguous()
                loss = F.cross_entropy(
                    shift_logits.view(-1, self.config.vocab_size),
                    shift_labels.view(-1),
                    ignore_index=-100
                )
                total_loss += loss
            
            outputs['loss'] = total_loss / len(all_logits)
        
        return outputs
    
    @torch.no_grad()
    def generate(
        self, 
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None
    ) -> torch.Tensor:
        """Generate text autoregressively."""
        self.eval()
        
        for _ in range(max_new_tokens):
            idx_cond = input_ids[:, -self.config.max_seq_len:]
            outputs = self(idx_cond)
            logits = outputs['logits'][:, -1, :] / temperature
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float('-inf')
            
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
        
        return input_ids


# ============================================================================
# Vision Encoder Components
# ============================================================================

class PatchEmbedding(nn.Module):
    """Convert image to patches and embed them"""
    def __init__(
        self, 
        img_size: int = 224, 
        patch_size: int = 16, 
        in_channels: int = 3, 
        embed_dim: int = 256
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x


class VisionEncoderBlock(nn.Module):
    """Transformer block for vision encoder"""
    def __init__(self, config: VisionEncoderConfig):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True
        )
        self.mlp = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim * 4),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.embed_dim * 4, config.embed_dim),
            nn.Dropout(config.dropout)
        )
        self.norm1 = nn.LayerNorm(config.embed_dim)
        self.norm2 = nn.LayerNorm(config.embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class TRMVisionEncoder(nn.Module):
    """Vision Encoder using TRM-style recursive reasoning."""
    def __init__(self, config: VisionEncoderConfig):
        super().__init__()
        self.config = config
        
        self.patch_embed = PatchEmbedding(
            img_size=config.img_size,
            patch_size=config.patch_size,
            in_channels=config.in_channels,
            embed_dim=config.embed_dim
        )
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.embed_dim) * 0.02)
        self.pos_embed = nn.Parameter(
            torch.randn(1, config.n_patches + 1, config.embed_dim) * 0.02
        )
        
        self.layers = nn.ModuleList([
            VisionEncoderBlock(config) for _ in range(config.num_layers)
        ])
        
        self.norm = nn.LayerNorm(config.embed_dim)
        self.z_init = nn.Parameter(torch.randn(config.embed_dim) * 0.02)
    
    def forward(
        self, 
        x: torch.Tensor, 
        return_all_features: bool = False
    ) -> torch.Tensor:
        B = x.shape[0]
        
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        
        z = self.z_init.unsqueeze(0).unsqueeze(0).expand(B, x.shape[1], -1)
        
        all_features = []
        
        for r in range(self.config.n_recursions):
            h = x + z
            for layer in self.layers:
                h = layer(h)
            z = z + h
            
            if return_all_features:
                all_features.append(self.norm(z)[:, 0])
        
        z = self.norm(z)
        
        if return_all_features:
            return torch.stack(all_features, dim=1)
        
        return z[:, 0]


# ============================================================================
# Projection MLP
# ============================================================================

class ProjectionMLP(nn.Module):
    """MLP to project embeddings to a shared space for contrastive learning."""
    def __init__(
        self, 
        input_dim: int, 
        projection_dim: int, 
        hidden_dim: Optional[int] = None
    ):
        super().__init__()
        hidden_dim = hidden_dim or input_dim * 2
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, projection_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        return F.normalize(x, dim=-1)


# ============================================================================
# TRM Vision-Language Model
# ============================================================================

class TRMVLM(nn.Module):
    """
    TRM Vision-Language Model.
    
    Architecture:
    1. Vision Encoder: TRM-style ViT with recursive reasoning
    2. Language Model: TRM-style decoder with recursive reasoning  
    3. Projection MLPs: Align vision and language to shared space
    """
    def __init__(self, config: TRMVLMConfig):
        super().__init__()
        self.config = config
        
        # Vision Encoder
        vision_config = VisionEncoderConfig(
            img_size=config.img_size,
            patch_size=config.patch_size,
            embed_dim=config.vision_embed_dim,
            num_heads=config.vision_num_heads,
            num_layers=config.vision_num_layers,
            dropout=config.dropout,
            n_recursions=config.vision_n_recursions
        )
        self.vision_encoder = TRMVisionEncoder(vision_config)
        
        # Language Model
        lang_config = TRMDecoderConfig(
            vocab_size=config.vocab_size,
            hidden_size=config.lang_hidden_size,
            num_heads=config.lang_num_heads,
            num_layers=config.lang_num_layers,
            max_seq_len=config.max_seq_len,
            n_recursions=config.lang_n_recursions,
            t_cycles=config.lang_t_cycles,
            dropout=config.dropout
        )
        self.language_model = TRMDecoderLLM(lang_config)
        
        # Projection heads
        self.vision_proj = ProjectionMLP(config.vision_embed_dim, config.projection_dim)
        self.text_proj = ProjectionMLP(config.lang_hidden_size, config.projection_dim)
        
        # Cross-modal projection
        self.vision_to_lang = nn.Linear(config.vision_embed_dim, config.lang_hidden_size)
        
        # Learnable temperature
        self.temperature = nn.Parameter(torch.tensor(config.temperature).log())
    
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images to embeddings."""
        return self.vision_encoder(images)
    
    def encode_text(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Encode text to embeddings."""
        return self.language_model.get_input_embeddings(input_ids)[:, -1, :]
    
    def contrastive_loss(
        self, 
        image_embeds: torch.Tensor, 
        text_embeds: torch.Tensor
    ) -> torch.Tensor:
        """Compute CLIP-style contrastive loss."""
        image_proj = self.vision_proj(image_embeds)
        text_proj = self.text_proj(text_embeds)
        
        temperature = self.temperature.exp()
        logits = (image_proj @ text_proj.T) / temperature
        
        B = logits.shape[0]
        labels = torch.arange(B, device=logits.device)
        
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.T, labels)
        
        return (loss_i2t + loss_t2i) / 2
    
    def forward(
        self,
        images: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        mode: str = 'vlm'
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            images: Input images (B, C, H, W)
            input_ids: Input token IDs (B, L)
            labels: Target labels for LM loss (B, L)
            mode: 'contrastive', 'generation', or 'vlm'
        """
        outputs = {}
        
        if mode == 'contrastive':
            assert images is not None and input_ids is not None
            image_embeds = self.encode_image(images)
            text_embeds = self.encode_text(input_ids)
            
            loss = self.contrastive_loss(image_embeds, text_embeds)
            outputs['loss'] = loss
            outputs['image_embeds'] = image_embeds
            outputs['text_embeds'] = text_embeds
            
        elif mode == 'generation':
            assert input_ids is not None
            lm_outputs = self.language_model(input_ids, labels=labels)
            outputs.update(lm_outputs)
            
        elif mode == 'vlm':
            assert images is not None and input_ids is not None
            image_embeds = self.encode_image(images)
            lm_outputs = self.language_model(input_ids, labels=labels)
            outputs.update(lm_outputs)
            outputs['image_embeds'] = image_embeds
        
        return outputs


# ============================================================================
# Utility Functions
# ============================================================================

def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_trm_llm(
    vocab_size: int = 32000,
    hidden_size: int = 256,
    num_layers: int = 2,
    n_recursions: int = 6,
    t_cycles: int = 3,
    **kwargs
) -> TRMDecoderLLM:
    """Factory function to create a TRM LLM."""
    config = TRMDecoderConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        n_recursions=n_recursions,
        t_cycles=t_cycles,
        **kwargs
    )
    return TRMDecoderLLM(config)


def create_trm_vlm(
    vocab_size: int = 32000,
    vision_embed_dim: int = 256,
    lang_hidden_size: int = 256,
    **kwargs
) -> TRMVLM:
    """Factory function to create a TRM VLM."""
    config = TRMVLMConfig(
        vocab_size=vocab_size,
        vision_embed_dim=vision_embed_dim,
        lang_hidden_size=lang_hidden_size,
        **kwargs
    )
    return TRMVLM(config)


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    # Example usage
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create and test LLM
    print("\n--- TRM Decoder LLM ---")
    llm = create_trm_llm(vocab_size=1000, hidden_size=128, n_recursions=4, t_cycles=2)
    llm = llm.to(device)
    print(f"Parameters: {count_parameters(llm):,}")
    
    dummy_input = torch.randint(0, 1000, (2, 32)).to(device)
    outputs = llm(dummy_input, labels=dummy_input)
    print(f"Loss: {outputs['loss'].item():.4f}")
    print(f"Logits shape: {outputs['logits'].shape}")
    
    # Create and test VLM
    print("\n--- TRM VLM ---")
    vlm = create_trm_vlm(
        vocab_size=1000, 
        vision_embed_dim=128, 
        lang_hidden_size=128,
        img_size=64,
        patch_size=8
    )
    vlm = vlm.to(device)
    print(f"Parameters: {count_parameters(vlm):,}")
    
    dummy_images = torch.randn(2, 3, 64, 64).to(device)
    dummy_text = torch.randint(0, 1000, (2, 16)).to(device)
    
    outputs = vlm(dummy_images, dummy_text, mode='contrastive')
    print(f"Contrastive Loss: {outputs['loss'].item():.4f}")
    
    print("\n--- All tests passed! ---")
