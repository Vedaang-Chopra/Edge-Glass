
import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Union
from src.decoders.qwen import QwenDecoder
from src.decoders.trm import TRMConfig, TRMLayer, RMSNorm

class QwenVLM(nn.Module):
    """
    Qwen-based VLM with optional TRM latent recursion.
    
    Architecture:
    1. Input (Image + Text) -> Aligned Embeddings
    2. Qwen Backbone (Pretrained) -> Hidden States
    3. TRM Recursion (on Hidden States) -> Refined States
    4. LM Head -> Logits
    
    This fixes the issue where Qwen layers were bypassed.
    """
    def __init__(
        self,
        qwen_decoder: QwenDecoder,
        vision_token_dim: int = 4096,
        use_trm_recursion: bool = False,
        trm_config: Optional[TRMConfig] = None,
        num_trm_layers: int = 2,
        num_recursion_steps: int = 4,
        confidence_threshold: float = 0.75,
    ):
        super().__init__()
        self.qwen = qwen_decoder
        self.use_trm_recursion = use_trm_recursion
        self.num_recursion_steps = num_recursion_steps
        self.confidence_threshold = confidence_threshold
        
        # Get dimensions from Qwen
        self.hidden_dim = self.qwen.hidden_dim
        self.vocab_size = self.qwen.vocab_size
        
        # Vision projection: Vision Dim -> Qwen Hidden Dim
        self.vision_proj = nn.Linear(vision_token_dim, self.hidden_dim)
        
        if use_trm_recursion:
            if trm_config is None:
                # Default TRM config matching Qwen dims if possible, or small
                # If we want "Tiny", we might project
                trm_config = TRMConfig(
                    hidden_dim=self.hidden_dim, # Default to matching
                    num_layers=num_trm_layers, # Tiny depth from arg
                    num_heads=16, # Adjust as needed
                )
            
            self.trm_config = trm_config
            self.trm_hidden_dim = trm_config.hidden_dim
            
            # Projectors if dims mismatch
            if self.trm_hidden_dim != self.hidden_dim:
                self.down_proj = nn.Linear(self.hidden_dim, self.trm_hidden_dim)
                self.up_proj = nn.Linear(self.trm_hidden_dim, self.hidden_dim)
            else:
                self.down_proj = nn.Identity()
                self.up_proj = nn.Identity()
                
            # TRM Layers
            self.trm_layers = nn.ModuleList([
                TRMLayer(trm_config) for _ in range(trm_config.num_layers)
            ])
            
            # Norms
            self.trm_norm = RMSNorm(self.trm_hidden_dim)
            
            # Latent token initialization (learnable)
            # z is the "reasoning state"
            self.z_init = nn.Parameter(torch.randn(1, 1, self.trm_hidden_dim) * 0.02)
            
    def forward(
        self,
        # Notebook-style arguments
        vision_tokens: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,  # Alias used in training loop
        question_ids: Optional[torch.Tensor] = None,
        answer_ids: Optional[torch.Tensor] = None,
        answer_mask: Optional[torch.Tensor] = None,
        # Standard HF arguments
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        prefix_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        # 0. Argument Resolution
        # Handle 'images' alias for vision_tokens
        if vision_tokens is None and images is not None:
            vision_tokens = images
            
        # Handle 'vision_tokens' -> 'prefix_embeds' mapping
        if prefix_embeds is None and vision_tokens is not None:
            # We need to project vision tokens first!
            # Project logic moved here to ensure we pass projected embeddings as prefix
            device = vision_tokens.device
            qwen_dtype = self.qwen.model.dtype

            # Align aux modules
            if self.vision_proj.weight.device != device or self.vision_proj.weight.dtype != qwen_dtype:
                self.vision_proj = self.vision_proj.to(device=device, dtype=qwen_dtype)
                
            if self.use_trm_recursion:
                self.trm_layers.to(device=device, dtype=qwen_dtype)
                self.trm_norm.to(device=device, dtype=qwen_dtype)
                if self.z_init.device != device or self.z_init.dtype != qwen_dtype:
                    self.z_init.data = self.z_init.data.to(device=device, dtype=qwen_dtype)

            vision_tokens = vision_tokens.to(device=device, dtype=qwen_dtype)
            prefix_embeds = self.vision_proj(vision_tokens)

        # Handle question_ids + answer_ids -> input_ids
        if input_ids is None and question_ids is not None and answer_ids is not None:
            input_ids = torch.cat([question_ids, answer_ids], dim=1)
            
            # Create labels if not provided
            # Standard VLM QA: -100 for question, answer_ids for answer
            if labels is None:
                question_labels = torch.full_like(question_ids, fill_value=-100)
                labels = torch.cat([question_labels, answer_ids], dim=1)
                
        # Fallback if just question_ids (e.g. generation start)
        if input_ids is None and question_ids is not None:
            input_ids = question_ids

        # 1. Run Qwen Backbone
        # We need hidden states, not just logits
        outputs = self.qwen(
            input_ids=input_ids,
            attention_mask=attention_mask,
            prefix_embeds=prefix_embeds,
            labels=labels, # Pass labels to get 'loss' automatically from Qwen
            output_hidden_states=True,
            return_dict=True,
        )
        
        # If no input_ids provided (impossible?), check
        if input_ids is None:
             raise ValueError("forward() requires input_ids OR (question_ids + answer_ids)")

        # Qwen hidden state (last layer)
        # Use last_hidden_state: (B, Seq, Dim)
        qwen_hidden = outputs.hidden_states[-1]
        
        if not self.use_trm_recursion:
            # Return standard dictionary matching what notebook expects ('loss', 'logits', 'confidence')
            # But QwenDecoder output is CausalLMOutputWithPast (loss, logits, hidden_states...)
            # We should wrap it into dictionary if notebook expects dict
            
            # Check what notebook expects:
            # outputs = model(...); loss = outputs.loss
            # It expects an object with .loss attribute.
            # CausalLMOutputWithPast has .loss attribute.
            return outputs
            
        # 2. TRM Recursion Mode
        
        # We need to refine the answer part of the sequence.
        # But this is training (teacher forcing), so we have the whole sequence.
        # TRM strategy usually: Fixed Context (Image+Question) | Answer
        # We refine Answer representation using Context.
        
        # For simplicity in this "VLM QA" setup where we train on the whole sequence:
        # We can apply TRM to the whole sequence (refining everything) OR just answer.
        # Applying to whole sequence is easier implementation-wise for a Transformer.
        
        # Project down if needed
        # Ensure qwen_hidden matches TRM device target
        # Align auxiliary modules to Qwen dtype/device to avoid dtype/device mismatches
        target_device = qwen_hidden.device
        target_dtype = qwen_hidden.dtype
        if hasattr(self.down_proj, 'weight'):
            self.down_proj = self.down_proj.to(device=target_device, dtype=target_dtype)
        if hasattr(self.up_proj, 'weight'):
            self.up_proj = self.up_proj.to(device=target_device, dtype=target_dtype)
        self.trm_layers.to(device=target_device, dtype=target_dtype)
        self.trm_norm.to(device=target_device, dtype=target_dtype)
        if self.z_init.device != target_device or self.z_init.dtype != target_dtype:
            self.z_init.data = self.z_init.data.to(device=target_device, dtype=target_dtype)

        x = self.down_proj(qwen_hidden) # (B, Seq, trm_dim)
        
        # Initialize latent z
        batch_size, seq_len, _ = x.shape
        z = self.z_init.expand(batch_size, seq_len, -1) # (B, Seq, trm_dim)
        
        # State 'y' initialized from x
        y = x
        
        # Iterate
        for _ in range(self.num_recursion_steps):
            combined = torch.cat([y, z], dim=1) # Concat along sequence dim -> (B, 2*Seq, Dim)
            
            hidden = combined
            for layer in self.trm_layers:
                hidden = layer(hidden)
            
            # Split back
            output_y, output_z = hidden.chunk(2, dim=1)
            
            # Update
            y = output_y
            z = output_z
            
        # Final projection up and residual connection
        y_up = self.up_proj(y) # (B, Seq, Qwen_Dim)
        # Move y_up back to qwen_hidden dtype/device for residual add
        y_up = y_up.to(device=qwen_hidden.device, dtype=qwen_hidden.dtype)
        final_hidden = qwen_hidden + y_up
        
        # LM Head
        if hasattr(self.qwen.model, "lm_head"):
            logits = self.qwen.model.lm_head(final_hidden)
        else:
             base = self.qwen.model.base_model.model if hasattr(self.qwen.model, "base_model") else self.qwen.model
             logits = base.lm_head(final_hidden)

        # Calculate loss if labels provided
        tr_loss = None
        if labels is not None:
            # If we had a prefix, logits include it. Labels likely don't (they are text only).
            # We need to slice logits to match labels length or pad labels.
            # Usually labels corresponds to input_ids (text).
            
            if prefix_embeds is not None:
                num_prefix = prefix_embeds.shape[1]
                # Logits shape: (B, num_prefix + seq_len, Vocab)
                # Labels shape: (B, seq_len)
                if logits.shape[1] > labels.shape[1]:
                    # Slice off the prefix part from logits
                    logits_for_loss = logits[:, num_prefix:, :]
                else:
                    logits_for_loss = logits
            else:
                logits_for_loss = logits

            # Shift so that tokens < n predict n
            shift_logits = logits_for_loss[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            tr_loss = loss_fct(shift_logits.view(-1, self.vocab_size), shift_labels.view(-1))
            
        from transformers.modeling_outputs import CausalLMOutputWithPast
        return CausalLMOutputWithPast(
            loss=tr_loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
        )

    @torch.no_grad()
    def generate(
        self,
        vision_tokens: torch.Tensor,
        question_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 32,
        temperature: float = 0.0,
        return_stats: bool = False,
        **kwargs,
    ):
        """Generate answers with optional TRM recursion."""
        # Project vision
        # Keep everything on the text/question device to avoid cross-GPU issues
        device = question_ids.device if question_ids is not None else vision_tokens.device
        target_dtype = self.qwen.model.dtype

        # Align vision projection with Qwen sharded dtype/device
        if self.vision_proj.weight.device != device or self.vision_proj.weight.dtype != target_dtype:
            self.vision_proj = self.vision_proj.to(device=device, dtype=target_dtype)

        vision_tokens = vision_tokens.to(
            device=device,
            dtype=self.vision_proj.weight.dtype,
        )

        vision_emb = self.vision_proj(vision_tokens)
        
        # Ensure mask exists if not provided
        if attention_mask is None:
            # Default to all-valid if no pad_token_id, or mask padding
            pad_id = self.qwen.tokenizer.pad_token_id
            if pad_id is None: 
                attention_mask = torch.ones_like(question_ids)
            else:
                attention_mask = (question_ids != pad_id).long()
        
        if attention_mask.device != question_ids.device:
            attention_mask = attention_mask.to(question_ids.device)
        
        if not self.use_trm_recursion:
            # Baseline: standard Qwen generation
            outputs = self.qwen.generate(
                input_ids=question_ids,
                prefix_embeds=vision_emb,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=max(temperature, 0.01) if temperature > 0 else 1.0,
                do_sample=temperature > 0,
                pad_token_id=self.qwen.tokenizer.pad_token_id,
                eos_token_id=self.qwen.tokenizer.eos_token_id,
            )
            
            # Extract only generated part
            # Qwen.generate returns [input + generated] usually
            prompt_len = question_ids.shape[1] 
            # Note: prefix_embeds are handled internally, input_ids length is just text prompt
            # But wait, QwenDecoder.generate builds input_ids internally?
            # No, QwenDecoder.generate takes input_ids.
            # outputs shape: (B, Seq)
            generated_ids = outputs[:, prompt_len:]
            
            return generated_ids
        
        else:
            # TRM mode: autoregressive with recursion
            # Re-implementation of the loop ensuring TRM is applied at each step
            batch_size = vision_tokens.shape[0]
            generated_ids_list = []
            
            # Helper to pad answer_ids for first step
            pad_token_id = self.qwen.tokenizer.pad_token_id
            
            # Initialize current attention mask with question mask
            current_attention_mask = attention_mask
            
            for step in range(max_new_tokens):
                # Construct current answer sequence
                if len(generated_ids_list) == 0:
                     # FIRST STEP: Just use the question (and vision prefix)
                     # Do NOT add a pad token, as it confuses the model
                     current_input_ids = question_ids
                else:
                    answer_ids = torch.stack(generated_ids_list, dim=1)
                    # Ensure answer_ids is on the same device as question_ids
                    if answer_ids.device != question_ids.device:
                        answer_ids = answer_ids.to(question_ids.device)
                    
                    current_input_ids = torch.cat([question_ids, answer_ids], dim=1)
                    
                    # Update mask for the new token (valid)
                    # We need to extend mask by 1 for each step. 
                    # Optimization: Extend just once per step
                    # current_attention_mask was for previous step?
                    # No, let's just rebuild/extend it.
                    # Base mask size: Q
                    # Current input size: Q + step
                    # We need to append '1' to current_attention_mask

                    ones = torch.ones((batch_size, 1), device=current_attention_mask.device, dtype=attention_mask.dtype)
                    current_attention_mask = torch.cat([current_attention_mask, ones], dim=1)

                out = self.forward(
                    input_ids=current_input_ids,
                    attention_mask=current_attention_mask,
                    prefix_embeds=vision_emb
                )
                
                # logits = out.logits
                logits = out['logits'][:, -1, :]  # Last position
                
                if temperature > 0:
                    logits = logits / temperature
                    probs = torch.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
                else:
                    next_token = torch.argmax(logits, dim=-1)
                
                # Stop if EOS
                # But we have batch.
                generated_ids_list.append(next_token)
                
            return torch.stack(generated_ids_list, dim=1)
