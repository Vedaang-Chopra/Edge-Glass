
import sys
import os
from pathlib import Path
import torch
import torch.nn as nn
from dataclasses import dataclass

# Setup path
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from encoders.vision import VisionEncoder
from encoders.perceiver import PerceiverResampler
from config import load_config

def debug_gradients():
    print("Initializing VisionEncoder for debug...")
    
    # Manually defined config to match perceiver_mrl_alignment.yaml
    model_name = "openai/clip-vit-large-patch14-336"
    projection_dim = 4096
    perceiver_latent_dim = 1024
    perceiver_num_latents = 64
    perceiver_num_layers = 4
    perceiver_num_heads = 8
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = VisionEncoder(
        model_name=model_name,
        projection_dim=projection_dim,
        freeze=True,
        use_perceiver=True,
        perceiver_num_latents=perceiver_num_latents,
        perceiver_latent_dim=perceiver_latent_dim,
        perceiver_num_layers=perceiver_num_layers,
        perceiver_num_heads=perceiver_num_heads,
        use_mrl=False # simplify for check
    ).to(device)
    
    print("Model initialized.")
    print("Checking trainable parameters:")
    trainable_params = [n for n, p in model.named_parameters() if p.requires_grad]
    print(f"Total trainable params: {len(trainable_params)}")
    for name in trainable_params[:5]:
        print(f"  - {name}")
    print("  ... (rest truncated)")
    
    if len(trainable_params) == 0:
        print("CRITICAL ERROR: No trainable parameters found!")
        return

    # Create dummy input
    print("\nCreating dummy input...")
    # Batch 2, 3 channels, 336x336
    images = torch.randn(2, 3, 336, 336).to(device)
    
    print("Running forward pass...")
    output = model(images)
    print(f"Output shape: {output.pooled.shape}")
    
    loss = output.pooled.sum()
    print(f"Dummy loss: {loss.item()}")
    
    print("Running backward pass...")
    loss.backward()
    
    print("\nInspecting gradients:")
    zero_grads = []
    none_grads = []
    good_grads = []
    
    for name, p in model.named_parameters():
        if p.requires_grad:
            if p.grad is None:
                none_grads.append(name)
            elif p.grad.abs().sum() == 0:
                zero_grads.append(name)
            else:
                good_grads.append(name)
                
    print(f"Params with Good Gradients: {len(good_grads)}")
    print(f"Params with Zero Gradients: {len(zero_grads)}")
    print(f"Params with None Gradients: {len(none_grads)}")
    
    if len(zero_grads) > 0:
        print("\nExamples of Zero Gradients:")
        for n in zero_grads[:5]:
            print(f"  - {n}")
            
    if len(good_grads) > 0:
        print("\nExamples of Good Gradients:")
        for n in good_grads[:5]:
            p = dict(model.named_parameters())[n]
            print(f"  - {n}: Mean {p.grad.abs().mean().item():.6f}, Max {p.grad.abs().max().item():.6f}")

if __name__ == "__main__":
    debug_gradients()
