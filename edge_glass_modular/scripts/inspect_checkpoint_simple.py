
import torch
import sys
import os

def inspect_checkpoint(checkpoint_path):
    print(f"Inspecting checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    
    keys_to_check = [
        "qwen.model.base_model.model.model.layers.0.self_attn.k_proj.base_layer.weight",
        "qwen.model.base_model.model.model.layers.0.self_attn.k_proj.base_layer.bias",
        "qwen.model.base_model.model.model.layers.0.self_attn.q_proj.base_layer.weight",
        "qwen.model.base_model.model.model.layers.0.self_attn.q_proj.base_layer.bias",
        "qwen.model.base_model.model.model.layers.0.self_attn.v_proj.base_layer.weight",
        "qwen.model.base_model.model.model.layers.0.self_attn.v_proj.base_layer.bias",
    ]
    
    # Also attempt matching without 'base_model...' prefix if keys are simpler
    
    for k in state_dict.keys():
        if "layers.0.self_attn" in k:
            print(f"{k}: {state_dict[k].shape}")
            
    # Explicit check
    # ...

if __name__ == "__main__":
    path = "outputs/demo_run_3b/checkpoint-debug/pytorch_model.bin/pytorch_model.bin"
    inspect_checkpoint(path)
