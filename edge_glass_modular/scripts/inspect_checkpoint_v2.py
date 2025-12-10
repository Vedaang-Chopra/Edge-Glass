
import torch
import sys
import os

def inspect_checkpoint(checkpoint_path):
    print(f"Inspecting checkpoint: {checkpoint_path}")
    try:
        state_dict = torch.load(checkpoint_path, map_location="cpu")
    except Exception as e:
        print(f"Failed to load with torch.load: {e}")
        return

    print(f"Total keys: {len(state_dict)}")
    
    # Check for any key containing 'k_proj' and 'bias'
    print("\n--- Keys matching 'k_proj' and 'bias' ---")
    for k in state_dict.keys():
        if "k_proj" in k and "bias" in k:
            print(f"{k}: {state_dict[k].shape}")

    # Check for any key matching 'k_proj' and 'weight'
    print("\n--- Keys matching 'k_proj' and 'weight' (first 5 layers) ---")
    count = 0
    for k in state_dict.keys():
        if "k_proj" in k and "weight" in k:
            layer_idx = k.split("layers.")[1].split(".")[0] if "layers." in k else "unknown"
            if layer_idx.isdigit() and int(layer_idx) < 5:
                print(f"{k}: {state_dict[k].shape}")

if __name__ == "__main__":
    path = "outputs/demo_run_3b/checkpoint-debug/pytorch_model.bin/pytorch_model.bin"
    inspect_checkpoint(path)
