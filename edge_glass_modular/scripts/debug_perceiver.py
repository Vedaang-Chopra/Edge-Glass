import sys
from pathlib import Path
import torch
from torch.optim import AdamW
import numpy as np

# Add src
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from config import load_config
from models.alignment import MultimodalAlignmentModel

def main():
    config_path = project_root / "configs/perceiver_mrl_alignment.yaml"
    config = load_config(str(config_path))
    
    # Disable decoder for faster debugging to isolate alignment
    print("Disabling decoder for debugging...")
    config.decoder = None 
    
    print("Initialize model...")
    model = MultimodalAlignmentModel(config)
    
    if torch.cuda.is_available():
        model.cuda()
    
    model.train()
    
    print("\n--- Parameter Status ---")
    trainable = []
    frozen = []
    for name, p in model.named_parameters():
        if p.requires_grad:
            trainable.append(name)
        else:
            frozen.append(name)
            
    print(f"Total params: {len(trainable) + len(frozen)}")
    print(f"Trainable params: {len(trainable)}")
    print(f"Frozen params: {len(frozen)}")
    
    print("\nChecking key components:")
    print("Vision Encoder Projector Trainable:", any("vision_encoder.projector" in n for n in trainable))
    print("Vision Encoder Perceiver Trainable:", any("vision_encoder.perceiver" in n for n in trainable))
    print("Text Encoder Projector Trainable:", any("text_encoder.projector" in n for n in trainable))
    
    # Dummy forward pass
    print("\n--- Dummy Forward Pass ---")
    image_size = config.dataset.image_size
    images = torch.randn(2, 3, image_size, image_size)
    if torch.cuda.is_available():
        images = images.cuda()
        
    texts = ["a photo of a cat", "a photo of a dog"]
    
    optimizer = AdamW(model.parameters(), lr=1e-4) # Pass all params like Trainer
    
    try:
        outputs = model(images=images, texts=texts)
        loss = outputs.loss
        
        print(f"Loss: {loss.item()}")
        
        loss.backward()
        
        print("\n--- Gradient Check ---")
        params_with_grad = []
        params_without_grad = []
        
        for name, p in model.named_parameters():
            if p.requires_grad:
                if p.grad is not None and p.grad.abs().sum() > 0:
                    params_with_grad.append(name)
                else:
                    params_without_grad.append(name)
                    
        print(f"Trainable params with grad: {len(params_with_grad)}")
        print(f"Trainable params ZERO/NO grad: {len(params_without_grad)}")
        
        if len(params_without_grad) > 0:
            print("Sample params without grad:", params_without_grad[:10])
            
        perceiver_grads = [n for n in params_with_grad if "perceiver" in n]
        if perceiver_grads:
            print(f"SUCCESS: {len(perceiver_grads)} Perceiver params have gradients.")
        else:
            print("FAILURE: Perceiver has NO gradients.")
        
        optimizer.step()
        print("Optimizer stepped.")
        
    except Exception as e:
        print(f"Error during forward/backward: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
