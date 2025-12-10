
import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell
import os

nb = new_notebook()

# Cell 1: Imports
nb.cells.append(new_markdown_cell("# VLM QA Evaluation Notebook\n\nThis notebook evaluates the trained TRM-VLM model on the validation dataset."))
nb.cells.append(new_code_cell("""
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
from pathlib import Path

# Add src to path
project_root = Path(os.path.abspath('..'))
sys.path.append(str(project_root))

from src.config import Config, load_config
from src.decoders.qwen import QwenDecoder
from src.encoders.vision import VisionEncoder
from src.models.trm_qwen_vlm import QwenVLM, TRMConfig
from src.data.pixmo_dataset import PixmoQADataset, get_image_transforms

%matplotlib inline
"""))

# Cell 2: Configuration
nb.cells.append(new_markdown_cell("## 1. Configuration & Setup"))
nb.cells.append(new_code_cell("""
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load Config
config_path = "../src/configs/trm_vlm_qa.yaml"
config = load_config(config_path)

# Verify paths
data_dir = Path(config.dataset.data_path)
print(f"Data Directory: {data_dir}")
"""))

# Cell 3: Load Model
nb.cells.append(new_markdown_cell("## 2. Load Model"))
nb.cells.append(new_code_cell("""
# 1. Vision Encoder
print("Loading Vision Encoder...")
vision_encoder = VisionEncoder(
    model_name=config.vision_encoder.model_name,
    trainable=False
).to(device)

# 2. Qwen Decoder (LoRA)
print("Loading Qwen Decoder...")
qwen_decoder = QwenDecoder(
    model_name=config.decoder.model_name,
    device_map="balanced", # Use balanced for multi-gpu inference
    use_lora=True,
    lora_config=config.decoder.lora
)

# 3. TRM VLM Wrapper
trm_config = TRMConfig(
    hidden_dim=qwen_decoder.hidden_dim,
    trm_hidden_dim=config.optimization.trm_hidden_dim,
    trm_num_layers=config.optimization.trm_num_layers,
    num_inner_steps=config.optimization.num_inner_steps,
    num_outer_steps=config.optimization.num_outer_steps,
    dropout=config.optimization.dropout
)

model = QwenVLM(
    qwen_decoder=qwen_decoder,
    vision_token_dim=vision_encoder.output_dim,
    use_trm_recursion=True,
    trm_config=trm_config,
    num_trm_layers=config.optimization.trm_num_layers,
    num_recursion_steps=config.optimization.num_inner_steps
).to(device)

print("Model Initialized.")
"""))

# Cell 4: Load Checkpoint
nb.cells.append(new_markdown_cell("## 3. Load Checkpoint"))
nb.cells.append(new_code_cell("""
ckpt_dir = Path("../checkpoints/qwen_vlm_qa_trm")
best_ckpt_path = ckpt_dir / "checkpoint_best.pt"

if best_ckpt_path.exists():
    print(f"Loading best checkpoint from {best_ckpt_path}...")
    checkpoint = torch.load(best_ckpt_path, map_location=device)
    
    # Load state dict with strict=False to avoid missing keys if any non-essential ones are missing
    # But usually we want strict=True for model validity. 
    # Since we saved wrapped model state, we should load it.
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Successfully loaded checkpoint (Loss: {checkpoint.get('loss', 'N/A'):.4f})")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
else:
    print(f"Warning: {best_ckpt_path} not found. Using initialized weights (untrained).")

model.eval()
"""))

# Cell 5: Dataset & Transforms
nb.cells.append(new_markdown_cell("## 4. Prepare Validation Data"))
nb.cells.append(new_code_cell("""
# Transforms
val_transforms = get_image_transforms(is_training=False, image_size=config.dataset.image_size)

# Dataset
val_dataset = PixmoQADataset(
    data_dir=config.dataset.data_path,
    tokenizer=qwen_decoder.tokenizer,
    transform=val_transforms,
    split="validation", # Use validation split
    max_tokens=config.dataset.max_tokens
)

print(f"Validation Samples: {len(val_dataset)}")

# Helper for vision encoding
def encode_images(images):
    # Shape: (B, C, H, W) -> (B, Num_Visual_Tokens, Dim)
    # The vision encoder expects images on its device
    if not isinstance(images, torch.Tensor):
        images = images.to(device)
    
    with torch.no_grad():
        features = vision_encoder(images)
    return features
"""))

# Cell 6: Evaluation Loop
nb.cells.append(new_markdown_cell("## 5. Evaluation Loop\nRunning inference on a subset of the validation set."))
nb.cells.append(new_code_cell("""
num_samples = 50 # Limit for quick evaluation
indices = np.random.choice(len(val_dataset), num_samples, replace=False)

results = []
print(f"Evaluating on {num_samples} random samples...")

for idx in tqdm(indices):
    sample = val_dataset[idx]
    image = sample['images'].unsqueeze(0).to(device)
    question_ids = sample['question_ids'].unsqueeze(0).to(device)
    answer_ids_gt = sample['answer_ids'].unsqueeze(0).to(device)
    
    # Decode helper
    tokenizer = qwen_decoder.tokenizer
    question_text = tokenizer.decode(sample['question_ids'], skip_special_tokens=True)
    ground_truth = tokenizer.decode(sample['answer_ids'], skip_special_tokens=True)
    
    # Inference
    with torch.no_grad():
        vision_tokens = encode_images(image)
        
        # We assume max new tokens 
        generated_ids = model.generate(
            vision_tokens=vision_tokens,
            question_ids=question_ids,
            max_new_tokens=32,
            temperature=0.0, # Greedy
            return_stats=True
        )
        
    prediction = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    results.append({
        "image_path": sample['image_path'],
        "question": question_text,
        "ground_truth": ground_truth,
        "prediction": prediction,
        "is_correct": ground_truth.strip().lower() in prediction.strip().lower() # Simple containment check
    })

print("Evaluation Complete.")
"""))

# Cell 7: Quantitative Analysis
nb.cells.append(new_markdown_cell("## 6. Quantitative Analysis"))
nb.cells.append(new_code_cell("""
correct_count = sum(1 for r in results if r['is_correct'])
accuracy = correct_count / len(results)

print(f"Estimated Accuracy (Exact/Partial Match): {accuracy:.2%} ({correct_count}/{len(results)})")

# Plot correctness distribution
plt.figure(figsize=(6, 4))
sns.barplot(x=['Incorrect', 'Correct'], y=[len(results)-correct_count, correct_count], palette=['salmon', 'lightgreen'])
plt.title("Prediction Correctness Distribution")
plt.ylabel("Count")
plt.show()
"""))

# Cell 8: Qualitative Visualization
nb.cells.append(new_markdown_cell("## 7. Qualitative Visualization\nDisplaying grid of predictions."))
nb.cells.append(new_code_cell("""
def visualize_results(results, num_display=16, cols=4):
    rows = (num_display + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*5))
    axes = axes.flatten()
    
    for i in range(num_display):
        if i >= len(results):
            break
            
        res = results[i]
        ax = axes[i]
        
        # Load image
        # Assuming image_path is absolute or relative to notebook?
        # Dataset stores usually full paths or absolute.
        try:
            img = Image.open(res['image_path'])
            ax.imshow(img)
        except:
            ax.text(0.5, 0.5, "Image Not Found", ha='center')
            
        color = 'green' if res['is_correct'] else 'red'
        
        # Text
        q_text = (res['question'][:40] + '..') if len(res['question']) > 40 else res['question']
        gt_text = res['ground_truth']
        pred_text = res['prediction']
        
        title = f"Q: {q_text}\nGT: {gt_text}\nPred: {pred_text}"
        
        ax.set_title(title, fontsize=9, color=color, fontweight='bold')
        ax.axis('off')
        
    plt.tight_layout()
    plt.show()

# Show first 16 results
visualize_results(results, num_display=16)
"""))

# Write notebook
output_path = "notebooks/04_vlm_qa_evaluation.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print(f"Notebook generated at {output_path}")
