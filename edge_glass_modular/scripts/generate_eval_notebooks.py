
import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell
import os

def create_quantitative_eval_notebook():
    nb = new_notebook()
    
    nb.cells.append(new_markdown_cell("# VLM Quantitative Evaluation\n\nThis notebook evaluates the VLM on the validation set, computing Accuracy, Exact Match, and F1 scores with visualizations."))
    
    # Imports
    nb.cells.append(new_code_cell("""
import os
import sys
import torch
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from pathlib import Path
from PIL import Image

# Add src to path
project_root = Path(os.path.abspath('..'))
sys.path.append(str(project_root))
sys.path.append(str(project_root / "src"))

from config import load_config
from models.alignment import MultimodalAlignmentModel
from models.trm_qwen_vlm import QwenVLM
from decoders.qwen import QwenDecoder
from data.dataset_builder import PixmoQADataset
from data.transforms import get_image_transforms
from evaluation.qa_metrics import evaluate_qa_metrics, compute_f1, compute_exact_match

%matplotlib inline
plt.rcParams['figure.figsize'] = [10, 6]
"""))

    # Config
    nb.cells.append(new_markdown_cell("## 1. Configuration"))
    nb.cells.append(new_code_cell("""
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Paths
config_path = "../configs/trm_vlm_qa_qwen1.5.yaml"
alignment_config_path = "../configs/pixmo_alignment.yaml"
# Update this path to your trained checkpoint!
checkpoint_path = "../checkpoints/vlm_run/checkpoint-epoch-9" 
# Or if you used the debug run:
# checkpoint_path = "../checkpoints/vlm_debug/checkpoint-epoch-0"

alignment_checkpoint = "../notebooks/checkpoints/pixmo_alignment/checkpoint_best.pt"

use_trm = True
"""))

    # Load Models
    nb.cells.append(new_markdown_cell("## 2. Load Models"))
    nb.cells.append(new_code_cell("""
# 1. Aligned Vision Encoder
print("Loading Vision Encoder...")
alignment_config = load_config(alignment_config_path)
alignment_config.decoder = None
alignment_config.text_encoder = None

aligned_model = MultimodalAlignmentModel(alignment_config)

if os.path.exists(alignment_checkpoint):
    ckpt = torch.load(alignment_checkpoint, map_location='cpu', weights_only=False)
    aligned_model.load_state_dict(ckpt['model_state_dict'], strict=False)
    print("✓ Loaded alignment checkpoint")
else:
    print(f"⚠ Warning: Alignment checkpoint not found at {alignment_checkpoint}")

aligned_model.eval().to(device)
for p in aligned_model.parameters(): p.requires_grad = False

def encode_images(images):
    with torch.no_grad():
        if images.dim() == 3: images = images.unsqueeze(0)
        images = images.to(device)
        return aligned_model.vision_encoder(images, return_sequence=True).sequence

# 2. VLM
print("Loading VLM...")
config = load_config(config_path)
qwen_decoder = QwenDecoder(
    model_name=config.decoder.model_name,
    load_in_8bit=config.decoder.load_in_8bit,
    load_in_4bit=config.decoder.load_in_4bit,
    use_lora=config.decoder.use_lora,
    lora_r=config.decoder.lora_r,
    lora_alpha=config.decoder.lora_alpha,
    lora_dropout=config.decoder.lora_dropout,
    lora_target_modules=config.decoder.lora_target_modules,
    device_map="auto",
    num_key_value_heads=getattr(config.decoder, "num_key_value_heads", None),
    intermediate_size=getattr(config.decoder, "intermediate_size", None),
)

vision_token_dim = alignment_config.vision_encoder.projection_dim
model = QwenVLM(
    qwen_decoder=qwen_decoder,
    vision_token_dim=vision_token_dim,
    use_trm_recursion=use_trm,
    num_trm_layers=4,
    num_recursion_steps=4
).to(device)

# Load Checkpoint
print(f"Loading VLM checkpoint from {checkpoint_path}")
if os.path.isdir(checkpoint_path):
    # If saved with accelerator.save_state, it's a dir. 
    # We might need to manually load pytorch_model.bin if it exists or use accelerator
    bin_path = Path(checkpoint_path) / "pytorch_model.bin"
    if bin_path.exists():
        state_dict = torch.load(bin_path, map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
        print("✓ Loaded state dict from bin file")
    else:
        print("⚠ Could not find pytorch_model.bin in directory. Checkpoint loading might fail.")
else:
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    if 'model_state_dict' in state_dict: state_dict = state_dict['model_state_dict']
    model.load_state_dict(state_dict, strict=False)
    print("✓ Loaded state dict from file")

model.eval()
print("Models Ready.")
"""))

    # Data Loading
    nb.cells.append(new_markdown_cell("## 3. Data Loading"))
    nb.cells.append(new_code_cell("""
val_transforms = get_image_transforms(config.dataset.image_size, is_training=False)
val_dataset = PixmoQADataset(
    parquet_path=config.dataset.val_parquet,
    tokenizer=qwen_decoder.tokenizer,
    image_transforms=val_transforms,
    max_question_length=128,
    max_answer_length=256,
    limit=100 # Evaluate on 100 samples for speed, set to None for full set
)

print(f"Validation Samples: {len(val_dataset)}")
"""))

    # Evaluation Loop
    nb.cells.append(new_markdown_cell("## 4. Evaluation Loop"))
    nb.cells.append(new_code_cell("""
results = []
print("Running Inference...")

for i in tqdm(range(len(val_dataset))):
    sample = val_dataset[i]
    image = sample['image']
    question = sample['question'] # Raw text
    answer_gt = sample['answer']  # Raw text
    
    # Encode Vision
    vision_tokens = encode_images(image)
    
    # Tokenize Question
    inputs = qwen_decoder.tokenizer([question], return_tensors='pt', padding=True).to(device)
    
    # Generate
    with torch.no_grad():
        gen_ids = model.generate(
            vision_tokens=vision_tokens,
            question_ids=inputs.input_ids,
            max_new_tokens=64,
            temperature=0.0 # Greedy
        )
    
    pred = qwen_decoder.tokenizer.decode(gen_ids[0], skip_special_tokens=True)
    
    # Metrics
    f1 = compute_f1(pred, answer_gt)
    em = compute_exact_match(pred, answer_gt)
    
    results.append({
        "question": question,
        "ground_truth": answer_gt,
        "prediction": pred,
        "f1": f1,
        "exact_match": em,
        "image_idx": i
    })

df = pd.DataFrame(results)
print(f"Average F1: {df['f1'].mean()*100:.2f}%")
print(f"Average EM: {df['exact_match'].mean()*100:.2f}%")
"""))

    # Visualizations
    nb.cells.append(new_markdown_cell("## 5. Visualizations"))
    nb.cells.append(new_code_cell("""
# Score Distribution
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(df['f1'], bins=10, kde=False, color='skyblue')
plt.title("F1 Score Distribution")
plt.xlabel("F1 Score")

plt.subplot(1, 2, 2)
sns.countplot(x='exact_match', data=df, palette=['salmon', 'lightgreen'])
plt.title("Exact Match Count")
plt.xticks([0, 1], ['Miss', 'Hit'])
plt.show()
"""))

    # Failure Analysis
    nb.cells.append(new_markdown_cell("## 6. Failure Analysis\nExamine top failures (Low F1)."))
    nb.cells.append(new_code_cell("""
failures = df[df['f1'] < 0.5].head(5)

for _, row in failures.iterrows():
    print(f"Q: {row['question']}")
    print(f"GT: {row['ground_truth']}")
    print(f"Pred: {row['prediction']}")
    print(f"F1: {row['f1']:.2f}")
    # Display image?
    # sample = val_dataset[row['image_idx']]
    # T.ToPILImage()(sample['image']).show() 
    print("-" * 40)
"""))

    return nb

def create_qualitative_inference_notebook():
    nb = new_notebook()
    
    nb.cells.append(new_markdown_cell("# VLM Qualitative Inference\n\nInteractive probing of the VLM with custom images and questions."))
    
    # Setup
    nb.cells.append(new_code_cell("""
import os
import sys
import torch
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

# Add src to path
project_root = Path(os.path.abspath('..'))
sys.path.append(str(project_root))
sys.path.append(str(project_root / "src"))

from config import load_config
from models.alignment import MultimodalAlignmentModel
from models.trm_qwen_vlm import QwenVLM
from decoders.qwen import QwenDecoder
from data.transforms import get_image_transforms

%matplotlib inline

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
"""))

    # Load Model (Collapsed for brevity in notebook)
    nb.cells.append(new_code_cell("""
# --- Configuration ---
config_path = "../configs/trm_vlm_qa_qwen1.5.yaml"
alignment_config_path = "../configs/pixmo_alignment.yaml"
# Update checkpoint path!
checkpoint_path = "../checkpoints/vlm_run/checkpoint-epoch-9" 
alignment_checkpoint = "../notebooks/checkpoints/pixmo_alignment/checkpoint_best.pt"
use_trm = True

# --- Load Models ---
print("Loading Models...")
# 1. Vision
alignment_config = load_config(alignment_config_path)
alignment_config.decoder = None; alignment_config.text_encoder = None
aligned_model = MultimodalAlignmentModel(alignment_config)
if os.path.exists(alignment_checkpoint):
    ckpt = torch.load(alignment_checkpoint, map_location='cpu', weights_only=False)
    aligned_model.load_state_dict(ckpt['model_state_dict'], strict=False)
aligned_model.eval().to(device)

    # 2. VLM
    config = load_config(config_path)
    qwen_decoder = QwenDecoder(
        config.decoder.model_name,
        load_in_8bit=config.decoder.load_in_8bit,
        load_in_4bit=config.decoder.load_in_4bit,
        use_lora=config.decoder.use_lora,
        lora_r=config.decoder.lora_r,
        lora_alpha=config.decoder.lora_alpha,
        lora_dropout=config.decoder.lora_dropout,
        lora_target_modules=config.decoder.lora_target_modules,
        device_map=\"auto\",
        num_key_value_heads=getattr(config.decoder, \"num_key_value_heads\", None),
        intermediate_size=getattr(config.decoder, \"intermediate_size\", None),
    )
model = QwenVLM(
    qwen_decoder, alignment_config.vision_encoder.projection_dim,
    use_trm_recursion=use_trm, num_trm_layers=4, num_recursion_steps=4
).to(device)

# Checkpoint
if os.path.isdir(checkpoint_path):
    bin_path = Path(checkpoint_path) / "pytorch_model.bin"
    if bin_path.exists(): model.load_state_dict(torch.load(bin_path, map_location='cpu'), strict=False)
else:
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu')['model_state_dict'], strict=False)
model.eval()
print("Ready!")
"""))

    # Inference Function
    nb.cells.append(new_markdown_cell("## Interactive Inference"))
    nb.cells.append(new_code_cell("""
def run_inference(image_path, question):
    # Load and Plot Image
    try:
        if isinstance(image_path, str):
            image = Image.open(image_path).convert('RGB')
        else:
            image = image_path # Allow passing PIL object
            
        plt.figure(figsize=(6,6))
        plt.imshow(image)
        plt.axis('off')
        plt.title(f"Q: {question}")
        plt.show()
        
        # Transform
        transform = get_image_transforms(config.dataset.image_size, is_training=False)
        img_tensor = transform(image).unsqueeze(0).to(device)
        
        # Encode
        with torch.no_grad():
            vision_tokens = aligned_model.vision_encoder(img_tensor, return_sequence=True).sequence
            
        # Generate
        inputs = qwen_decoder.tokenizer([question], return_tensors='pt', padding=True).to(device)
        with torch.no_grad():
            gen_ids = model.generate(
                vision_tokens=vision_tokens,
                question_ids=inputs.input_ids,
                max_new_tokens=128,
                temperature=0.2
            )
        
        answer = qwen_decoder.tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        print(f"🤖 Answer: {answer}")
        
    except Exception as e:
        print(f"Error: {e}")

# Example Usage
# run_inference("/path/to/image.jpg", "What is in this image?")
"""))

    return nb

if __name__ == "__main__":
    os.makedirs("notebooks", exist_ok=True)
    
    # 1. Quantitative
    nb1 = create_quantitative_eval_notebook()
    with open("notebooks/04_vlm_qa_evaluation.ipynb", "w") as f:
        nbformat.write(nb1, f)
    print("Created notebooks/04_vlm_qa_evaluation.ipynb")
    
    # 2. Qualitative
    nb2 = create_qualitative_inference_notebook()
    with open("notebooks/05_vlm_qualitative_inference.ipynb", "w") as f:
        nbformat.write(nb2, f)
    print("Created notebooks/05_vlm_qualitative_inference.ipynb")
