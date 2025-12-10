
import json
import sys
from pathlib import Path

def fix_notebook(notebook_path):
    path = Path(notebook_path)
    if not path.exists():
        print(f"Error: {path} not found")
        sys.exit(1)
        
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    # --- Cell 1: Imports ---
    imports_source = [
        "import os\n",
        "import sys\n",
        "import torch\n",
        "import torch.nn as nn\n",
        "from torch.utils.data import DataLoader\n",
        "from transformers import AutoTokenizer, AutoModelForCausalLM\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "import numpy as np\n",
        "from PIL import Image\n",
        "from tqdm.auto import tqdm\n",
        "from pathlib import Path\n",
        "\n",
        "# Add src to path\n",
        "project_root = Path.cwd().parent\n",
        "if str(project_root) not in sys.path:\n",
        "    sys.path.append(str(project_root))\n",
        "    \n",
        "# Import from src\n",
        "from src.config import ExperimentConfig, load_config\n",
        "from src.decoders.qwen import QwenDecoder\n",
        "from src.encoders.vision import VisionEncoder\n",
        "from src.models.trm_qwen_vlm import QwenVLM, TRMConfig\n",
        "from src.data.dataset_builder import PixmoQADataset\n",
        "from src.data.transforms import get_image_transforms\n",
        "\n",
        "%matplotlib inline\n"
    ]
    # Replace imports cell (index 1 and 3 seem to be imports or path setup, let's look for known markers)
    # The existing notebook has cells with IDs.
    # Cell 1 (index 1) has imports. Cell 2 (index 2) adds path. Cell 3 (index 3) acts as imports part 2.
    # We will coalesce them into one clean setup.
    
    # Let's rebuild the cells list pretty much.
    
    cells = nb['cells']
    
    # 0: Markdown header (Keep)
    
    # 1: Code Imports (Replace)
    cells[1]['source'] = imports_source
    cells[1]['outputs'] = [] # clear outputs
    cells[1]['execution_count'] = None
    
    # 2: Path setup (Remove or Empty, since handled in Imports)
    cells[2]['source'] = ["# Path setup moved to imports cell\n"]
    
    # 3: Second import block (Remove or Empty)
    cells[3]['source'] = ["# Imports merged to first cell\n"]
    
    # 4: Markdown Config (Keep)
    
    # 5: Device and Config Load (Update)
    config_source = [
        "\n",
        "device = \"cuda\" if torch.cuda.is_available() else \"cpu\"\n",
        "print(f\"Using device: {device}\")\n",
        "\n",
        "# Load Config\n",
        "config_path = \"../configs/trm_vlm_qa.yaml\" # Corrected path relative to notebooks\n",
        "config = load_config(config_path)\n",
        "print(f\"Loaded config: {config.name}\")\n"
    ]
    cells[5]['source'] = config_source
    cells[5]['outputs'] = []
    cells[5]['execution_count'] = None

    # 7: Model Load (Update)
    # Need to find index for "2. Load Model" markdown is cell 6. Cell 7 is code.
    model_source = [
        "\n",
        "# 1. Vision Encoder\n",
        "print(\"Loading Vision Encoder...\")\n",
        "vision_encoder = VisionEncoder(\n",
        "    model_name=config.vision_encoder.model_name,\n",
        "    trainable=False\n",
        ").to(device)\n",
        "\n",
        "# 2. Qwen Decoder (LoRA)\n",
        "print(\"Loading Qwen Decoder...\")\n",
        "qwen_decoder = QwenDecoder(\n",
        "    model_name=config.decoder.model_name,\n",
        "    device_map=\"balanced\", # Use balanced for multi-gpu inference\n",
        "    use_lora=True,\n",
        "    lora_r=config.decoder.lora_r,\n",
        "    lora_alpha=config.decoder.lora_alpha,\n",
        "    lora_dropout=config.decoder.lora_dropout\n",
        ")\n",
        "\n",
        "# 3. TRM VLM Wrapper\n",
        "# Using defaults matching training notebook since config might be partial\n",
        "USE_TRM_RECURSION = True\n",
        "NUM_TRM_LAYERS = 4\n",
        "NUM_RECURSION_STEPS = 4\n",
        "\n",
        "trm_config = TRMConfig(\n",
        "    hidden_dim=qwen_decoder.hidden_dim,\n",
        "    num_layers=NUM_TRM_LAYERS,\n",
        "    num_heads=16,\n",
        ")\n",
        "\n",
        "model = QwenVLM(\n",
        "    qwen_decoder=qwen_decoder,\n",
        "    vision_token_dim=vision_encoder.output_dim,\n",
        "    use_trm_recursion=USE_TRM_RECURSION,\n",
        "    trm_config=trm_config,\n",
        "    num_trm_layers=NUM_TRM_LAYERS,\n",
        "    num_recursion_steps=NUM_RECURSION_STEPS\n",
        ").to(device)\n",
        "\n",
        "print(\"Model Initialized.\")\n"
    ]
    cells[7]['source'] = model_source
    cells[7]['outputs'] = []
    cells[7]['execution_count'] = None

    # 9: Load Checkpoint (Update)
    checkpoint_source = [
        "\n",
        "CKPT_ROOT_CANDIDATES = [\n",
        "    Path.cwd() / 'checkpoints',\n",
        "    Path.cwd().parent / 'checkpoints',\n",
        "    Path.cwd() / 'edge_glass_modular/notebooks/checkpoints',\n",
        "    Path.cwd().parent / 'edge_glass_modular/notebooks/checkpoints',\n",
        "]\n",
        "CKPT_ROOT = next((p for p in CKPT_ROOT_CANDIDATES if p.exists()), None)\n",
        "if CKPT_ROOT is None:\n",
        "     # Fallback to creating one for robustness\n",
        "     CKPT_ROOT = Path.cwd() / 'checkpoints'\n",
        "\n",
        "ckpt_dir = CKPT_ROOT / \"qwen_vlm_qa_trm\"\n",
        "best_ckpt_path = ckpt_dir / \"checkpoint_best.pt\"\n",
        "\n",
        "if best_ckpt_path.exists():\n",
        "    print(f\"Loading best checkpoint from {best_ckpt_path}...\")\n",
        "    checkpoint = torch.load(best_ckpt_path, map_location=device)\n",
        "    \n",
        "    try:\n",
        "        model.load_state_dict(checkpoint['model_state_dict'])\n",
        "        print(f\"Successfully loaded checkpoint (Loss: {checkpoint.get('loss', 'N/A'):.4f})\")\n",
        "    except RuntimeError as e:\n",
        "        print(f\"Error loading checkpoint (likely minor mismatch): {e}\")\n",
        "else:\n",
        "    print(f\"Warning: {best_ckpt_path} not found. Using initialized weights (untrained).\")\n",
        "\n",
        "model.eval()\n"
    ]
    cells[9]['source'] = checkpoint_source
    cells[9]['outputs'] = []
    cells[9]['execution_count'] = None

    # 11: Validation Data (Update)
    val_data_source = [
        "\n",
        "# Transforms\n",
        "val_transforms = get_image_transforms(config.dataset.image_size, is_training=False)\n",
        "\n",
        "# Dataset\n",
        "# Assuming we use the val_parquet from config\n",
        "val_dataset = PixmoQADataset(\n",
        "    parquet_path=config.dataset.val_parquet,\n",
        "    tokenizer=qwen_decoder.tokenizer,\n",
        "    image_transforms=val_transforms,\n",
        "    max_question_length=128,\n",
        "    max_answer_length=256,\n",
        ")\n",
        "\n",
        "print(f\"Validation Samples: {len(val_dataset)}\")\n",
        "\n",
        "# Helper for vision encoding\n",
        "def encode_images(images):\n",
        "    # Shape: (B, C, H, W) -> (B, Num_Visual_Tokens, Dim)\n",
        "    # The vision encoder expects images on its device\n",
        "    if not isinstance(images, torch.Tensor):\n",
        "        images = images.to(device)\n",
        "    \n",
        "    with torch.no_grad():\n",
        "        features = vision_encoder(images, return_sequence=True).sequence\n",
        "    return features\n"
    ]
    cells[11]['source'] = val_data_source
    cells[11]['outputs'] = []
    cells[11]['execution_count'] = None
    
    # 13: Loop (Update loop to use proper keys)
    loop_source = [
        "\n",
        "from collections import Counter\n",
        "import string\n",
        "\n",
        "# Metric Helpers\n",
        "def normalize_answer(s: str) -> str:\n",
        "    \"\"\"Normalize answer for evaluation.\"\"\"\n",
        "    s = ''.join(ch for ch in s if ch not in string.punctuation)\n",
        "    s = s.lower().strip()\n",
        "    s = ' '.join([w for w in s.split() if w not in {'a', 'an', 'the'}])\n",
        "    return s\n",
        "\n",
        "def compute_f1(pred: str, target: str) -> float:\n",
        "    \"\"\"Compute token-level F1.\"\"\"\n",
        "    pred_tokens = normalize_answer(pred).split()\n",
        "    target_tokens = normalize_answer(target).split()\n",
        "    if len(pred_tokens) == 0 or len(target_tokens) == 0:\n",
        "        return float(pred_tokens == target_tokens)\n",
        "    common = Counter(pred_tokens) & Counter(target_tokens)\n",
        "    num_common = sum(common.values())\n",
        "    if num_common == 0:\n",
        "        return 0.0\n",
        "    precision = num_common / len(pred_tokens)\n",
        "    recall = num_common / len(target_tokens)\n",
        "    f1 = 2 * precision * recall / (precision + recall)\n",
        "    return f1\n",
        "\n",
        "num_samples = 50 # Limit for quick evaluation\n",
        "indices = np.random.choice(len(val_dataset), num_samples, replace=False)\n",
        "\n",
        "results = []\n",
        "print(f\"Evaluating on {num_samples} random samples...\")\n",
        "\n",
        "for idx in tqdm(indices):\n",
        "    sample = val_dataset[idx]\n",
        "    image = sample['image'].unsqueeze(0).to(device) # 'image' key in PixmoQADataset\n",
        "    question_ids = sample['question_ids'].unsqueeze(0).to(device)\n",
        "    answer_ids_gt = sample['answer_ids'].unsqueeze(0).to(device)\n",
        "    \n",
        "    # Decode helper\n",
        "    tokenizer = qwen_decoder.tokenizer\n",
        "    question_text = tokenizer.decode(sample['question_ids'], skip_special_tokens=True)\n",
        "    ground_truth = tokenizer.decode(sample['answer_ids'], skip_special_tokens=True)\n",
        "    \n",
        "    # Inference\n",
        "    with torch.no_grad():\n",
        "        vision_tokens = encode_images(image)\n",
        "        \n",
        "        # We assume max new tokens \n",
        "        output = model.generate(\n",
        "            vision_tokens=vision_tokens,\n",
        "            question_ids=question_ids,\n",
        "            max_new_tokens=32,\n",
        "            temperature=0.0, # Greedy\n",
        "            return_stats=False\n",
        "        )\n",
        "        \n",
        "        # Handle both tensor and dict return\n",
        "        generated_ids = output['predictions'] if isinstance(output, dict) else output\n",
        "        \n",
        "    prediction = tokenizer.decode(generated_ids[0], skip_special_tokens=True)\n",
        "    \n",
        "    # Metrics\n",
        "    f1 = compute_f1(prediction, ground_truth)\n",
        "    exact_match = (normalize_answer(prediction) == normalize_answer(ground_truth))\n",
        "\n",
        "    results.append({\n",
        "        \"image_tensor\": sample['image'], # Save tensor for vis\n",
        "        \"question\": question_text,\n",
        "        \"ground_truth\": ground_truth,\n",
        "        \"prediction\": prediction,\n",
        "        \"f1\": f1,\n",
        "        \"exact_match\": exact_match,\n",
        "        \"is_correct\": exact_match # Use EM for boolean correct\n",
        "    })\n",
        "\n",
        "print(\"Evaluation Complete.\")\n"
    ]
    cells[13]['source'] = loop_source
    cells[13]['outputs'] = []
    cells[13]['execution_count'] = None

    # 17: Visualizer (Fix image loading)
    vis_source = [
        "\n",
        "def visualize_results(results, num_display=16, cols=4):\n",
        "    rows = (num_display + cols - 1) // cols\n",
        "    fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*5))\n",
        "    axes = axes.flatten()\n",
        "    \n",
        "    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)\n",
        "    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)\n",
        "\n",
        "    for i in range(num_display):\n",
        "        if i >= len(results):\n",
        "            axes[i].axis('off')\n",
        "            continue\n",
        "            \n",
        "        res = results[i]\n",
        "        ax = axes[i]\n",
        "        \n",
        "        # Un-normalize image for display\n",
        "        img_tensor = res['image_tensor'].cpu()\n",
        "        img_tensor = img_tensor * std + mean\n",
        "        img_tensor = torch.clamp(img_tensor, 0, 1)\n",
        "        img = img_tensor.permute(1, 2, 0).numpy()\n",
        "\n",
        "        ax.imshow(img)\n",
        "            \n",
        "        color = 'green' if res['is_correct'] else 'red'\n",
        "        \n",
        "        # Text\n",
        "        q_text = (res['question'][:40] + '..') if len(res['question']) > 40 else res['question']\n",
        "        gt_text = res['ground_truth']\n",
        "        pred_text = res['prediction']\n",
        "        \n",
        "        title = f\"Q: {q_text}\\nGT: {gt_text}\\nPred: {pred_text}\"\n",
        "        \n",
        "        ax.set_title(title, fontsize=9, color=color, fontweight='bold')\n",
        "        ax.axis('off')\n",
        "        \n",
        "    plt.tight_layout()\n",
        "    plt.show()\n",
        "\n",
        "# Show first 16 results\n",
        "visualize_results(results, num_display=16)\n"
    ]
    cells[17]['source'] = vis_source
    cells[17]['outputs'] = []
    cells[17]['execution_count'] = None

    # Write back
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
        
    print("Evaluation Notebook Fixed Successfully.")

if __name__ == "__main__":
    notebook_path = "/home/hice1/vchopra37/scratch/projects/edge_glass/edge_glass_modular/notebooks/04_vlm_qa_evaluation.ipynb"
    fix_notebook(notebook_path)
