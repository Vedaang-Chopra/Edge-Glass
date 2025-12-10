
import json
from pathlib import Path

def create_alignment_eval_notebook():
    notebook_path = Path("/home/hice1/vchopra37/scratch/projects/edge_glass/edge_glass_modular/notebooks/05_alignment_evaluation.ipynb")
    
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Pixmo Vision-Text Alignment Evaluation\n",
                "\n",
                "This notebook evaluates the trained Vision-Text Alignment model using retrieval metrics (Recall@K) and qualitative visualization.\n",
                "\n",
                "**Goal**: Measure how well the vision encoder and text encoder align in the shared embedding space."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "import sys\n",
                "from pathlib import Path\n",
                "import torch\n",
                "import numpy as np\n",
                "import pandas as pd\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "from PIL import Image\n",
                "from tqdm.auto import tqdm\n",
                "from torch.utils.data import DataLoader\n",
                "\n",
                "# Add src to path\n",
                "current_dir = Path.cwd()\n",
                "if str(current_dir).endswith(\"notebooks\"):\n",
                "    root_dir = current_dir.parent\n",
                "    os.chdir(root_dir)\n",
                "    sys.path.insert(0, str(root_dir))\n",
                "else:\n",
                "    root_dir = current_dir\n",
                "\n",
                "print(f\"Working directory: {Path.cwd()}\")\n",
                "\n",

                "from src.config import load_config\n",
                "from src.models.alignment import MultimodalAlignmentModel\n",
                "from src.data.dataset_builder import build_image_datasets_from_parquet\n",
                "from src.data.transforms import get_image_transforms\n",
                "\n",
                "# Setup Device\n",
                "device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n",
                "print(f\"Using device: {device}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Load Configuration and Model"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Load config\n",
                "config_path = root_dir / \"configs/alignment.yaml\"\n",
                "print(f\"Loading config from {config_path}...\")\n",
                "config = load_config(str(config_path))\n",
                "\n",
                "# Initialize Model\n",
                "print(\"Initializing model...\")\n",
                "model = MultimodalAlignmentModel(config)\n",
                "model.to(device)\n",
                "model.eval()\n",
                "print(\"Model initialized.\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Load Checkpoint\n",
                "checkpoint_path = root_dir / \"outputs/pixmo_alignment/checkpoint_best.pt\"\n",
                "if not checkpoint_path.exists():\n",
                "    # Fallback search\n",
                "    candidates = list(root_dir.glob(\"outputs/*/checkpoint_best.pt\")) + list(root_dir.glob(\"checkpoints/*/checkpoint_best.pt\"))\n",
                "    if candidates:\n",
                "        checkpoint_path = candidates[0]\n",
                "    else:\n",
                "        raise FileNotFoundError(\"Could not find checkpoint_best.pt\")\n",
                "\n",
                "print(f\"Loading checkpoint from {checkpoint_path}...\")\n",
                "checkpoint = torch.load(checkpoint_path, map_location=device)\n",
                "model.load_state_dict(checkpoint['model_state_dict'])\n",
                "print(f\"Checkpoint loaded. Best Val Loss: {checkpoint.get('best_val_loss', 'N/A')}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Load Validation Data"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Load Datasets\n",
                "print(\"Loading datasets...\")\n",
                "train_dataset, val_dataset = build_image_datasets_from_parquet(\n",
                "    root_dir,\n",
                "    config,\n",
                "    transform=get_image_transforms(config.model.image_size, is_train=False)\n",
                ")\n",
                "\n",
                "print(f\"Validation samples: {len(val_dataset)}\")\n",
                "\n",
                "# Create DataLoader\n",
                "val_loader = DataLoader(\n",
                "    val_dataset,\n",
                "    batch_size=64, # Larger batch size for inference\n",
                "    shuffle=False,\n",
                "    num_workers=4,\n",
                "    pin_memory=True\n",
                ")\n",
                "print(f\"Validation batches: {len(val_loader)}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Compute Embeddings"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "print(\"Computing embeddings for validation set...\")\n",
                "all_vision_embs = []\n",
                "all_text_embs = []\n",
                "all_captions = []\n",
                "\n",
                "with torch.no_grad():\n",
                "    for batch in tqdm(val_loader, desc=\"Extracting embeddings\"):\n",
                "        images = batch['image'].to(device)\n",
                "        texts = batch['text']\n",
                "        \n",
                "        # Forward pass\n",
                "        # Use return_embeddings=True to get pooled embeddings\n",
                "        outputs = model(images=images, texts=texts, return_embeddings=True)\n",
                "        \n",
                "        # Normalize embeddings\n",
                "        v_emb = outputs.vision_emb / outputs.vision_emb.norm(dim=-1, keepdim=True)\n",
                "        t_emb = outputs.text_emb / outputs.text_emb.norm(dim=-1, keepdim=True)\n",
                "        \n",
                "        all_vision_embs.append(v_emb.cpu())\n",
                "        all_text_embs.append(t_emb.cpu())\n",
                "        all_captions.extend(texts)\n",
                "\n",
                "# Concatenate\n",
                "vision_embs = torch.cat(all_vision_embs, dim=0)\n",
                "text_embs = torch.cat(all_text_embs, dim=0)\n",
                "\n",
                "print(f\"Vision Embeddings Shape: {vision_embs.shape}\")\n",
                "print(f\"Text Embeddings Shape: {text_embs.shape}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Quantitative Evaluation (Recall@K)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "def compute_recall_metrics(query_embs, gallery_embs, ks=[1, 5, 10]):\n",
                "    \"\"\"\n",
                "    Compute Recall@K for retrieval.\n",
                "    Assumes 1-to-1 mapping where index i in query corresponds to index i in gallery.\n",
                "    \"\"\"\n",
                "    # Compute similarity matrix (Q x G)\n",
                "    sim_matrix = torch.matmul(query_embs, gallery_embs.T)\n",
                "    \n",
                "    num_samples = sim_matrix.shape[0]\n",
                "    metrics = {}\n",
                "    \n",
                "    # Get top-k indices\n",
                "    max_k = max(ks)\n",
                "    _, topk_indices = sim_matrix.topk(max_k, dim=1)\n",
                "    \n",
                "    # Ground truth is simply arange(num_samples)\n",
                "    ground_truth = torch.arange(num_samples).unsqueeze(1).expand(-1, max_k)\n",
                "    \n",
                "    matches = (topk_indices.cpu() == ground_truth)\n",
                "    \n",
                "    for k in ks:\n",
                "        # Check if ground truth is in top K\n",
                "        hits = matches[:, :k].any(dim=1).float().sum()\n",
                "        recall = (hits / num_samples).item() * 100\n",
                "        metrics[f\"R@{k}\"] = recall\n",
                "        \n",
                "    return metrics\n",
                "\n",
                "print(\"Computing Image-to-Text Retrieval Metrics...\")\n",
                "i2t_metrics = compute_recall_metrics(vision_embs, text_embs)\n",
                "for k, v in i2t_metrics.items():\n",
                "    print(f\"Image-to-Text {k}: {v:.2f}%\")\n",
                "\n",
                "print(\"\\nComputing Text-to-Image Retrieval Metrics...\")\n",
                "t2i_metrics = compute_recall_metrics(text_embs, vision_embs)\n",
                "for k, v in t2i_metrics.items():\n",
                "    print(f\"Text-to-Image {k}: {v:.2f}%\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. Qualitative Visualization"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "def show_retrieval_example(idx, top_k=3):\n",
                "    # Image Query\n",
                "    query_img_emb = vision_embs[idx]\n",
                "    scores = torch.matmul(text_embs, query_img_emb)\n",
                "    top_scores, top_indices = scores.topk(top_k)\n",
                "    \n",
                "    print(f\"\\n--- Example {idx} ---\")\n",
                "    print(\"Query Image Caption (GT):\", all_captions[idx])\n",
                "    \n",
                "    # Display image\n",
                "    try:\n",
                "        # Assuming we can get the image path from dataset internal structure or logic\n",
                "        # If not accessible easily, we might just display index/text\n",
                "        if hasattr(val_dataset, 'dataset') and hasattr(val_dataset.dataset, 'data'):\n",
                "             img_path = val_dataset.dataset.data.iloc[idx]['image_path']\n",
                "             if not Path(img_path).is_absolute():\n",
                "                 img_path = root_dir / img_path\n",
                "             display(Image.open(img_path).resize((200, 200)))\n",
                "    except Exception as e:\n",
                "        print(f\"Could not load image: {e}\")\n",
                "\n",
                "    print(\"Retrieved Captions:\")\n",
                "    for i, res_idx in enumerate(top_indices):\n",
                "        res_idx = res_idx.item()\n",
                "        score = top_scores[i].item()\n",
                "        is_gt = (res_idx == idx)\n",
                "        marker = \"[GT]\" if is_gt else \"\"\n",
                "        print(f\"{i+1}. [{score:.4f}] {all_captions[res_idx]} {marker}\")\n",
                "\n",
                "# Show a few random examples\n",
                "indices = np.random.choice(len(all_captions), 3, replace=False)\n",
                "for idx in indices:\n",
                "    show_retrieval_example(idx)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 6. Explainability Plots"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from sklearn.manifold import TSNE\n",
                "\n",
                "# --- t-SNE Visualization ---\n",
                "print(\"Generating t-SNE plot...\")\n",
                "num_points = 500  # Subsample for speed/clarity\n",
                "indices = np.random.choice(len(vision_embs), min(num_points, len(vision_embs)), replace=False)\n",
                "\n",
                "v_sample = vision_embs[indices].numpy()\n",
                "t_sample = text_embs[indices].numpy()\n",
                "\n",
                "# Combine to joint space\n",
                "combined_embs = np.concatenate([v_sample, t_sample], axis=0)\n",
                "tsne = TSNE(n_components=2, random_state=42, perplexity=30)\n",
                "embeddings_2d = tsne.fit_transform(combined_embs)\n",
                "\n",
                "# Plot\n",
                "plt.figure(figsize=(10, 8))\n",
                "plt.scatter(embeddings_2d[:len(v_sample), 0], embeddings_2d[:len(v_sample), 1], \n",
                "            c='blue', alpha=0.6, label='Vision Embeddings', s=20)\n",
                "plt.scatter(embeddings_2d[len(v_sample):, 0], embeddings_2d[len(v_sample):, 1], \n",
                "            c='red', alpha=0.6, label='Text Embeddings', s=20)\n",
                "\n",
                "# Draw lines for ground truth pairs\n",
                "for i in range(len(v_sample)):\n",
                "    v_pt = embeddings_2d[i]\n",
                "    t_pt = embeddings_2d[len(v_sample) + i]\n",
                "    plt.plot([v_pt[0], t_pt[0]], [v_pt[1], t_pt[1]], 'k-', alpha=0.1)\n",
                "\n",
                "plt.title(\"t-SNE of Vision and Text Embeddings (Paired)\")\n",
                "plt.legend()\n",
                "plt.show()\n",
                "\n",
                "# --- Similarity Distribution ---\n",
                "print(\"Generating Similarity Distribution...\")\n",
                "pos_sims = torch.sum(vision_embs * text_embs, dim=1).cpu().numpy()\n",
                "perm = torch.randperm(len(text_embs))\n",
                "neg_sims = torch.sum(vision_embs * text_embs[perm], dim=1).cpu().numpy()\n",
                "\n",
                "plt.figure(figsize=(10, 5))\n",
                "sns.kdeplot(pos_sims, fill=True, label='Positive Pairs')\n",
                "sns.kdeplot(neg_sims, fill=True, label='Negative Pairs')\n",
                "plt.title(\"Cosine Similarity Distribution\")\n",
                "plt.xlabel(\"Cosine Similarity\")\n",
                "plt.legend()\n",
                "plt.show()"
            ]
        },
         {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# --- Similarity Matrix Heatmap ---\n",
                "print(\"Generating Similarity Heatmap...\")\n",
                "# Take a small batch\n",
                "n_ht = 20\n",
                "ht_indices = np.arange(n_ht)\n",
                "heatmap_v = vision_embs[ht_indices]\n",
                "heatmap_t = text_embs[ht_indices]\n",
                "\n",
                "sim_matrix_batch = torch.matmul(heatmap_v, heatmap_t.T).cpu().numpy()\n",
                "\n",
                "plt.figure(figsize=(10, 8))\n",
                "sns.heatmap(sim_matrix_batch, cmap=\"viridis\", annot=False)\n",
                "\n",
                "plt.xlabel(\"Text Index\")\n",
                "plt.ylabel(\"Image Index\")\n",
                "plt.title(f\"Similarity Matrix (First {n_ht} samples)\\nDiagonal should be high for good alignment\")\n",
                "plt.show()"
            ]
        }
    ]
    
    notebook_content = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.8.10"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    
    with open(notebook_path, 'w') as f:
        json.dump(notebook_content, f, indent=2)
    
    print(f"Notebook created at {notebook_path}")

if __name__ == "__main__":
    create_alignment_eval_notebook()
