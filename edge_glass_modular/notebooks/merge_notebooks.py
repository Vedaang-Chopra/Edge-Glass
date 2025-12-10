import json
import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell

def read_notebook(path):
    with open(path, 'r', encoding='utf-8') as f:
        return nbformat.read(f, as_version=4)

def write_notebook(nb, path):
    with open(path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

def main():
    nb04_path = 'notebooks/04_alignment_evaluation.ipynb'
    nb05_path = 'notebooks/05_alignment_evaluation.ipynb'
    
    nb04 = read_notebook(nb04_path)
    nb05 = read_notebook(nb05_path)
    
    print(f"Loaded 04 with {len(nb04.cells)} cells")
    print(f"Loaded 05 with {len(nb05.cells)} cells")

    # 1. Inject Imports and Setup
    setup_code = """
# --- Evaluation Tools Setup ---
import sys
import os
sys.path.append(os.path.abspath('../src'))

# Import evaluation tools
from evaluation.benchmark import AlignmentBenchmark, RetrievalMetrics
from evaluation.explainability import ExplainabilityAnalyzer
from utils.visualization import TrainingVisualizer
import torch.nn.functional as F
import numpy as np
import pandas as pd # For table formatting if needed

# Initialize tools
print("Initializing evaluation tools...")
# Ensure we have the device variable
if 'device' not in locals():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Re-wrap model if needed or use existing 'model'
# visualizer will calculate metrics and plots
output_dir = Path("outputs/pixmo_alignment_eval") / "visualizations"
output_dir.mkdir(parents=True, exist_ok=True)

visualizer = TrainingVisualizer(
    save_dir=output_dir,
    style="dark", # or "white"
)

explainability = ExplainabilityAnalyzer(
    model=model,
    device=device,
)
print("Tools initialized.")
"""
    # Insert setup after imports (usually early in notebook, but for appending, we can put it before new viz cells)
    # Better to append to end for now to avoid disrupting existing flow, assuming 'model', 'vision_embs', 'text_embs' exist.
    
    # 2. Bridge Variables
    # 05 uses: all_vision_embs, all_text_embeddings (dictionary by checkpoint)
    # We need to compute metrics for the BEST checkpoint or the last one iterated.
    # Let's assume the user wants to evaluate the specific loaded checkpoint 'ckpts[-1]' or the one currently in memory.
    # Looking at 05, it iterates checkpoints. We should probably pick the best performing one or the last one.
    # For now, let's assume we use the variables from the LAST loop iteration in 05, 
    # but 05 seems to be collecting them in 'results'.
    
    # Let's inspect 05 content again to be sure where `vision_embs` comes from. 
    # Actually, 05 computes embeddings inside a loop. We might need to grab them from the 'results' dictionary.
    
    bridge_code = """
# --- Prepare Data for Visualizations ---
# Extract embeddings from the results of the best/last checkpoint
# Assuming 'results' dict exists from previous cells in 05
# specific structure of 'results' needs to be confirmed, usually results[ckpt_name]['vision_embs']

metric_data = None
best_r1 = -1
best_ckpt_name = ""

# Find best checkpoint based on R@1
if 'results' in locals() and results:
    print(f"Found {len(results)} checkpoints in results.")
    for name, data in results.items():
        # Calculate R@1 if not pre-calculated, or extract it
        # Assuming data contains 'vision_embs' and 'text_embs'
        if 'vision_embs' in data and 'text_embs' in data:
            v_emb = data['vision_embs']
            t_emb = data['text_embs']
            
            # Quick R@1 calc for selection
            # Normalize
            v_norm = F.normalize(v_emb, p=2, dim=-1)
            t_norm = F.normalize(t_emb, p=2, dim=-1)
            sim = v_norm @ t_norm.T
            # i2t
            i2t_inds = torch.argsort(sim, dim=1, descending=True)[:, :1]
            ground_truth = torch.arange(len(sim), device=sim.device).view(-1, 1)
            correct = (i2t_inds == ground_truth).sum().item()
            r1 = correct / len(sim)
            
            if r1 > best_r1:
                best_r1 = r1
                best_ckpt_name = name
                metric_data = data

    print(f"Selected best checkpoint for visualization: {best_ckpt_name} (R@1: {best_r1:.1%})")
    
    # Set global variables for 04 cells to use
    vision_embs = metric_data['vision_embs']
    text_embs = metric_data['text_embs']
    
    # Ensure they are tensors on CPU for some plot functions if needed, or keep as is
    if not isinstance(vision_embs, torch.Tensor):
        vision_embs = torch.tensor(vision_embs)
    if not isinstance(text_embs, torch.Tensor):
        text_embs = torch.tensor(text_embs)
        
    vision_embs = vision_embs.to(device)
    text_embs = text_embs.to(device)

else:
    print("WARNING: 'results' variable not found or empty. Using 'vision_embs' and 'text_embs' if they exist in global scope.")
    # Fallback if standard 05 variables are lingering
    if 'embeddings_vision' in locals(): vision_embs = embeddings_vision
    if 'embeddings_text' in locals(): text_embs = embeddings_text

# --- Metric Computation for Custom Table ---
# We need rigorous metrics now
print("Computing rigorous metrics for report...")
# Compute similarity matrix
v_norm_custom = F.normalize(vision_embs, p=2, dim=-1)
t_norm_custom = F.normalize(text_embs, p=2, dim=-1)
sim_matrix = v_norm_custom @ t_norm_custom.T

# Helper to calc R@K
def calc_recall_at_k(sim_mat, k_vals=[1, 5, 10], i2t=True):
    # sim_mat: (N_images, N_texts) if i2t
    # if t2i, we transpose: (N_texts, N_images)
    
    if not i2t:
        sim_mat = sim_mat.T
        
    num_samples = sim_mat.shape[0]
    # Do in chunks if OOM, but 4096 samples usually fit
    # Top-k
    max_k = max(k_vals)
    topk_indices = torch.topk(sim_mat, k=max_k, dim=1).indices # (N, max_k)
    gt = torch.arange(num_samples, device=sim_mat.device).view(-1, 1)
    
    recalls = {}
    for k in k_vals:
        # check if gt is in top k
        hits = (topk_indices[:, :k] == gt).any(dim=1).float().mean().item()
        recalls[k] = hits * 100 # percentage
    return recalls

i2t_recalls = calc_recall_at_k(sim_matrix, k_vals=[1, 5, 10], i2t=True)
t2i_recalls = calc_recall_at_k(sim_matrix, k_vals=[1, 5, 10], i2t=False)

print(f"I2T: {i2t_recalls}")
print(f"T2I: {t2i_recalls}")

# --- Generate Custom Table ---
print("\\n" + "="*80)
print("PIXMO VISION–TEXT RETRIEVAL (VALIDATION)")
print("="*80)
print(f"Direction {'R@1 (%)':>10} {'R@5 (%)':>10} {'R@10 (%)':>10}")
print("-" * 45)
print(f"I2T (4096-d) {i2t_recalls[1]:10.1f} {i2t_recalls[5]:10.1f} {i2t_recalls[10]:10.1f}")
print(f"T2I (4096-d) {t2i_recalls[1]:10.1f} {t2i_recalls[5]:10.1f} {t2i_recalls[10]:10.1f}")
print("="*80 + "\\n")

# Calculate metrics object for Visualizer
# Visualizer expects 'RetrievalMetrics' object which has .ranks etc.
# We will manually construct a metrics-like object or calculate ranks for plotting
def get_ranks(sim_mat): # i2t
    # Returns rank of ground truth for each query
    # simple loop to save memory or broadcast
    n = sim_mat.shape[0]
    ranks = []
    # rank = number of entries with score > gt_score
    # diagonal is gt
    d = torch.diag(sim_mat) # (N,)
    # compare each row to its diagonal
    # (N, N) > (N, 1) broadcast
    # count how many > gt
    # +1 for 1-based rank
    for i in range(n):
        r = (sim_mat[i] > d[i]).sum().item() + 1
        ranks.append(r)
    return np.array(ranks)

print("Calculating ranks for plots...")
i2t_ranks = get_ranks(sim_matrix)
t2i_ranks = get_ranks(sim_matrix.T)

# Mock object for Visualizer if it strictly checks types, or just pass dict
class MockMetrics:
    def __init__(self, ranks, recalls):
        self.ranks = ranks
        self.r_at_1 = recalls[1]
        self.r_at_5 = recalls[5]
        self.r_at_10 = recalls[10]

i2t_metrics_obj = MockMetrics(i2t_ranks, i2t_recalls)
t2i_metrics_obj = MockMetrics(t2i_ranks, t2i_recalls)

"""

    nb05.cells.append(new_code_cell(setup_code))
    nb05.cells.append(new_code_cell(bridge_code))

    # 3. Copy Visualization Cells from 04
    # We want cells 11 to 15 (indices roughly 10-14 depending on implementation)
    # Based on file inspection:
    # Cell 11: Rank Analysis
    # Cell 12: Similarity Analysis
    # Cell 13: MRL Curves
    # Cell 14: Embedding Space
    # Cell 15: Explainability
    # Cell 16: Retrieval Examples
    
    # Let's search for headers to be robust
    cells_to_copy = []
    capture = False
    
    # We grab everything from "## 11. Visualization: Rank Analysis" onwards
    # except the "Save Results" and "Finish" cells which might duplicate 05 logic?
    # 05 likely saves results too. Let's just grab vis cells.
    
    for cell in nb04.cells:
        if cell.cell_type == 'markdown':
            if "Visualization: Rank Analysis" in cell.source:
                capture = True
            if "## 17. Save Results" in cell.source:
                capture = False
        
        if capture:
            # Modify code cells to use our i2t_metrics_obj
            if cell.cell_type == 'code':
                src = cell.source
                # Replace i2t_metrics with i2t_metrics_obj
                src = src.replace("i2t_metrics", "i2t_metrics_obj")
                src = src.replace("t2i_metrics", "t2i_metrics_obj")
                
                # Replace results['embeddings'] access as we set globals
                src = src.replace("vision_embs = results['embeddings']['vision']", "# vision_embs already set")
                src = src.replace("text_embs = results['embeddings']['text']", "# text_embs already set")
                
                # Replace config access if needed
                # 04 uses 'eval_config' which 05 might not have or is named 'config'
                # We should patch these. 05: config object usually exists.
                # Let's check 05 config var name. Usually 'config' or 'cfg'.
                # We'll wrap in try/except or safe get
                if "eval_config" in src:
                    src = src.replace("eval_config", "config") 
                
                cell.source = src
                
            nb05.cells.append(cell)

    write_notebook(nb05, nb05_path)
    print(f"Merged notebook saved to {nb05_path}")

if __name__ == "__main__":
    main()
