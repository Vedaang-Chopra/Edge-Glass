
import json
from pathlib import Path

notebook_path = Path("/home/hice1/vchopra37/scratch/projects/edge_glass/edge_glass_modular/notebooks/05_vlm_qualitative_inference.ipynb")

with open(notebook_path, "r") as f:
    nb = json.load(f)

# 1. Inject CUDA_VISIBLE_DEVICES env var before torch import
for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        source_str = "".join(cell["source"])
        if "import torch" in source_str:
            new_source = []
            env_set = False
            for line in cell["source"]:
                if "import torch" in line and not env_set:
                    new_source.append("import os\n")
                    new_source.append("os.environ['CUDA_VISIBLE_DEVICES'] = '0' # Force single GPU to avoid mismatches\n")
                    new_source.append(line)
                    env_set = True
                else:
                    new_source.append(line)
            cell["source"] = new_source
            print("Injected CUDA_VISIBLE_DEVICES=0")
            break

# 2. Uncomment metrics import
found_import = False
for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        new_source = []
        for line in cell["source"]:
            if "from evaluation.simple_metrics import" in line and "#" in line:
                new_source.append(line.replace("#", "").strip() + "\n")
                new_source.insert(0, "import sys\nsys.path.append('../src')\n")
                found_import = True
            else:
                new_source.append(line)
        cell["source"] = new_source
        
if not found_import:
    # If not found (maybe overwritten), inject it in the first code cell
    print("Metrics import not found/uncommented, injecting...")
    first_cell = next(c for c in nb["cells"] if c["cell_type"] == "code")
    found_sys = any("sys.path.append" in line for line in first_cell["source"])
    if not found_sys:
         first_cell["source"].insert(0, "import sys\nsys.path.append('../src')\n")
    first_cell["source"].append("from evaluation.simple_metrics import compute_bleu, compute_rouge_l, compute_perplexity\n")

with open(notebook_path, "w") as f:
    json.dump(nb, f, indent=4)
print("Successfully modified notebook with v4 fixes (Single GPU Force + Metrics).")
