
import json
import sys
from pathlib import Path

def force_replace_cell(notebook_path):
    path = Path(notebook_path)
    if not path.exists():
        print(f"Error: {path} not found")
        sys.exit(1)
        
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    modified = False
    
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = "".join(cell.get('source', []))
            if "class QwenVLM(nn.Module):" in source:
                print("Found QwenVLM class definition cell. overwriting...")
                
                # New content: just the import
                new_source = [
                    "# Replaced inline class with imported refactored class\n",
                    "from models.trm_qwen_vlm import QwenVLM\n",
                    "\n",
                    "print(\"✓ QwenVLM class imported from src.models.trm_qwen_vlm\")\n"
                ]
                
                cell['source'] = new_source
                modified = True
                # We stop after first match to avoid replacing multiple if any
                break 
                
    if modified:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)
        print(f"Successfully replaced QwenVLM class in {path}")
    else:
        print("Could not find QwenVLM class definition cell to replace.")

if __name__ == "__main__":
    notebook_path = "/home/hice1/vchopra37/scratch/projects/edge_glass/edge_glass_modular/notebooks/03_trm_vlm_qa_training_FIXED.ipynb"
    force_replace_cell(notebook_path)
