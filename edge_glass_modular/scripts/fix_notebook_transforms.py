
import json
import sys
from pathlib import Path

def fix_transforms(notebook_path):
    path = Path(notebook_path)
    if not path.exists():
        print(f"Error: {path} not found")
        sys.exit(1)
        
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    modified = False
    count = 0
    
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            new_source = []
            cell_modified = False
            for line in cell.get('source', []):
                if "get_image_transforms" in line and "is_train=" in line:
                    start = line
                    line = line.replace("is_train=", "is_training=")
                    if line != start:
                        cell_modified = True
                        count += 1
                new_source.append(line)
            
            if cell_modified:
                cell['source'] = new_source
                modified = True
                
    if modified:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)
        print(f"Fixed {count} occurrences of is_train -> is_training in {path}")
    else:
        print("No occurrences found to fix.")

if __name__ == "__main__":
    notebook_path = "/home/hice1/vchopra37/scratch/projects/edge_glass/edge_glass_modular/notebooks/03_trm_vlm_qa_training_FIXED.ipynb"
    fix_transforms(notebook_path)
