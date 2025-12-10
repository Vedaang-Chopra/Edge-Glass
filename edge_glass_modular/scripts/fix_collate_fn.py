
import json
import sys
from pathlib import Path

def fix_collate_fn(notebook_path):
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
            
            # Find the collate_fn definition
            if "def collate_fn(batch):" in source and "'answer_mask': a_mask," in source:
                print("Found collate_fn cell. Injecting 'answers' key...")
                
                new_source = []
                lines = cell.get('source', [])
                
                inserted = False
                for line in lines:
                    new_source.append(line)
                    if "'answer_mask': a_mask," in line and not inserted:
                        # Insert answers list
                        new_source.append("        'answers': [b['answer'] for b in batch],\n")
                        inserted = True
                
                if inserted:
                    cell['source'] = new_source
                    modified = True
                    print("Injected 'answers' key into collate_fn.")
    
    if modified:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)
        print(f"Updated {path}")
    else:
        print("Could not find collate_fn to patch.")

if __name__ == "__main__":
    notebook_path = "/home/hice1/vchopra37/scratch/projects/edge_glass/edge_glass_modular/notebooks/03_trm_vlm_qa_training_FIXED.ipynb"
    fix_collate_fn(notebook_path)
