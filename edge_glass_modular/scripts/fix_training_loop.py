
import json
import sys
from pathlib import Path

def fix_training_loop(notebook_path):
    path = Path(notebook_path)
    if not path.exists():
        print(f"Error: {path} not found")
        sys.exit(1)
        
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    modified = False
    
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source = "".join(cell.get('source', []))
            
            # Explicitly target Cell 32 or search fallback
            is_target_cell = (i == 32)
            
            if is_target_cell or ("outputs = model(" in source and "images=images" in source and "loss.backward()" in source):
                print(f"Found training loop cell at index {i}")
                
                # We need to insert the encoding step before model call
                # And change model call to use vision_tokens=vision_tokens
                
                new_source = []
                lines = cell.get('source', [])
                
                encoding_inserted = False
                
                for line in lines:
                    # Check for images extraction
                    # It might be: images = batch['image'].to(device)
                    # OR: images = batch['images'].to(device)
                    # The error log showed: images=images
                    
                    # We look for where 'images' var is defined or used before model
                    if "images =" in line and ".to(device)" in line:
                         # Append the line first
                        new_source.append(line)
                        if not encoding_inserted:
                            # Add encoding immediately after
                            indent = line[:len(line) - len(line.lstrip())]
                            new_source.append(f"{indent}with torch.no_grad():\n")
                            new_source.append(f"{indent}    vision_tokens = encode_images(images)\n")
                            encoding_inserted = True
                        continue
                        
                    # Update model call argument
                    if "images=images" in line:
                         replaced = line.replace("images=images", "vision_tokens=vision_tokens")
                         new_source.append(replaced)
                         continue
                        
                    new_source.append(line)
                
                if encoding_inserted:
                    cell['source'] = new_source
                    modified = True
                    print(f"Fixed training loop at index {i} to include encode_images()")
    
    if modified:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)
        print(f"Updated {path}")
    else:
        print("Could not find training loop to fix, or it was ambiguous.")
        # Fallback: Print what it found to help debug
        for i, cell in enumerate(nb['cells']):
             if "outputs = model(" in "".join(cell.get('source', [])):
                 print(f"Cell {i} contains model call but wasn't patched.")

if __name__ == "__main__":
    notebook_path = "/home/hice1/vchopra37/scratch/projects/edge_glass/edge_glass_modular/notebooks/03_trm_vlm_qa_training_FIXED.ipynb"
    fix_training_loop(notebook_path)
