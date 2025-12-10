
import json
import sys
from pathlib import Path

def fix_val_transforms(notebook_path):
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
            
            # Find the dataset cell
            if "train_dataset = PixmoQADataset" in source and "val_transforms =" not in source:
                print("Found dataset cell. Injecting transforms definition...")
                
                new_source = []
                # Inject definition at the top of the cell (after headers)
                
                # Check if we have the header
                lines = cell.get('source', [])
                
                inserted = False
                for line in lines:
                    new_source.append(line)
                    # Insert after "INITIALIZING DATASETS" block or at start if not found
                    if "print(\"INITIALIZING DATASETS\")" in line and not inserted:
                        new_source.append("print(\"=\"*60)\n") # The existing code has this, but let's just append after the print block finishes?
                        # Actually, let's just insert before 'train_dataset ='
                        pass
                        
                # Let's rebuild source list more carefully
                final_source = []
                definitions_added = False
                
                for line in lines:
                    if "train_dataset = PixmoQADataset" in line and not definitions_added:
                        final_source.append("# Define transforms globally\n")
                        final_source.append("train_transforms = get_image_transforms(config.dataset.image_size, is_training=True)\n")
                        final_source.append("val_transforms = get_image_transforms(config.dataset.image_size, is_training=False)\n")
                        final_source.append("\n")
                        definitions_added = True
                        
                    # Fix usage in PixmoQADataset calls to use the variables (optional but cleaner)
                    # But replacing the complex call with variable is hard via simple string replace if multiline.
                    # So we just define them. The existing code calls get_image_transforms(...) inline.
                    # That is fine. We just need 'val_transforms' to exist for the probe cell.
                    
                    final_source.append(line)
                
                if definitions_added:
                    cell['source'] = final_source
                    modified = True
                    print("Injected val_transforms definition.")
                else:
                    # Fallback: Just prepend it to the cell
                    print("Could not find insertion point 'train_dataset =', prepending...")
                    # This might fail if config is not defined yet? 
                    # No, this cell uses config.dataset
                    final_source = [
                        "train_transforms = get_image_transforms(config.dataset.image_size, is_training=True)\n",
                        "val_transforms = get_image_transforms(config.dataset.image_size, is_training=False)\n"
                    ] + lines
                    cell['source'] = final_source
                    modified = True

    if modified:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)
        print(f"Updated {path}")
    else:
        print("Could not find dataset cell to patch.")

if __name__ == "__main__":
    notebook_path = "/home/hice1/vchopra37/scratch/projects/edge_glass/edge_glass_modular/notebooks/03_trm_vlm_qa_training_FIXED.ipynb"
    fix_val_transforms(notebook_path)
