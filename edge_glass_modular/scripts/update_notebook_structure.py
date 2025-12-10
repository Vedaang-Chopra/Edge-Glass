
import json
import sys
from pathlib import Path

def update_notebook(notebook_path):
    path = Path(notebook_path)
    if not path.exists():
        print(f"Error: {path} not found")
        sys.exit(1)
        
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    modified = False
    
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = cell.get('source', [])
            # Check if this is the cell defining QwenVLM
            # Join loosely to check content
            full_source = "".join(source)
            if "class QwenVLM(nn.Module):" in full_source and "from src.models.trm_qwen_vlm" not in full_source:
                print("Found QwenVLM class definition cell. Modifying...")
                
                # We want to wrap the class in ''' ''' and add import
                # Find the start line index
                start_idx = -1
                for i, line in enumerate(source):
                    if "class QwenVLM(nn.Module):" in line:
                        start_idx = i
                        break
                
                if start_idx != -1:
                    # Insert import and open quote before class def
                    new_header = [
                        "from src.models.trm_qwen_vlm import QwenVLM\n",
                        "'''\n"
                    ]
                    
                    # Add closing quote at the end
                    new_footer = [
                        "\n",
                        "'''\n",
                        "print(\"✓ QwenVLM class imported from src.models.trm_qwen_vlm\")"
                    ]
                    
                    # We insert the header at start_idx
                    # But we should keep the indentation of the class? 
                    # The class def line has no indent usually in global scope.
                    # source list is list of strings including \n
                    
                    # Create new source list
                    # Keep lines before class def (if any, e.g. empty lines)
                    new_source = source[:start_idx] + new_header + source[start_idx:] + new_footer
                    
                    cell['source'] = new_source
                    modified = True
                else:
                    print("Could not find exact class definition line.")
            elif "class QwenVLM(nn.Module):" in full_source and "from src.models.trm_qwen_vlm" in full_source:
                print("Notebook already mapped to src.")
                    
    if modified:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1) # notebook usually has indent=1 or space
        print(f"Successfully updated {path}")
    else:
        print("No changes made.")

if __name__ == "__main__":
    notebook_path = "/home/hice1/vchopra37/scratch/projects/edge_glass/edge_glass_modular/notebooks/03_trm_vlm_qa_training_FIXED.ipynb"
    update_notebook(notebook_path)
