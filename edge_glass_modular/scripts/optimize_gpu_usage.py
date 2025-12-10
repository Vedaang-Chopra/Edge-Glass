
import json
import sys
from pathlib import Path

def optimize_gpu_usage(notebook_path):
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
            
            # 1. Move Aligned Model to CUDA:1
            if "aligned_model = MultimodalAlignmentModel(alignment_config).to(device)" in source:
                new_source = source.replace(
                    "aligned_model = MultimodalAlignmentModel(alignment_config).to(device)",
                    "aligned_model = MultimodalAlignmentModel(alignment_config).to('cuda:1')"
                )
                if new_source != source:
                    cell['source'] = new_source.splitlines(keepends=True)
                    modified = True
                    print("Moved aligned_model to cuda:1")
            
            # 2. Update QwenDecoder device_map to 'balanced'
            if 'device_map="auto",' in source and "qwen_decoder = QwenDecoder" in source:
                new_source = source.replace('device_map="auto",', 'device_map="balanced",')
                if new_source != source:
                    cell['source'] = new_source.splitlines(keepends=True)
                    modified = True
                    print("Updated QwenDecoder device_map to 'balanced'")

            # 3. Update encode_images to handle device placement
            if "def encode_images(images: torch.Tensor)" in source:
                # We want to insert image device movement
                # Pattern: func def -> docstring (optional) -> code
                
                new_lines = []
                lines = cell.get('source', [])
                
                patched = False
                for line in lines:
                    new_lines.append(line)
                    if "def encode_images" in line:
                         # We'll rely on finding the line to insert after
                         pass
                    
                    if "vision_output = aligned_model.vision_encoder(images" in line and not patched:
                         # Insert device move before this line
                         # Find indentation
                         indent = line[:len(line) - len(line.lstrip())]
                         
                         # Pop the line we just added to insert before it
                         new_lines.pop()
                         
                         new_lines.append(f"{indent}# Ensure images are on same device as model\n")
                         new_lines.append(f"{indent}device = next(aligned_model.parameters()).device\n")
                         new_lines.append(f"{indent}images = images.to(device)\n")
                         new_lines.append(line) # Add back the call
                         patched = True
                
                if patched:
                    cell['source'] = new_lines
                    modified = True
                    print("Updated encode_images to handle device placement")

    if modified:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)
        print(f"Updated {path}")
    else:
        print("No changes made (patterns not found?)")

if __name__ == "__main__":
    notebook_path = "/home/hice1/vchopra37/scratch/projects/edge_glass/edge_glass_modular/notebooks/03_trm_vlm_qa_training_FIXED.ipynb"
    optimize_gpu_usage(notebook_path)
