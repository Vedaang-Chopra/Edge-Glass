
import json
from pathlib import Path

notebook_path = Path("/home/hice1/vchopra37/scratch/projects/edge_glass/edge_glass_modular/notebooks/05_vlm_qualitative_inference.ipynb")

with open(notebook_path, "r") as f:
    nb = json.load(f)

# Iterate through cells to find the configuration cell
target_cell_source = None
target_cell_idx = -1

start_marker = "config_path = \"../configs/trm_vlm_qa_qwen2.5-3b_regularized.yaml\""

for idx, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code":
        source = cell["source"]
        # source is a list of strings
        # Join them to check easier, or check lines
        source_text = "".join(source)
        if start_marker in source_text:
            target_cell_idx = idx
            break

if target_cell_idx != -1:
    print(f"Found target cell at index {target_cell_idx}")
    # Construct new source logic
    # We want to insert the auto-detection logic before alignment_checkpoint
    
    # Let's locate the checkpoint_dir line
    old_source = nb["cells"][target_cell_idx]["source"]
    new_source = []
    
    found_target = False
    for line in old_source:
        if "checkpoint_dir =" in line and "checkpoint_best" in line and not found_target:
            # Replace this line with our block
            new_source.append("# Checkpoints\n")
            new_source.append("# Auto-detect latest checkpoint\n")
            new_source.append("import glob\n")
            new_source.append("\n")
            new_source.append("output_dir = Path(\"../outputs/trm_vlm_qa_3b_reg\")\n")
            new_source.append("epoch_checkpoints = sorted(output_dir.glob(\"checkpoint-epoch-*\"), key=lambda p: int(str(p).split('-')[-1]))\n")
            new_source.append("\n")
            new_source.append("if epoch_checkpoints:\n")
            new_source.append("    checkpoint_dir = str(epoch_checkpoints[-1])\n")
            new_source.append("    print(f\"Auto-detected latest checkpoint: {checkpoint_dir}\")\n")
            new_source.append("else:\n")
            new_source.append("    # Fallback\n")
            new_source.append("    checkpoint_dir = str(output_dir / \"checkpoint_best\")\n")
            new_source.append("    print(f\"No epoch checkpoints found. Using: {checkpoint_dir}\")\n")
            
            found_target = True
        else:
            new_source.append(line)
            
    if found_target:
        nb["cells"][target_cell_idx]["source"] = new_source
        
        with open(notebook_path, "w") as f:
            json.dump(nb, f, indent=4)
        print("Successfully modified notebook.")
    else:
        print("Could not find the target line to replace.")

else:
    print("Could not find the target cell.")
