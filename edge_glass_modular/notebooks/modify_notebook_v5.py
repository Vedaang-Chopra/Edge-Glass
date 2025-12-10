
import json
from pathlib import Path

notebook_path = Path("/home/hice1/vchopra37/scratch/projects/edge_glass/edge_glass_modular/notebooks/05_vlm_qualitative_inference.ipynb")

with open(notebook_path, "r") as f:
    nb = json.load(f)

# Locate cell with device definition
for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        source_str = "".join(cell["source"])
        if "device =" in source_str and "cuda:1" in source_str:
            print("Found cell with hardcoded cuda:1")
            new_source = []
            for line in cell["source"]:
                if "device =" in line and "cuda:1" in line:
                    # Replace with generic cuda
                    new_line = line.replace("cuda:1", "cuda")
                    new_source.append(new_line)
                    print(f"Replaced: {line.strip()} -> {new_line.strip()}")
                else:
                    new_source.append(line)
            cell["source"] = new_source

with open(notebook_path, "w") as f:
    json.dump(nb, f, indent=4)
print("Successfully modified notebook with v5 fixes (Fixed hardcoded device index).")
