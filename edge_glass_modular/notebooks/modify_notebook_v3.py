
import json
from pathlib import Path

notebook_path = Path("/home/hice1/vchopra37/scratch/projects/edge_glass/edge_glass_modular/notebooks/05_vlm_qualitative_inference.ipynb")

with open(notebook_path, "r") as f:
    nb = json.load(f)

# Locate cell with `qwen_decoder = QwenDecoder`
for cell in nb["cells"]:
    if cell["cell_type"] == "code" and "qwen_decoder = QwenDecoder" in "".join(cell["source"]):
        source = "".join(cell["source"])
        if "device_map=decoder_device_map" in source:
            print("Found target cell for QwenDecoder init.")
            new_source = []
            for line in cell["source"]:
                # Remove the map definition logic
                if "decoder_device_map =" in line:
                    continue
                if "torch.cuda.is_available" in line:
                    continue
                if "decoder_device_map = {" in line:
                    continue
                    
                # Change the line passing the map
                if "device_map=decoder_device_map" in line:
                    new_source.append("    device_map=None # Disable auto-map to prevent sharding errors on single node\n")
                else:
                    new_source.append(line)
            
            cell["source"] = new_source
            print("Disabled device_map in QwenDecoder init.")

with open(notebook_path, "w") as f:
    json.dump(nb, f, indent=4)
print("Successfully modified notebook with v3 fixes.")
