
import json
import sys

notebook_path = '/home/hice1/vchopra37/scratch/projects/edge_glass/edge_glass_modular/notebooks/05_alignment_evaluation.ipynb'

try:
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    modified = False
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source_lines = cell['source']
            new_source = []
            cell_modified = False
            for line in source_lines:
                if "model.load_state_dict(checkpoint['model_state_dict'])" in line:
                    new_line = line.replace("model.load_state_dict(checkpoint['model_state_dict'])", "model.load_state_dict(checkpoint['model_state_dict'], strict=False)")
                    new_source.append(new_line)
                    cell_modified = True
                elif "print('Checkpoint loaded.')" in line:
                     new_source.append(line.replace("print('Checkpoint loaded.')", "print('Checkpoint loaded with strict=False.')"))
                else:
                    new_source.append(line)
            
            if cell_modified:
                cell['source'] = new_source
                modified = True
                print("Found and modified the checkpoint loading cell.")
                # We can stop after modifying the specific cell if we want, or continue. 
                # Since this line is unique enough, it's safe.

    if modified:
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)
        print(f"Successfully modified {notebook_path}")
    else:
        print("Could not find the target line to modify.")
        sys.exit(1)

except Exception as e:
    print(f"Error modifying notebook: {e}")
    sys.exit(1)
