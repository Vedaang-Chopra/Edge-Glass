import json

notebook_path = "01_all_allignment.ipynb"
script_path = "run_notebook.py"

with open(notebook_path, "r") as f:
    nb = json.load(f)

with open(script_path, "w") as f:
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            f.write(source)
            f.write("\n\n")

print(f"Converted {notebook_path} to {script_path}")
