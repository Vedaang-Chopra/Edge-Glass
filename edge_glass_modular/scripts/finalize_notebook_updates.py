
import json
import sys
from pathlib import Path

def finalize_notebook(notebook_path):
    path = Path(notebook_path)
    if not path.exists():
        print(f"Error: {path} not found")
        sys.exit(1)
        
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    # 1. Fix max_steps
    fixed_steps = False
    for cell in nb['cells']:
        source = "".join(cell.get('source', []))
        if "max_steps = 100" in source:
            print("Found max_steps limitation. Fixing...")
            new_source = []
            for line in cell['source']:
                if "max_steps = 100" in line:
                    new_source.append("    max_steps = len(train_loader) * NUM_EPOCHS # Full training\n")
                else:
                    new_source.append(line)
            cell['source'] = new_source
            fixed_steps = True
            break
            
    if not fixed_steps:
        print("Could not find max_steps = 100 line. Maybe already fixed?")

    # 2. Append Test Evaluation Cell
    test_eval_source = [
        "print(\"\\n\" + \"=\"*60)\n",
        "print(\"TEST EVALUATION\")\n",
        "print(\"=\"*60)\n",
        "\n",
        "test_predictions = []\n",
        "test_targets = []\n",
        "\n",
        "with torch.no_grad():\n",
        "    for batch in tqdm(test_loader, desc=\"Testing\"):\n",
        "        images = batch['images'].to(device)\n",
        "        question_ids = batch['question_ids'].to(device)\n",
        "        answers = batch['answers']\n",
        "        \n",
        "        vision_tokens = encode_images(images)\n",
        "        \n",
        "        gen_outputs = model.generate(\n",
        "            vision_tokens,\n",
        "            question_ids,\n",
        "            max_new_tokens=32,\n",
        "            temperature=0.0,\n",
        "            return_stats=False,\n",
        "        )\n",
        "        \n",
        "        if isinstance(gen_outputs, dict):\n",
        "             generated_ids = gen_outputs['predictions']\n",
        "        else:\n",
        "             generated_ids = gen_outputs\n",
        "        \n",
        "        predictions = qwen_decoder.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)\n",
        "        \n",
        "        test_predictions.extend(predictions)\n",
        "        test_targets.extend(answers)\n",
        "\n",
        "test_metrics = evaluate_qa(test_predictions, test_targets)\n",
        "print(f\"\\nTest Results:\")\n",
        "print(f\"  EM: {test_metrics['em']:.2f}%\")\n",
        "print(f\"  F1: {test_metrics['f1']:.2f}%\")\n"
    ]
    
    new_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": test_eval_source
    }
    
    nb['cells'].append(new_cell)
    print("Appended Test Evaluation cell.")

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
        
    print("Notebook finalized successfully.")

if __name__ == "__main__":
    notebook_path = "/home/hice1/vchopra37/scratch/projects/edge_glass/edge_glass_modular/notebooks/03_trm_vlm_qa_training_FIXED.ipynb"
    finalize_notebook(notebook_path)
