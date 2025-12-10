
import json
import sys
from pathlib import Path

def insert_cell(notebook_path):
    path = Path(notebook_path)
    if not path.exists():
        print(f"Error: {path} not found")
        sys.exit(1)
        
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    data_loader_source = [
        "# 5. Dataset and Data Loader\n",
        "print(\"\\n\" + \"=\"*60)\n",
        "print(\"INITIALIZING DATASETS\")\n",
        "print(\"=\"*60)\n",
        "\n",
        "# Create datasets\n",
        "train_dataset = PixmoQADataset(\n",
        "    parquet_path=config.dataset.train_parquet,\n",
        "    tokenizer=qwen_decoder.tokenizer,\n",
        "    image_transforms=get_image_transforms(config.dataset.image_size, is_train=True),\n",
        "    max_question_length=128,\n",
        "    max_answer_length=256,\n",
        ")\n",
        "\n",
        "val_dataset = PixmoQADataset(\n",
        "    parquet_path=config.dataset.val_parquet,\n",
        "    tokenizer=qwen_decoder.tokenizer,\n",
        "    image_transforms=get_image_transforms(config.dataset.image_size, is_train=False),\n",
        "    max_question_length=128,\n",
        "    max_answer_length=256,\n",
        ")\n",
        "\n",
        "print(f\"\\nTrain dataset: {len(train_dataset)} samples\")\n",
        "print(f\"Val dataset: {len(val_dataset)} samples\")\n",
        "\n",
        "# Collate function\n",
        "def collate_fn(batch):\n",
        "    from torch.nn.utils.rnn import pad_sequence\n",
        "    pad_idx = qwen_decoder.tokenizer.pad_token_id\n",
        "    \n",
        "    images = torch.stack([b['image'] for b in batch])\n",
        "    \n",
        "    q_padded = pad_sequence([b['question_ids'] for b in batch], batch_first=True, padding_value=pad_idx)\n",
        "    a_padded = pad_sequence([b['answer_ids'] for b in batch], batch_first=True, padding_value=pad_idx)\n",
        "    q_mask = pad_sequence([b['question_mask'] for b in batch], batch_first=True, padding_value=0)\n",
        "    a_mask = pad_sequence([b['answer_mask'] for b in batch], batch_first=True, padding_value=0)\n",
        "    \n",
        "    return {\n",
        "        'images': images,\n",
        "        'question_ids': q_padded,\n",
        "        'answer_ids': a_padded,\n",
        "        'question_mask': q_mask,\n",
        "        'answer_mask': a_mask,\n",
        "    }\n",
        "\n",
        "# Data Loaders\n",
        "train_loader = DataLoader(\n",
        "    train_dataset,\n",
        "    batch_size=config.dataset.batch_size,\n",
        "    shuffle=True,\n",
        "    num_workers=4,\n",
        "    pin_memory=True,\n",
        "    collate_fn=collate_fn\n",
        ")\n",
        "\n",
        "val_loader = DataLoader(\n",
        "    val_dataset,\n",
        "    batch_size=config.dataset.batch_size,\n",
        "    shuffle=False,\n",
        "    num_workers=4,\n",
        "    pin_memory=True,\n",
        "    collate_fn=collate_fn\n",
        ")\n",
        "\n",
        "print(f\"Train loader: {len(train_loader)} batches\")\n"
    ]
    
    new_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": data_loader_source
    }
    
    # Find insertion point: After "Initialize Pretrained Qwen Decoder" cell
    insert_idx = -1
    for i, cell in enumerate(nb['cells']):
        source = "".join(cell.get('source', []))
        if "INITIALIZING QWEN DECODER" in source and cell['cell_type'] == 'code':
            insert_idx = i + 1
            break
            
    if insert_idx != -1:
        # Check if already present to avoid multiple insertions
        next_cell_source = "".join(nb['cells'][insert_idx].get('source', []))
        if "INITIALIZING DATASETS" in next_cell_source:
             print("Dataset cell already appears to be present.")
        else:
             nb['cells'].insert(insert_idx, new_cell)
             print(f"Inserted dataset cell at index {insert_idx}")
             with open(path, 'w', encoding='utf-8') as f:
                json.dump(nb, f, indent=1)
    else:
        print("Could not find Qwen Decoder initialization cell.")

if __name__ == "__main__":
    notebook_path = "/home/hice1/vchopra37/scratch/projects/edge_glass/edge_glass_modular/notebooks/03_trm_vlm_qa_training_FIXED.ipynb"
    insert_cell(notebook_path)
