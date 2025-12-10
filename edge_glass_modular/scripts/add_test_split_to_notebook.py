
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
        
    # Content to replace the "INITIALIZING DATASETS" cell
    dataset_cell_source = [
        "# 5. Dataset and Data Loader\n",
        "print(\"\\n\" + \"=\"*60)\n",
        "print(\"INITIALIZING DATASETS\")\n",
        "print(\"=\"*60)\n",
        "\n",
        "# Create datasets\n",
        "# Define transforms globally\n",
        "train_transforms = get_image_transforms(config.dataset.image_size, is_training=True)\n",
        "val_transforms = get_image_transforms(config.dataset.image_size, is_training=False)\n",
        "\n",
        "train_dataset = PixmoQADataset(\n",
        "    parquet_path=config.dataset.train_parquet,\n",
        "    tokenizer=qwen_decoder.tokenizer,\n",
        "    image_transforms=get_image_transforms(config.dataset.image_size, is_training=True),\n",
        "    max_question_length=128,\n",
        "    max_answer_length=256,\n",
        ")\n",
        "\n",
        "val_dataset = PixmoQADataset(\n",
        "    parquet_path=config.dataset.val_parquet,\n",
        "    tokenizer=qwen_decoder.tokenizer,\n",
        "    image_transforms=get_image_transforms(config.dataset.image_size, is_training=False),\n",
        "    max_question_length=128,\n",
        "    max_answer_length=256,\n",
        ")\n",
        "\n",
        "test_dataset = PixmoQADataset(\n",
        "    parquet_path=config.dataset.test_parquet,\n",
        "    tokenizer=qwen_decoder.tokenizer,\n",
        "    image_transforms=get_image_transforms(config.dataset.image_size, is_training=False),\n",
        "    max_question_length=128,\n",
        "    max_answer_length=256,\n",
        ")\n",
        "\n",
        "print(f\"\\nTrain dataset: {len(train_dataset)} samples\")\n",
        "print(f\"Val dataset: {len(val_dataset)} samples\")\n",
        "print(f\"Test dataset: {len(test_dataset)} samples\")\n",
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
        "        'answers': [b['answer'] for b in batch],\n",
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
        "test_loader = DataLoader(\n",
        "    test_dataset,\n",
        "    batch_size=config.dataset.batch_size,\n",
        "    shuffle=False,\n",
        "    num_workers=4,\n",
        "    pin_memory=True,\n",
        "    collate_fn=collate_fn\n",
        ")\n",
        "\n",
        "print(f\"Train loader: {len(train_loader)} batches\")\n"
    ]

    # Find and update the dataset cell
    dataset_cell_idx = -1
    for i, cell in enumerate(nb['cells']):
        source = "".join(cell.get('source', []))
        if "INITIALIZING DATASETS" in source:
            dataset_cell_idx = i
            break
            
    if dataset_cell_idx != -1:
        print(f"Updating dataset initialization at cell {dataset_cell_idx}")
        nb['cells'][dataset_cell_idx]['source'] = dataset_cell_source
    else:
        print("Could not find dataset initialization cell to update.")

    # Write back
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
        
    print("Notebook updated successfully.")

if __name__ == "__main__":
    notebook_path = "/home/hice1/vchopra37/scratch/projects/edge_glass/edge_glass_modular/notebooks/03_trm_vlm_qa_training_FIXED.ipynb"
    update_notebook(notebook_path)
