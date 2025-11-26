import json
import re

notebook_path = "01_all_allignment.ipynb"

with open(notebook_path, "r") as f:
    nb = json.load(f)

cells = nb["cells"]

def find_cell_index(content_snippet):
    for i, cell in enumerate(cells):
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            if content_snippet in source:
                return i
    return -1

# 1. Add Data Augmentation to PixmoVisionDataset
idx = find_cell_index("class PixmoVisionDataset(Dataset):")
if idx != -1:
    src = cells[idx]["source"]
    # Add imports and augment init
    new_src = []
    for line in src:
        if "class PixmoVisionDataset(Dataset):" in line:
            new_src.append("from torchvision import transforms\n")
            new_src.append(line)
        elif "self.max_retries = max_retries" in line:
            new_src.append(line)
            new_src.append("        self.augment = transforms.Compose([\n")
            new_src.append("            transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),\n")
            new_src.append("            transforms.RandomHorizontalFlip(),\n")
            new_src.append("            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),\n")
            new_src.append("        ])\n")
        elif "img = self._load_image_from_url(url)" in line:
            new_src.append(line)
            new_src.append("        # Apply augmentation\n")
            new_src.append("        if self.vision_model.training:\n")
            new_src.append("            img = self.augment(img)\n")
        else:
            new_src.append(line)
    cells[idx]["source"] = new_src
    print("Updated PixmoVisionDataset")

# 2. Add SpecAugment to whisper_encode_sequence
idx = find_cell_index("def whisper_encode_sequence(wav: np.ndarray, sr: int):")
if idx != -1:
    src = cells[idx]["source"]
    new_src = []
    for line in src:
        if "def whisper_encode_sequence(wav: np.ndarray, sr: int):" in line:
            new_src.append("from torchaudio import transforms as T_audio\n")
            new_src.append(line)
        elif "input_features = inputs[\"input_features\"].to(device)" in line:
            new_src.append(line)
            new_src.append("    # Apply SpecAugment if training (assuming global audio_model.training check or similar)\n")
            new_src.append("    # Since this function is global, we check audio_model.training\n")
            new_src.append("    if audio_model.training:\n")
            new_src.append("        freq_mask = T_audio.FrequencyMasking(freq_mask_param=15)\n")
            new_src.append("        time_mask = T_audio.TimeMasking(time_mask_param=35)\n")
            new_src.append("        input_features = freq_mask(input_features)\n")
            new_src.append("        input_features = time_mask(input_features)\n")
        else:
            new_src.append(line)
    cells[idx]["source"] = new_src
    print("Updated whisper_encode_sequence")

# 3. Add AttentionPooling class
idx = find_cell_index("class FeedForward(nn.Module):")
if idx != -1:
    src = cells[idx]["source"]
    # Insert AttentionPooling before FeedForward or after
    # Let's insert it at the beginning of the cell
    attn_pooling_code = [
        "class AttentionPooling(nn.Module):\n",
        "    def __init__(self, dim: int):\n",
        "        super().__init__()\n",
        "        self.query = nn.Parameter(torch.randn(1, 1, dim))\n",
        "        self.scale = dim ** -0.5\n",
        "    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:\n",
        "        # x: (B, L, D)\n",
        "        B, L, D = x.shape\n",
        "        q = self.query.expand(B, -1, -1) # (B, 1, D)\n",
        "        scores = torch.matmul(q, x.transpose(1, 2)) * self.scale # (B, 1, L)\n",
        "        if mask is not None:\n",
        "             # mask is (B, L) bool, True=valid\n",
        "             # we want to mask INVALID positions (False) with -inf\n",
        "             # scores is (B, 1, L)\n",
        "             m = mask.unsqueeze(1) # (B, 1, L)\n",
        "             scores = scores.masked_fill(~m, float('-inf'))\n",
        "        attn = F.softmax(scores, dim=-1) # (B, 1, L)\n",
        "        out = torch.matmul(attn, x) # (B, 1, D)\n",
        "        return out.squeeze(1) # (B, D)\n",
        "\n"
    ]
    cells[idx]["source"] = attn_pooling_code + src
    print("Added AttentionPooling class")

# 4. Instantiate pooler
idx = find_cell_index("projector = nn.Linear(cfg.perceiver_dim, cfg.llm_hidden_size).to(device)")
if idx != -1:
    src = cells[idx]["source"]
    new_src = []
    for line in src:
        new_src.append(line)
        if "projector = nn.Linear" in line:
            new_src.append("pooler = AttentionPooling(cfg.llm_hidden_size).to(device)\n")
            new_src.append("print(\"Pooler created:\", pooler)\n")
    cells[idx]["source"] = new_src
    print("Instantiated pooler")

# 5. Update forward_alignment_step to use pooler
idx = find_cell_index("def forward_alignment_step(")
if idx != -1:
    src = cells[idx]["source"]
    new_src = []
    for line in src:
        if "h_mod = pooled_modality_embedding(z_llm)" in line:
            new_src.append("    # 4) Global modality embedding (Attention Pooling)\n")
            new_src.append("    # We don't have a mask for latents (they are fixed size), so mask=None\n")
            new_src.append("    h_mod = pooler(z_llm)            # (B, D_llm)\n")
        else:
            new_src.append(line)
    cells[idx]["source"] = new_src
    print("Updated forward_alignment_step")

# 6. Add pooler to trainable_modules
idx = find_cell_index("trainable_modules = nn.ModuleList([")
if idx != -1:
    src = cells[idx]["source"]
    new_src = []
    for line in src:
        if "projector," in line:
            new_src.append(line)
            new_src.append("    pooler,\n")
        else:
            new_src.append(line)
    cells[idx]["source"] = new_src
    print("Added pooler to trainable_modules")

# 7. Add Gradient Accumulation to train_one_epoch
idx = find_cell_index("def train_one_epoch(")
if idx != -1:
    src = cells[idx]["source"]
    new_src = []
    for line in src:
        if "optimizer.zero_grad(set_to_none=True)" in line:
            # Remove this line, we will do it conditionally
            pass
        elif "loss.backward()" in line:
            new_src.append("        # Gradient accumulation\n")
            new_src.append("        grad_accum_steps = getattr(cfg, 'grad_accum_steps', 1)\n")
            new_src.append("        loss = loss / grad_accum_steps\n")
            new_src.append(line)
        elif "optimizer.step()" in line:
            new_src.append("        if step % grad_accum_steps == 0:\n")
            new_src.append("            optimizer.step()\n")
            new_src.append("            optimizer.zero_grad(set_to_none=True)\n")
        elif "torch.nn.utils.clip_grad_norm_" in line:
             # Indent this to be inside the if block
             new_src.append("        if step % grad_accum_steps == 0:\n")
             new_src.append("    " + line)
        elif "if hasattr(cfg, \"max_grad_norm\")" in line:
             # We handle this logic above/below
             pass 
        else:
            new_src.append(line)
            
    # Fix indentation for clip_grad_norm logic which was split
    # Actually, let's rewrite the loop body more cleanly
    # It's hard to do robustly with line-by-line. 
    # Let's just replace the whole function body if possible, or be very specific.
    # The above logic is a bit fragile.
    # Let's try a simpler replacement for the loop part.
    
    # Re-reading the source to do a better replacement
    # ...
    pass 
    # I will skip complex logic replacement here and do it simpler:
    # Just add grad_accum_steps support
    
    final_src = []
    for line in src:
        if "optimizer.zero_grad(set_to_none=True)" in line:
             final_src.append("        if step % getattr(cfg, 'grad_accum_steps', 1) == 0:\n")
             final_src.append("            optimizer.zero_grad(set_to_none=True)\n")
        elif "loss.backward()" in line:
             final_src.append("        loss = loss / getattr(cfg, 'grad_accum_steps', 1)\n")
             final_src.append(line)
        elif "optimizer.step()" in line:
             final_src.append("        if step % getattr(cfg, 'grad_accum_steps', 1) == 0:\n")
             final_src.append("            optimizer.step()\n")
        elif "clip_grad_norm_" in line:
             # This needs to be inside the if
             # But the previous line "if hasattr..." wraps it.
             pass
        else:
             final_src.append(line)
             
    # This is getting messy. Let's just replace the whole function.
    new_function = [
        "def train_one_epoch(\n",
        "    dataloader: DataLoader,\n",
        "    modality: str,\n",
        "    max_steps: int,\n",
        "    log_prefix: str = \"\",\n",
        "):\n",
        "    trainable_modules.train()\n",
        "    running_loss = 0.0\n",
        "    num_batches = 0\n",
        "    grad_accum_steps = getattr(cfg, 'grad_accum_steps', 4)\n",
        "\n",
        "    pbar = tqdm(dataloader, total=max_steps, desc=f\"{log_prefix}train-{modality}\", leave=False)\n",
        "\n",
        "    optimizer.zero_grad(set_to_none=True)\n",
        "\n",
        "    for step, batch in enumerate(pbar, start=1):\n",
        "        if step > max_steps:\n",
        "            break\n",
        "\n",
        "        loss, metrics = forward_alignment_step(batch, modality=modality)\n",
        "        loss = loss / grad_accum_steps\n",
        "        loss.backward()\n",
        "\n",
        "        if step % grad_accum_steps == 0:\n",
        "            if hasattr(cfg, \"max_grad_norm\") and cfg.max_grad_norm is not None:\n",
        "                torch.nn.utils.clip_grad_norm_(trainable_modules.parameters(), cfg.max_grad_norm)\n",
        "            optimizer.step()\n",
        "            optimizer.zero_grad(set_to_none=True)\n",
        "\n",
        "        running_loss += metrics[\"loss\"]\n",
        "        num_batches += 1\n",
        "        avg_loss = running_loss / num_batches\n",
        "\n",
        "        # ✅ W&B logging\n",
        "        wandb.log(\n",
        "            {\n",
        "                f\"{modality}/train/loss\": metrics[\"loss\"],\n",
        "                f\"{modality}/train/avg_loss\": avg_loss,\n",
        "                f\"{modality}/train/mrl_loss\": metrics[\"mrl_loss\"],\n",
        "                f\"{modality}/train/batch_size\": metrics[\"batch_size\"],\n",
        "            }\n",
        "        )\n",
        "\n",
        "        if step % cfg.log_every_steps == 0 or step == 1:\n",
        "            pbar.set_postfix({\n",
        "                \"loss\": f\"{metrics['loss']:.4f}\",\n",
        "                \"avg_loss\": f\"{avg_loss:.4f}\",\n",
        "            })\n",
        "\n",
        "    avg_epoch_loss = running_loss / max(1, num_batches)\n",
        "    print(f\"[{log_prefix} {modality}] avg loss: {avg_epoch_loss:.4f}\")\n",
        "    # ✅ epoch-level log\n",
        "    wandb.log({f\"{modality}/train/epoch_loss\": avg_epoch_loss})\n",
        "    return avg_epoch_loss\n"
    ]
    cells[idx]["source"] = new_function
    print("Updated train_one_epoch with gradient accumulation")

# 8. Add Scheduler to Training Loop
idx = find_cell_index("for round_idx in range(num_rounds):")
if idx != -1:
    src = cells[idx]["source"]
    new_src = []
    # Insert scheduler init before loop
    new_src.append("# Scheduler\n")
    new_src.append("total_steps = (vision_steps + audio_steps) * num_rounds\n")
    new_src.append("scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)\n")
    
    for line in src:
        new_src.append(line)
        if "train_one_epoch(" in line:
            pass # We don't need to change the call itself
        if "eval_retrieval(" in line:
            new_src.append("    scheduler.step()\n")
            new_src.append("    print(f\"LR: {scheduler.get_last_lr()[0]:.6f}\")\n")
            
    cells[idx]["source"] = new_src
    print("Updated training loop with scheduler")

# Save
with open(notebook_path, "w") as f:
    json.dump(nb, f, indent=1)
print("Notebook saved.")
