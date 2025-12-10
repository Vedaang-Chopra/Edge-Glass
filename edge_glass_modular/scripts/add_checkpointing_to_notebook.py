
import nbformat

nb_path = "/home/hice1/vchopra37/scratch/projects/edge_glass/edge_glass_modular/notebooks/03_trm_vlm_qa_training_FIXED.ipynb"

def update_training_loop(nb):
    target_source_start = "# 4. Training Loop (Uncommented and Fixed)"
    
    new_source = """# 4. Training Loop (Uncommented and Fixed)
from tqdm.auto import tqdm
import torch

# Initialize optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
steps = 0
max_steps = 100 # Short run for verification

model.train()
progress_bar = tqdm(range(max_steps))
best_loss = float('inf')

print(f"Training started for {max_steps} steps...")

for batch in train_loader:
    images = batch["images"].to(device)
    with torch.no_grad():
        vision_tokens = encode_images(images)
    question_ids = batch["question_ids"].to(device)
    answer_ids = batch["answer_ids"].to(device)
    answer_mask = batch["answer_mask"].to(device) # Get answer mask
    
    # Forward pass
    outputs = model(
        vision_tokens=vision_tokens, 
        question_ids=question_ids, 
        answer_ids=answer_ids,
        answer_mask=answer_mask # Pass answer mask
    )
    loss = outputs.loss
    
    # Backward
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    # Checkpointing Logic (Save Best and Last)
    current_loss = loss.item()
    
    # Save best
    if current_loss < best_loss:
        best_loss = current_loss
        torch.save({
            'step': steps,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': best_loss,
        }, ckpt_dir / 'checkpoint_best.pt')
        
    # Save last periodically (every 50 steps)
    if steps > 0 and steps % 50 == 0:
         torch.save({
            'step': steps,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': current_loss,
        }, ckpt_dir / 'checkpoint_last.pt')
    
    progress_bar.set_description(f"Loss: {current_loss:.4f} | Best: {best_loss:.4f}")
    progress_bar.update(1)
    
    steps += 1
    if steps >= max_steps:
        break
        
# Final save
torch.save({
    'step': steps,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': current_loss,
}, ckpt_dir / 'checkpoint_last.pt')
print(f"Training complete. Best loss: {best_loss:.4f}")
"""

    found = False
    for cell in nb.cells:
        if cell.cell_type == 'code' and target_source_start in cell.source:
            cell.source = new_source
            found = True
            print("Updated Training Loop cell.")
            break
            
    if not found:
        print("Could not find the Training Loop cell to update.")

def update_evaluation_cell(nb):
    # Search for the cell that does evaluation and loads checkpoint
    target_eval_start = "# Load best checkpoint"
    
    new_eval_source = """# Load best checkpoint
try:
    ckpt_path = ckpt_dir / 'checkpoint_best.pt'
    if ckpt_path.exists():
        print(f"Loading best checkpoint from {ckpt_path}")
        best_ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(best_ckpt['model_state_dict'])
        print("✓ Loaded best checkpoint")
    else:
        print(f"⚠ Checkpoint not found at {ckpt_path}. Using current model state.")
except RuntimeError as e:
    print(f"⚠ Error loading checkpoint: {e}")
    print("This is likely due to a size mismatch from an old checkpoint.")
    print("Continuing with current model state...")

model.eval()
"""
    
    found = False
    # This cell might be harder to find by exact string if user modified it or if it's short.
    # I'll look for `best_ckpt = torch.load`
    for cell in nb.cells:
        if cell.cell_type == 'code' and "best_ckpt = torch.load" in cell.source:
            cell.source = new_eval_source
            found = True
            print("Updated Evaluation/Loading cell.")
            break
            
    if not found:
        print("Could not find the Evaluation Loading cell to update.")

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

update_training_loop(nb)
update_evaluation_cell(nb)

with open(nb_path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)
    print(f"Saved updated notebook to {nb_path}")
