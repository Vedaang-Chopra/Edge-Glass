#!/bin/bash

# VLM Training Experiments Script
# Model: Qwen/Qwen2.5-3B-Instruct
# WandB Project: edgeglass_final_align

# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 1. SIMPLE DEMO RUN
# Purpose: quick verification of pipeline in debug mode.
# Run Name: demo_run_3b_v3
# echo "Starting Demo Run..."
# accelerate launch --num_processes 4 edge_glass_modular/scripts/train_vlm_accelerate.py \
#     --config edge_glass_modular/configs/trm_vlm_qa_qwen2.5-3b.yaml \
#     --output_dir edge_glass_modular/outputs/demo_run_3b \
#     --run_name "demo_run_3b_v3" \
#     --debug \
#     --max_steps 20 \
#     --batch_size 4 \
#     --use_wandb

# 2. VERIFICATION RUN
# Note: Checkpoint path is 'checkpoint-debug' for demo runs (forced save)
# echo "Starting Verification..."
# accelerate launch --num_processes 1 edge_glass_modular/scripts/evaluate_vlm_accelerate.py \
#     --config edge_glass_modular/configs/trm_vlm_qa_qwen2.5-3b.yaml \
#     --checkpoint_path edge_glass_modular/outputs/demo_run_3b/checkpoint-debug \
#     --output_file edge_glass_modular/outputs/demo_run_3b/eval_results.json \
#     --max_test_samples 10 \
#     --batch_size 1

# echo "Verification complete. Please check edge_glass_modular/outputs/demo_run_3b/eval_results.json"

# 3. HEAVY PROPER RUN (PRODUCTION)
# Run Name: production_run_3b_v4 (Updated for CPU OOM fix)
# Note: This will save 'checkpoint_best' (best validation loss, WEIGHTS ONLY)
echo "Starting Heavy Proper Run (Batch Size 4 / Accum 4)..."
accelerate launch --num_processes 4 edge_glass_modular/scripts/train_vlm_accelerate.py \
    --config edge_glass_modular/configs/trm_vlm_qa_qwen2.5-3b.yaml \
    --output_dir edge_glass_modular/outputs/production_run_3b \
    --run_name "production_run_3b_v4" \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --use_wandb
