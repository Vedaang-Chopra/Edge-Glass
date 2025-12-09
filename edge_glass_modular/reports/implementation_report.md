# Edge Glass VLM: Implementation Report

## 1. System Architecture

The **Edge Glass** framework is designed as a modular pipeline for training aligned Multimodal Large Language Models (MLLMs). It prioritizes composability, allowing researchers to easily swap encoders, projection layers, and decoders via configuration files.

### 1.1 Core Components

#### Encoders
*   **Vision Encoder**: `CLIPVisionModel` initialized from `openai/clip-vit-large-patch14-336`.
    *   **Input**: 336x336 pixel images.
    *   **Frozen Backbone**: The CLIP encoder is kept frozen to preserve robust pre-trained features.
    *   **Pooling Strategies**: 
        *   `AttentionPooling`: A learnable pooling layer (default baseline).
        *   `PerceiverResampler`: An experimental resampler with learnable latent queries.
    *   **MRL Head**: A `MatryoshkaProjection` layer outputs embeddings at multiple granularities: `[4096, 2048, 1024, 512, 256, 128]`.

*   **Text Encoder**: `SentenceTransformer` initialized from `sentence-transformers/all-mpnet-base-v2`.
    *   **Frozen Backbone**: Ensures semantic text embeddings remain stable.
    *   **MRL Projection**: Aligned with the vision encoder's output dimensions.

#### Decoders (Generative)
*   **Qwen2.5**: Integration of `Qwen/Qwen2.5-7B-Instruct` and `.3B-Instruct`.
    *   **Optimization**: Supports 4-bit and 8-bit quantization via `BitsAndBytes`.
    *   **LoRA**: Parameter-Efficient Fine-Tuning (PEFT) using LoRA (Rank 64, Alpha 16) is applied to projection and attention layers.
*   **TRM (Tiny Recursive Model)**: A bespoke, lightweight transformer decoder with Rotary Embeddings (RoPE) and RMSNorm, intended for extreme resource-constrained environments.

#### Multimodal Fusion
*   **Fusion Strategy**: Visual features are projected into the LLM's embedding space and inserted as "prefix tokens" before the text prompt.
*   **Connectivity**: 
    *   `concat`: Simple concatenation followed by an MLP.
    *   `cross_attention`: Cross-attention layers where text queries attend to visual keys/values.
    *   `gated`: Learned gating to modulate visual information injection.

---

## 2. Training Pipeline

### 2.1 Optimization & Infrastructure
The project employs two distinct training strategies depending on the phase:

1.  **Alignment Phase**: Uses `ImprovedMultimodalTrainer` (Custom wrapper around PyTorch DDP).
2.  **VLM QA Phase**: Uses `scripts/train_vlm_accelerate.py` leveraging **HuggingFace Accelerate** + **DeepSpeed**.

#### Accelerate-based Pipeline (VLM QA)
*   **Framework**: `accelerate` library manages distributed training (multi-GPU) and mixed precision.
*   **DeepSpeed**: Integrated via `DeepSpeedPlugin` (ZeRO Stage 2) for memory optimization during instruction tuning.
*   **Optimizations**: 
    *   Explicitly disables `torch.compile` / `dynamo` to prevent backend compiler errors (e.g. CUDAGraphs issues).
    *   **BF16**: Enforced mixed precision.
    *   **Gradient Accumulation**: Supported for simulating larger batch sizes.
*   **Loader**: Uses a custom training loop with `accelerate.prepare()` wrapping model, optimizer, and dataloaders.

### 2.2 Data Processing
*   **Dataset**: The **Pixmo** dataset is processed from Parquet files.
*   **Decoding**: Images are decoded on-the-fly using PIL to optimize memory bandwidth.
*   **Augmentation**: 
    *   **Text Dropout**: Captions are dropped with 10% probability during alignment training to force the model to rely on visual features (improving unimodal robustness).
    *   **Filtering**: Automated checks remove corrupted images or empty captions.

### 2.3 Loss Functions
*   **Alignment Loss**: 
    *   Combination of **Contrastive Loss** (InfoNCE) and **MRL Loss**.
    *   The MRL loss is a weighted sum of contrastive losses calculated at each embedding dimension `[4096, 2048, ..., 128]`.
    *   `sample_single_mrl_dim=True`: Randomly samples one dimension per step during training to efficiently enforce the Matryoshka structure.
*   **Instruction Tuning Loss**: 
    *   **Causal Language Modeling (CLM)**: Standard next-token prediction loss, applied only to the answer portion of the visual question-answering pairs.

## 3. Utility Scripts & Automation

The project includes a suite of python scripts in `edge_glass_modular/scripts/` to automate experimentation and notebook management:

### 3.1 Evaluation Notebook Generators
*   `create_alignment_eval_notebook.py`: Automatically generates a Jupyter notebook (`05_alignment_evaluation.ipynb`) pre-filled with code for:
    *   Loading the trained alignment model.
    *   Computing **Recall@K** metrics (R@1, R@5, R@10) for image-to-text and text-to-image retrieval.
    *   Visualizing t-SNE plots of the shared embedding space.
    *   Generating similarity heatmaps for batch analysis.
*   `create_eval_notebook.py` & `fix_eval_notebook.py`: Scripts to generate and patch general evaluation notebooks, ensuring path correctness and consistency.

### 3.2 Infrastructure & Optimization Tools
*   `optimize_gpu_usage.py`: A utility to parse Jupyter notebooks and inject code for optimized device placement (e.g., forcing models to specific CUDA devices, balancing `device_map` for QwenDecoder) to avoid OOM errors during interactive debugging.
*   `fix_collate_fn.py` / `fix_notebook_transforms.py`: Targeted "hotfix" scripts that modify existing notebooks to correct data collation logic and image transform pipelines without manual editing.
