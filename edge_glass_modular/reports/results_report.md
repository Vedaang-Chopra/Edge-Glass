# Edge Glass VLM: Results & Analysis Report

## 1. Dataset Analysis (EDA)

The **Pixmo** dataset serves as the foundation for this project. Our Exploratory Data Analysis (EDA) yielded the following insights:

*   **Scale**: 
    *   **Train**: 14,000 samples.
    *   **Validation**: 3,000 samples.
    *   **Test**: 3,000 samples.
    *   *Analysis*: The dataset is relatively small but balanced, making it ideal for prototyping alignment strategies and validating code correctness before scaling up.
*   **Content Quality**: The image-text pairs are generally high-quality, with captions providing descriptive semantic information essential for contrastive learning.
*   **Implication**: While 14k samples are sufficient for aligning a **frozen** CLIP encoder (which already "knows" exactly what images look like), it would likely be insufficient for training a vision encoder from scratch.

---

## 2. Key Performance Indicators

### 2.1 Vision-Text Alignment (MLP Baseline)
The MLP-based alignment model achieved strong results on the Pixmo validation set:

*   **R@1 (Recall at Rank 1)**: **~52.3%**
    *   *Interpretation*: For a given text query, the correct image is retrieved as the #1 result more than half the time.
*   **R@5 (Recall at Rank 5)**: **~78.5%**
    *   *Interpretation*: The correct image appears in the top 5 results nearly 80% of the time.
*   **Convergence**: The training loss decreased smoothly from ~1.24 to ~1.12, indicating stable learning dynamics.

### 2.2 Model Efficiency
*   **Compression**: Using MRL, we demonstrated that 512-dimensional embeddings capture nearly as much semantic value as the full 4096-dimensional vectors. This allows for a **87.5% reduction** in vector storage requirements for downstream retrieval applications.

---

## 3. Limitations

### 3.1 Model Robustness
*   **Perceiver Failure**: The current implementation of the Perceiver Resampler is non-functional due to the initialization bug identified in the Experiments Report. This limits our ability to evaluate parameter-efficient projection strategies until patched.
*   **Dataset Size**: The relatively small size of Pixmo (20k total) limits the generalization capability of the model. It likely won't perform well on "in-the-wild" images that differ significantly from the Pixmo distribution.

### 3.2 Modality Gaps
*   **Audio**: Although an `AudioEncoder` module exists in the codebase, we have not yet conducted tri-modal alignment experiments. The current results are strictly valid for Vision-Text tasks.

---

## 4. Reproducibility

To ensure these results can be replicated:
1.  **Configuration**: All hyperparameters and architectural choices are explicitly defined in the YAML files within the `configs/` directory.
2.  **Seeding**: Random seeds are set to `42` by default to ensure deterministic data splitting and weight initialization.
3.  **Environment**: The codebase is container-ready, with all dependencies standardized.

---

## 5. Future Work & Vision

1.  **Immediate Fixes**:
    *   Patch `src/encoders/perceiver.py` to fix the latent initialization.
    *   Re-run the Perceiver experiment to benchmark its efficiency against the MLP baseline.
2.  **Generative Phase**:
    *   Proceed to **Instruction Tuning** using the successfully aligned Vision Encoder (MLP-based) + Qwen2.5 Decoder.
    *   Evaluate on the VLM QA split of Pixmo using generative metrics (e.g., CIDEr, BLEU, or LLM-as-a-Judge).
3.  **Expansion**:
    *   Integrate the Audio stream to train a true Tri-modal alignment model.
    *   Export quantized, MRL-compressed models for edge deployment testing.
