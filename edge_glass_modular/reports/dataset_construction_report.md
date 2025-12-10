# Dataset Construction & Methodology

## 1. Introduction
This report details the methodology used to construct the multi-modal datasets for the **Edge Glass** project. The primary objective was to create robust, high-performance, and offline-accessible datasets for training Vision-Language Models (VLMs) and Audio-Language Models. By converting raw data sources into structured **Parquet** files with embedded media (raw bytes or waveforms), we eliminate runtime network dependencies and optimize I/O throughput during training.

## 2. Design Principles
*   **Offline-First**: All media assets (images, audio) are downloaded and stored directly within the dataset files. This ensures training stability even without internet access and prevents "link rot" (missing URLs).
*   **Unified Format**: Apache Parquet is used as the container format due to its columnar efficiency and compatibility with the Hugging Face `datasets` library.
*   **Raw Media Embedding**: 
    *   **Images**: Stored as raw JPEG bytes (binary). This allows for on-the-fly decoding and augmentation.
    *   **Audio**: Stored as floating-point waveforms (numpy arrays) or raw bytes, resampled to a standard sample rate (typically 16kHz or 48kHz).

## 3. Dataset Preparation Pipelines

### 3.1. PixMo-Cap (Vision-Text)
The **PixMo-Cap** dataset (sourced from `allenai/pixmo-cap`) serves as the primary source for high-quality image-caption pairs.

*   **Source**: Hugging Face Hub (`allenai/pixmo-cap`).
*   **Methodology**:
    1.  **Streaming & Filtering**: The dataset is streamed to avoid downloading the entire metadata at once.
    2.  **Parallel Downloading**: A multiprocessing pool `(num_workers=16)` iterates through the image URLs.
    3.  **Image Processing**:
        *   Images are downloaded using `requests`.
        *   Invalid or timed-out URLs are filtered out (strict error handling).
        *   Images are saved as raw bytes column `image_bytes`.
    4.  **Storage**: The final dataset is saved as a Parquet file containing columns: `caption`, `image_url`, `id`, and `image_bytes`.
*   **Outcome**: A local Parquet subset (e.g., `pixmo_10k.parquet`) that is fully self-contained.

### 3.2. MusicCaps (Audio-Text)
The **MusicCaps** dataset (sourced from `google/MusicCaps`) provides high-quality music clips with rich textual descriptions.

*   **Source**: Hugging Face Hub (`google/MusicCaps`), identifying YouTube videos via `yt_id`.
*   **Methodology**:
    1.  **Audio Extraction**: The extraction pipeline utilizes `yt-dlp` and `ffmpeg`.
    2.  **Clip Processing**:
        *   The full audio is downloaded from YouTube.
        *   A 10-second clip is extracted based on the `start_s` and `end_s` timestamps.
        *   Audio is resampled to **48kHz** (or 16kHz depending on target model) and converted to mono/stereo as required.
    3.  **Storage**: The audio waveform is stored as a numpy array in the `audio` column, alongside `caption` and `aspect_list`.
*   **Technical Challenges**: `yt-dlp` requires external system dependencies (`ffmpeg`, `ffprobe`) and is sensitive to YouTube's changing API.

### 3.3. Clotho v2 (Audio-Captioning)
The **Clotho** dataset focuses on audio captioning for environmental sounds.

*   **Source**: Zenodo (Development Split).
*   **Methodology**:
    1.  **Input Data**: Raw `.wav` files and a metadata `.csv` file containing captions (5 captions per file).
    2.  **Waveform Loading**: The pipeline iterates through the CSV, matches filenames to the local `.wav` directory, and loads audio using `soundfile` or `librosa`.
    3.  **Structure**: The resulting Parquet file nests the 5 captions into a list and stores the raw waveform.
*   **Usage**: Primary validation set for audio-text alignment tasks.

### 3.4. Valor (Video-Text)
The **Valor** dataset is used for video-centric tasks, represented as multimodal shards.

*   **Quality Assessment**: The dataset was found to be of **very low quality**. Approximately **50% of the files were missing** or unavailable from YouTube during the download process.
*   **Methodology (derived from inspection)**:
    *   **Sharded Parquet**: Data is split into multiple shards (e.g., `valor32k_train_shard000.parquet`) for efficient distributed loading.
    *   **Multimodal Columns**:
        *   `image_jpegs`: A list of byte-strings, each acting as a keyframe for the video clip.
        *   `audio_wav`: The corresponding audio waveform for the clip.
        *   `caption`: Textual description.
        *   `video_id`, `start`, `end`: Metadata for temporal grounding.

### 3.5. LibriSpeech ASR (Audio-Text Alignment)
The **LibriSpeech ASR** dataset (`train.clean.100`) was used for audio text alignment experiments.

*   **Source**: OpenSLR/Hugging Face (`openslr/librispeech_asr`).
*   **Methodology**:
    1.  **Encoder Probing**: Utilized **Whisper Tiny** to generate fixed-size audio embeddings from the waveforms.
    2.  **Cosine Similarity**: Calculated audio-audio cosine similarity between clips to verify embedding consistency.
    3.  **ASR Verification**: Used Whisper ASR to transcribe audio and compare against ground truth text to ensure alignment correctness.
*   **Usage**: Primary dataset for proving the audio-text alignment pipeline before scaling to more complex audio tasks.

### 3.6. Dataset Performance & Alignment Ablation
Initial experiments revealed that the **pure audio datasets (MusicCaps, Clotho)** did not yield high-quality results for end-to-end multimodal tasks compared to vision-text pairs.
*   **Alignment Ablation**: Due to these performance limitations, the audio datasets were primarily used for **alignment ablation studies**. Ideally, this helps us understand the effectiveness of the projection layers (adapters) without relying on the audio tasks for the final benchmark performance.


## 4. Technical Implementation Details

### Parallelism & Efficiency
*   **`multiprocessing.Pool`**: Used for high-latency I/O operations (image downloads).
*   **`datasets.map(..., num_proc=N)`**: Used for CPU-intensive tasks (audio transcoding).
*   **Error Handling**: Robust `try-except` blocks ensure that a single failed download (404 error, private video) does not halt the entire pipeline. Failed samples are dropped.

### File Formats
*   **Parquet**: Chosen for compression (Snappy/Zstd) and columnar access speed. It allows loading *only* text columns for text-only pre-training or *only* image columns for vision encoding, significantly saving memory.

## 5. Conclusion
The constructed datasets provide a stable, high-performance foundation for the Edge Glass project. By decoupling the training data from the internet, we achieve deterministic training loops and faster epoch times, essential for iterating on experimental VLM architectures.
