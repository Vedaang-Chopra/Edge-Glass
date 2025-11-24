import json
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import random
import requests
from io import BytesIO
from PIL import Image
import numpy as np

class PixmoFeatureDataset(Dataset):
    """
    Dataset for pre-extracted PixMo image features.

    Expects an index JSON where each record has at least:
      - "file": path to a .pt blob with keys:
          * "features": (L, D) tensor
          * "caption": str
      - optionally: "num_patches", "orig_idx"

    This class:
      - fixes old paths like 'data/pixmo/...' -> 'data/data/pixmo/...'
      - filters out any entries whose feature file cannot be found
      - returns samples compatible with `collate_alignment`:
          {
            "features": (L, D) tensor,
            "text": str,
            "modality": "vision",
            ...extras...
          }
    """
    def __init__(self, index_file: str | Path):
        self.index_file = Path(index_file)
        self.base_dir = self.index_file.parent

        with open(self.index_file, "r") as f:
            raw_index = json.load(f)

        clean_index: list[dict] = []
        for rec in raw_index:
            raw_path = rec["file"]
            try:
                resolved = self._fix_path(raw_path)
            except FileNotFoundError:
                print(
                    f"[PixmoFeatureDataset] WARNING: skipping missing feature file: {raw_path}"
                )
                continue

            # store a resolved, absolute-ish path so __getitem__ is cheap
            rec = dict(rec)
            rec["resolved_path"] = str(resolved)
            clean_index.append(rec)

        self.index = clean_index
        print(
            f"[PixmoFeatureDataset] Loaded {len(self.index)} valid entries "
            f"from {len(raw_index)} total."
        )

    def _fix_path(self, raw_path: str) -> Path:
        """
        Try multiple strategies to fix incorrect dataset paths.

        Order:
          1. Use raw path if absolute + exists
          2. Fix common 'data/pixmo' → 'data/data/pixmo'
          3. Treat raw_path as relative to index dir
          4. Use just the filename under index dir
          5. Brute-force search for filename under base_dir

        Raises:
          FileNotFoundError if nothing works.
        """
        p = Path(raw_path)

        # Case 1: Raw path is already correct
        if p.is_absolute() and p.exists():
            return p

        # Case 2: Fix common double-"data" mistake
        s = str(raw_path)
        if "data/pixmo" in s:
            fixed = s.replace("data/pixmo", "data/data/pixmo")
            p2 = Path(fixed)
            if p2.exists():
                return p2

        # Case 3: Resolve relative to the index JSON's base_dir
        candidate = (self.base_dir / raw_path).resolve()
        if candidate.exists():
            return candidate

        # Case 4: Only filename under base_dir
        candidate2 = (self.base_dir / Path(raw_path).name).resolve()
        if candidate2.exists():
            return candidate2

        # Case 5: brute-force search for matching filename somewhere under base_dir
        matches = list(self.base_dir.rglob(Path(raw_path).name))
        if matches:
            return matches[0]

        raise FileNotFoundError(
            f"Cannot resolve feature file: '{raw_path}'. "
            f"Tried absolute, fixed, base_dir-relative, and recursive search."
        )

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict:
        rec = self.index[idx]
        file_path = Path(rec["resolved_path"])

        blob = torch.load(file_path, map_location="cpu")

        feats = blob["features"]              # (L, D)
        text = blob.get("caption", None)

        # fallback: allow text stored in index as "caption" or "text"
        if text is None:
            text = rec.get("caption", rec.get("text", ""))

        item: dict[str, Any] = {
            "features": feats,
            "text": text,
            "modality": "vision",
            "file": str(file_path),
        }

        # preserve extra metadata if present
        if "num_patches" in rec:
            item["num_patches"] = rec["num_patches"]
        if "orig_idx" in rec:
            item["orig_idx"] = rec["orig_idx"]

        return item




class LibriSpeechFeatureDataset(Dataset):
    """
    Loads pre-extracted Whisper features.
    Fixes broken paths like 'data/librispeech/...' -> 'data/data/librispeech/...'
    just like PixmoFeatureDataset does.
    """

    def __init__(self, index_file: str | Path):
        index_file = Path(index_file)
        with open(index_file, "r") as f:
            self.index = json.load(f)

        # Base directory where index.json lives
        self.base_dir = index_file.parent

    def _fix_path(self, raw_path: str) -> Path:
        """
        Try multiple strategies to fix incorrect dataset paths.
        1. Use raw path if absolute + exists
        2. Fix common 'data/librispeech' → 'data/data/librispeech'
        3. Try rewriting relative to index dir
        """
        p = Path(raw_path)

        # 1. If fully absolute and exists → OK
        if p.is_absolute() and p.exists():
            return p

        s = str(p)

        # 2. Common mismatch:
        #    raw: "data/librispeech/features/train_feat_123.pt"
        #    actual: "data/data/librispeech/features/train_feat_123.pt"
        if "data/librispeech" in s and not p.exists():
            s2 = s.replace("data/librispeech", "data/data/librispeech")
            p2 = Path(s2)
            if p2.exists():
                return p2

        # 3. Fallback: resolve relative to the index directory
        #    (useful if someone moved the index folder)
        p3 = self.base_dir / p.name
        if p3.exists():
            return p3

        # ❌ Last fallback → just return original (torch.load will raise)
        return p

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        meta = self.index[idx]

        raw_path = meta["file"]
        file_path = self._fix_path(raw_path)

        blob = torch.load(file_path)

        return {
            "features": blob["features"],      # (T_enc, d_audio)
            "text": blob["text"],
            "duration": blob["duration"],
            "sampling_rate": blob["sampling_rate"],
            "orig_idx": blob["orig_idx"],
            "modality": "audio",
        }




def collate_alignment(batch, tokenizer, device=None):
    """
    batch: list of items from PixmoFeatureDataset / LibriSpeechFeatureDataset
    Each item must have:
      - "features": (L_i, D)
      - "text": str
      - "modality": "vision" or "audio"
    Returns **CPU tensors** only (safe for Mac + CUDA with num_workers>0).
    """
    # 1) Feature padding
    seqs = [b["features"] for b in batch]  # list of (L_i, D)
    lengths = [s.size(0) for s in seqs]
    max_len = max(lengths)
    feat_dim = seqs[0].size(1)
    B = len(batch)

    feats = torch.zeros(B, max_len, feat_dim, dtype=seqs[0].dtype)
    feat_mask = torch.zeros(B, max_len, dtype=torch.bool)

    for i, (s, L) in enumerate(zip(seqs, lengths)):
        feats[i, :L] = s
        feat_mask[i, :L] = True

    # 2) Tokenize text with LLM tokenizer
    texts = [b["text"] for b in batch]
    tok = tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

    # 3) Modality ids: 0 = vision, 1 = audio
    modality_strs = [b.get("modality", "vision") for b in batch]
    modality_ids = torch.tensor(
        [0 if m == "vision" else 1 for m in modality_strs],
        dtype=torch.long,
    )

    return {
        "features": feats,                    # (B, L, D) on CPU
        "feature_mask": feat_mask,            # (B, L) on CPU
        "input_ids": tok["input_ids"],        # (B, T) on CPU
        "attention_mask": tok["attention_mask"],  # (B, T) on CPU
        "modality_ids": modality_ids,         # (B,) on CPU
        "raw_text": texts,
    }



# ---------------------------------------------------------------------------
# On-the-fly Datasets (from notebook)
# ---------------------------------------------------------------------------

class PixmoVisionDataset(Dataset):
    """
    On-the-fly image loading + CLIP feature extraction.

    If 'image' column exists: uses HF-managed images (no manual HTTP).
    Else: falls back to 'image_url' with robust skipping of bad URLs.

    Returns:
        {
          "features": Tensor(T, d_vision),
          "text": str
        }
    """
    def __init__(self, hf_dataset, vision_model, vision_processor, device, max_retries: int = 5):
        self.ds = hf_dataset
        self.vision_model = vision_model
        self.vision_processor = vision_processor
        self.device = device
        self.max_retries = max_retries

        # Determine columns
        cols = hf_dataset.column_names
        self.has_image_col = "image" in cols
        self.img_col = "image" if self.has_image_col else "image_url"
        self.txt_col = "caption"

    def __len__(self):
        return len(self.ds)

    def _load_image_from_url(self, url: str) -> Image.Image:
        resp = requests.get(url, timeout=10)
        # do NOT let this propagate; we'll catch in __getitem__
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content)).convert("RGB")
        return img

    def _encode_image(self, img: Image.Image):
        proc = self.vision_processor(images=img, return_tensors="pt")
        pixel_values = proc["pixel_values"].to(self.device)

        with torch.no_grad():
            out = self.vision_model(pixel_values=pixel_values)
            # (1, T, d_vision)
            feats = out.last_hidden_state.squeeze(0).to("cpu")  # (T, d_vision)
        return feats

    def _get_example(self, idx: int):
        ex = self.ds[idx]
        caption = ex[self.txt_col]

        if self.has_image_col:
            # HF has already downloaded/cached images; this is usually a PIL.Image
            img = ex[self.img_col]
            if not isinstance(img, Image.Image):
                img = img.convert("RGB")
        else:
            url = ex[self.img_col]
            img = self._load_image_from_url(url)

        feats = self._encode_image(img)
        return {
            "features": feats,
            "text": caption,
        }

    def __getitem__(self, idx: int):
        """
        Try up to max_retries times with different indices if something fails
        (HTTP error, decoding error, etc).
        """
        n = len(self.ds)
        attempt = 0
        cur_idx = idx

        while attempt < self.max_retries:
            try:
                return self._get_example(cur_idx)
            except Exception as e:
                # print(f"[PixmoVisionDataset] Failed idx={cur_idx}, attempt={attempt+1}, err={e}")
                attempt += 1
                cur_idx = (cur_idx + 1) % n

        # Final fallback: try random indices
        for _ in range(self.max_retries):
            j = random.randint(0, n - 1)
            try:
                return self._get_example(j)
            except Exception:
                continue

        raise RuntimeError("PixmoVisionDataset: could not load any valid images after multiple retries.")


class LibriSpeechAudioDataset(Dataset):
    """
    Dataset over the in-memory filtered LibriSpeech examples.
    Returns:
        {
          "features": Tensor(T_enc, d_audio),
          "text": str,
          "duration": float
        }
    """
    def __init__(self, examples, audio_processor, audio_model, device, max_len: int | None = None):
        self.examples = examples
        self.audio_processor = audio_processor
        self.audio_model = audio_model
        self.device = device
        if max_len is not None and max_len < len(examples):
            # Optionally cut down further for faster experiments
            self.examples = examples[:max_len]

    def __len__(self):
        return len(self.examples)

    def _whisper_encode_sequence(self, wav: np.ndarray, sr: int):
        """
        wav: 1D numpy array (time,)
        sr:  sampling rate (expected 16k)
        Returns:
            feats: Tensor(T_enc, d_audio) on CPU (float16)
        """
        # WhisperProcessor: raw waveform -> log-Mel spectrogram features
        inputs = self.audio_processor(
            wav,
            sampling_rate=sr,
            return_tensors="pt",
        )
        input_features = inputs["input_features"].to(self.device)  # (1, T_mel, 80)

        with torch.no_grad():
            enc_out = self.audio_model.encoder(input_features)
            hidden = enc_out.last_hidden_state  # (1, T_enc, d_audio)

        feats = hidden.squeeze(0).to(torch.float16).cpu()  # (T_enc, d_audio)
        return feats

    def __getitem__(self, idx: int):
        ex = self.examples[idx]
        wav = ex["waveform"]
        sr = ex["sampling_rate"]
        text = ex["text"]
        dur = ex["duration"]

        feats = self._whisper_encode_sequence(wav, sr)  # (T_enc, d_audio)

        return {
            "features": feats,
            "text": text,
            "duration": dur,
        }
