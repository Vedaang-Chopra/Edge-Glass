import json
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

class PixmoFeatureDataset(Dataset):
    """
    Loads the pre-extracted image features.
    It also fixes old paths like 'data/pixmo/...' -> 'data/data/pixmo/...'
    """
    def __init__(self, index_file: str | Path):
        index_file = Path(index_file)
        with open(index_file, "r") as f:
            self.index = json.load(f)

        # Base dir where your index lives (e.g. ./data/data/pixmo)
        self.base_dir = index_file.parent

    def _fix_path(self, raw_path: str) -> Path:
        p = Path(raw_path)

        # If it's already absolute and exists, just return
        if p.is_absolute() and p.exists():
            return p

        # Common case: path stored as "data/pixmo/features/xxx.pt"
        # but actual is "data/data/pixmo/features/xxx.pt"
        s = str(p)

        if "data/pixmo" in s and not p.exists():
            s = s.replace("data/pixmo", "data/data/pixmo")
            p2 = Path(s)
            if p2.exists():
                return p2

        # Otherwise, try resolving relative to the index directory
        p3 = (self.base_dir / p.name)  # fallback: same dir, same filename
        if p3.exists():
            return p3

        # Last resort: just return the original; will raise if missing
        return p

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        meta = self.index[idx]
        raw_path = meta["file"]
        file_path = self._fix_path(raw_path)

        blob = torch.load(file_path)

        return {
            "features": blob["features"],          # (num_patches, feat_dim)
            "caption": blob["caption"],            # raw caption text
            "num_patches": meta["num_patches"],
            "orig_idx": meta["orig_idx"],
        }



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
        }


def collate_alignment(batch, tokenizer, device="cpu"):
    """
    batch: list of dicts with keys:
      - features: (L_i, D)
      - text: str
      - modality: "vision" or "audio"
    """
    # 1) Feature padding
    seqs = [b["features"] for b in batch]             # list of (L_i, D)
    lengths = [s.size(0) for s in seqs]
    max_len = max(lengths)
    feat_dim = seqs[0].size(1)
    B = len(batch)

    feats = torch.zeros(B, max_len, feat_dim, dtype=seqs[0].dtype)
    feat_mask = torch.zeros(B, max_len, dtype=torch.bool)

    for i, (s, L) in enumerate(zip(seqs, lengths)):
        feats[i, :L] = s
        feat_mask[i, :L] = True   # True where there is real data

    # 2) Text tokenization (for LLM)
    texts = [b["text"] for b in batch]
    tok = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    )

    # 3) Modality as tensor of ints (0 = vision, 1 = audio), if you want
    modality_strs = [b.get("modality", "vision") for b in batch]
    modality_ids = torch.tensor(
        [0 if m == "vision" else 1 for m in modality_strs],
        dtype=torch.long,
    )

    batch_out = {
        "features": feats.to(device),                 # (B, max_L, D)
        "feature_mask": feat_mask.to(device),         # (B, max_L)
        "input_ids": tok["input_ids"].to(device),     # (B, T_text)
        "attention_mask": tok["attention_mask"].to(device),
        "modality_ids": modality_ids.to(device),      # (B,)
        "raw_text": texts,
    }
    return batch_out
