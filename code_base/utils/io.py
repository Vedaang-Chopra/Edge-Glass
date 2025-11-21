# code_base/utils/io.py
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Union
import json
import torch

PathLike = Union[str, Path]

def ensure_dir(p: PathLike) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

def save_json(obj: Dict[str, Any], path: PathLike) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    return path

def load_json(path: PathLike) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)

def save_pt(obj: Any, path: PathLike) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    torch.save(obj, path)
    return path

def load_pt(path: PathLike) -> Any:
    return torch.load(path, map_location="cpu")

def save_embeddings(ids, X: torch.Tensor, path: PathLike, meta: Dict[str, Any] = None) -> Path:
    """
    Saves a standard embeddings payload:
      { "ids": List[str], "X": torch.Tensor[B,D], "meta": Dict }
    """
    payload = {
        "ids": list(map(str, ids)),
        "X": X.detach().cpu(),   # keep it portable
        "meta": meta or {},
    }
    return save_pt(payload, path)

def load_embeddings(path: PathLike):
    """
    Returns (ids: List[str], X: torch.Tensor[B,D], meta: Dict)
    """
    payload = load_pt(path)
    ids = payload.get("ids", [])
    X = payload.get("X", None)
    meta = payload.get("meta", {})
    if not isinstance(X, torch.Tensor):
        raise ValueError(f"Embeddings file missing tensor 'X': {path}")
    return ids, X, meta
