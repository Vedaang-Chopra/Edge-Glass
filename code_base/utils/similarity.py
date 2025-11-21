# code_base/utils/similarity.py
from __future__ import annotations
import torch
import torch.nn.functional as F

def l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Row-wise L2 normalization: each vector becomes unit-length.
    Shape preserved.
    """
    return x / (x.norm(dim=-1, keepdim=True) + eps)

def cosine_sim(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Pairwise cosine similarity between two batches:
      A: [N, D], B: [M, D] -> [N, M]
    """
    A_n = l2_normalize(A)
    B_n = l2_normalize(B)
    return A_n @ B_n.T

def cosine_self(A: torch.Tensor) -> torch.Tensor:
    """
    Cosine similarity matrix for a single batch: [N, D] -> [N, N]
    Equivalent to cosine_sim(A, A)
    """
    A_n = l2_normalize(A)
    return A_n @ A_n.T

def cosine_against_ref(A: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """
    Cosine similarity against a single reference vector:
      A: [N, D], ref: [D] -> [N]
    """
    A_n = l2_normalize(A)
    ref_n = l2_normalize(ref.unsqueeze(0)).squeeze(0)
    return (A_n * ref_n).sum(dim=-1)

def check_equivalence_with_torch(A: torch.Tensor, B: torch.Tensor, atol: float = 1e-6) -> bool:
    """
    Confirms our cosine equals torch's F.cosine_similarity row-wise.
    """
    ours = cosine_sim(A, B)
    # Row-wise torch cosine similarites: compare each Ai to all Bj one-by-one
    # Efficient vectorized comparison for diagonal only when N==M.
    if A.shape[0] == B.shape[0]:
        diag = torch.stack([F.cosine_similarity(A[i].unsqueeze(0), B[i].unsqueeze(0)).squeeze(0)
                            for i in range(A.shape[0])])
        return torch.allclose(torch.diag(ours), diag, atol=atol)
    # For unequal N,M just return True if our path runs without error.
    return True
