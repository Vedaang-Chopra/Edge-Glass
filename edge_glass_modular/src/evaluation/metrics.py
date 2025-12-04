"""Retrieval and alignment evaluation metrics."""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class RetrievalMetrics:
    """Container for retrieval metrics."""

    # Recall @ K metrics
    r_at_1: float
    r_at_5: float
    r_at_10: float
    r_at_50: float

    # Rank statistics
    mean_rank: float
    median_rank: float

    # Precision metrics
    map_at_10: float  # Mean Average Precision @ 10
    map_at_50: float  # Mean Average Precision @ 50

    # NDCG metrics
    ndcg_at_10: float
    ndcg_at_50: float

    # Raw ranks for analysis
    ranks: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary (excluding ranks array)."""
        return {
            'R@1': self.r_at_1,
            'R@5': self.r_at_5,
            'R@10': self.r_at_10,
            'R@50': self.r_at_50,
            'mean_rank': self.mean_rank,
            'median_rank': self.median_rank,
            'mAP@10': self.map_at_10,
            'mAP@50': self.map_at_50,
            'NDCG@10': self.ndcg_at_10,
            'NDCG@50': self.ndcg_at_50,
        }


def compute_retrieval_metrics(
    query_embs: torch.Tensor,
    key_embs: torch.Tensor,
    ks: Tuple[int, ...] = (1, 5, 10, 50),
) -> RetrievalMetrics:
    """Compute comprehensive retrieval metrics.

    Args:
        query_embs: Query embeddings (N, D)
        key_embs: Key embeddings (N, D)
        ks: K values for Recall@K

    Returns:
        RetrievalMetrics with all computed metrics
    """
    # Normalize embeddings
    query_embs = F.normalize(query_embs, p=2, dim=-1)
    key_embs = F.normalize(key_embs, p=2, dim=-1)

    # Compute similarity matrix
    sims = torch.matmul(query_embs, key_embs.t())  # (N, N)
    N = sims.size(0)

    # Ground truth: each query matches the key at the same index
    targets = torch.arange(N, device=sims.device)

    # Sort by similarity (descending)
    _, indices = sims.sort(dim=1, descending=True)

    # Find ranks of correct matches (0-indexed)
    ranks = (indices == targets.unsqueeze(1)).nonzero(as_tuple=False)[:, 1]
    ranks_np = ranks.cpu().numpy()

    # Recall@K
    recall_scores = {}
    for k in ks:
        hit_rate = (ranks < k).float().mean().item()
        recall_scores[k] = hit_rate * 100.0

    # Rank statistics (1-indexed for interpretability)
    mean_rank = (ranks.float() + 1).mean().item()
    median_rank = (ranks.float() + 1).median().item()

    # Mean Average Precision @ K
    map_10 = compute_map_at_k(ranks, k=10)
    map_50 = compute_map_at_k(ranks, k=50)

    # NDCG @ K
    ndcg_10 = compute_ndcg_at_k(ranks, k=10)
    ndcg_50 = compute_ndcg_at_k(ranks, k=50)

    return RetrievalMetrics(
        r_at_1=recall_scores.get(1, 0.0),
        r_at_5=recall_scores.get(5, 0.0),
        r_at_10=recall_scores.get(10, 0.0),
        r_at_50=recall_scores.get(50, 0.0),
        mean_rank=mean_rank,
        median_rank=median_rank,
        map_at_10=map_10,
        map_at_50=map_50,
        ndcg_at_10=ndcg_10,
        ndcg_at_50=ndcg_50,
        ranks=ranks_np,
    )


def compute_map_at_k(ranks: torch.Tensor, k: int) -> float:
    """Compute Mean Average Precision @ K.

    For single relevant item: AP = 1 / (rank + 1) if rank < k else 0
    """
    ap = torch.where(
        ranks < k,
        1.0 / (ranks.float() + 1.0),
        torch.zeros_like(ranks, dtype=torch.float)
    )
    return ap.mean().item()


def compute_ndcg_at_k(ranks: torch.Tensor, k: int) -> float:
    """Compute Normalized Discounted Cumulative Gain @ K.

    DCG = 1 / log2(rank + 2) if rank < k else 0
    IDCG = 1 (since there's only one relevant item)
    """
    gains = torch.where(
        ranks < k,
        1.0 / torch.log2(ranks.float() + 2.0),
        torch.zeros_like(ranks, dtype=torch.float),
    )
    return gains.mean().item()


def compute_mrl_performance(
    vision_embs: torch.Tensor,
    text_embs: torch.Tensor,
    mrl_dims: List[int],
) -> Dict[int, Dict[str, RetrievalMetrics]]:
    """Evaluate retrieval performance at different MRL dimensions.

    Args:
        vision_embs: Vision embeddings (N, D)
        text_embs: Text embeddings (N, D)
        mrl_dims: List of dimensions to evaluate

    Returns:
        Dict mapping dimension -> {'i2t': metrics, 't2i': metrics}
    """
    results = {}

    for dim in sorted(mrl_dims):
        # Truncate embeddings
        v_trunc = vision_embs[:, :dim]
        t_trunc = text_embs[:, :dim]

        # Image-to-text
        i2t_metrics = compute_retrieval_metrics(v_trunc, t_trunc)

        # Text-to-image
        t2i_metrics = compute_retrieval_metrics(t_trunc, v_trunc)

        results[dim] = {
            'i2t': i2t_metrics,
            't2i': t2i_metrics,
        }

    return results


def compute_similarity_statistics(
    vision_embs: torch.Tensor,
    text_embs: torch.Tensor,
) -> Dict[str, float]:
    """Compute statistics about similarity distributions.

    Args:
        vision_embs: Vision embeddings (N, D)
        text_embs: Text embeddings (N, D)

    Returns:
        Dict with positive/negative similarity statistics
    """
    # Normalize
    vision_embs = F.normalize(vision_embs, p=2, dim=-1)
    text_embs = F.normalize(text_embs, p=2, dim=-1)

    # Compute similarities
    sims = torch.matmul(vision_embs, text_embs.t())
    N = sims.size(0)

    # Positive pairs (diagonal)
    pos_sims = sims.diag()

    # Negative pairs (off-diagonal)
    mask = torch.eye(N, device=sims.device).bool()
    neg_sims = sims.masked_select(~mask)

    return {
        'pos_mean': pos_sims.mean().item(),
        'pos_std': pos_sims.std().item(),
        'pos_min': pos_sims.min().item(),
        'pos_max': pos_sims.max().item(),
        'neg_mean': neg_sims.mean().item(),
        'neg_std': neg_sims.std().item(),
        'neg_min': neg_sims.min().item(),
        'neg_max': neg_sims.max().item(),
        'separation': (pos_sims.mean() - neg_sims.mean()).item(),
    }
