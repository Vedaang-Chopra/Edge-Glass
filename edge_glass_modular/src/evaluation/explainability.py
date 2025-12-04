"""Explainability and interpretability analysis for aligned models."""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from pathlib import Path


class ExplainabilityAnalyzer:
    """Analyzer for model explainability and interpretability.

    Provides:
    - Embedding space visualization (PCA, t-SNE)
    - Attention analysis (if model has attention)
    - Similarity heatmaps
    - Feature importance analysis
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
    ):
        """Initialize analyzer.

        Args:
            model: Model to analyze
            device: Device for computation
        """
        self.model = model
        self.device = device
        self.model.eval()

    def reduce_dimensions(
        self,
        embeddings: np.ndarray,
        method: str = "pca",
        n_components: int = 2,
    ) -> np.ndarray:
        """Reduce embedding dimensions for visualization.

        Args:
            embeddings: High-dimensional embeddings (N, D)
            method: 'pca' or 'tsne'
            n_components: Number of output dimensions

        Returns:
            Reduced embeddings (N, n_components)
        """
        if method == "pca":
            reducer = PCA(n_components=n_components)
        elif method == "tsne":
            reducer = TSNE(n_components=n_components, random_state=42, perplexity=30)
        else:
            raise ValueError(f"Unknown method: {method}")

        return reducer.fit_transform(embeddings)

    def analyze_embedding_space(
        self,
        vision_embs: np.ndarray,
        text_embs: np.ndarray,
        method: str = "pca",
        n_samples: int = 500,
    ) -> Dict[str, np.ndarray]:
        """Analyze and visualize embedding space.

        Args:
            vision_embs: Vision embeddings (N, D)
            text_embs: Text embeddings (N, D)
            method: Dimensionality reduction method
            n_samples: Number of samples to analyze

        Returns:
            Dict with 2D projections
        """
        # Subsample if needed
        if len(vision_embs) > n_samples:
            indices = np.random.choice(len(vision_embs), n_samples, replace=False)
            vision_embs = vision_embs[indices]
            text_embs = text_embs[indices]

        # Combine and reduce
        all_embs = np.vstack([vision_embs, text_embs])
        all_embs_2d = self.reduce_dimensions(all_embs, method=method)

        # Split back
        vision_embs_2d = all_embs_2d[:len(vision_embs)]
        text_embs_2d = all_embs_2d[len(vision_embs):]

        return {
            'vision': vision_embs_2d,
            'text': text_embs_2d,
            'method': method,
        }

    def compute_feature_norms(
        self,
        embeddings: torch.Tensor,
    ) -> Dict[str, float]:
        """Analyze feature magnitudes.

        Args:
            embeddings: Embeddings to analyze (N, D)

        Returns:
            Dict with norm statistics
        """
        norms = embeddings.norm(dim=1)

        return {
            'mean': norms.mean().item(),
            'std': norms.std().item(),
            'min': norms.min().item(),
            'max': norms.max().item(),
        }

    def analyze_dimension_importance(
        self,
        embeddings: torch.Tensor,
        top_k: int = 20,
    ) -> Dict[str, np.ndarray]:
        """Analyze which embedding dimensions are most important.

        Args:
            embeddings: Embeddings to analyze (N, D)
            top_k: Number of top dimensions to return

        Returns:
            Dict with dimension importance metrics
        """
        # Variance per dimension
        variance = embeddings.var(dim=0).cpu().numpy()

        # Mean absolute value per dimension
        mean_abs = embeddings.abs().mean(dim=0).cpu().numpy()

        # Top-k most important dimensions by variance
        top_variance_dims = np.argsort(variance)[-top_k:][::-1]

        # Top-k most important dimensions by mean absolute value
        top_meanabs_dims = np.argsort(mean_abs)[-top_k:][::-1]

        return {
            'variance': variance,
            'mean_abs': mean_abs,
            'top_variance_dims': top_variance_dims,
            'top_meanabs_dims': top_meanabs_dims,
        }

    @torch.no_grad()
    def compute_retrieval_attention(
        self,
        query_emb: torch.Tensor,
        key_embs: torch.Tensor,
        top_k: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute attention scores for retrieval.

        Args:
            query_emb: Single query embedding (D,) or (1, D)
            key_embs: Key embeddings (N, D)
            top_k: Number of top matches to return

        Returns:
            (indices, scores) for top-k matches
        """
        if query_emb.dim() == 1:
            query_emb = query_emb.unsqueeze(0)

        # Normalize
        query_emb = torch.nn.functional.normalize(query_emb, p=2, dim=-1)
        key_embs = torch.nn.functional.normalize(key_embs, p=2, dim=-1)

        # Compute similarities
        similarities = torch.matmul(query_emb, key_embs.t()).squeeze(0)

        # Get top-k
        scores, indices = similarities.topk(top_k)

        return indices.cpu().numpy(), scores.cpu().numpy()

    def analyze_modality_separation(
        self,
        vision_embs: torch.Tensor,
        text_embs: torch.Tensor,
    ) -> Dict[str, float]:
        """Analyze how well modalities are separated in embedding space.

        Args:
            vision_embs: Vision embeddings (N, D)
            text_embs: Text embeddings (N, D)

        Returns:
            Dict with separation metrics
        """
        # Normalize
        vision_embs = torch.nn.functional.normalize(vision_embs, p=2, dim=-1)
        text_embs = torch.nn.functional.normalize(text_embs, p=2, dim=-1)

        # Within-modality similarities
        vision_sim = torch.matmul(vision_embs, vision_embs.t())
        text_sim = torch.matmul(text_embs, text_embs.t())

        # Remove diagonal (self-similarity)
        N = vision_embs.size(0)
        mask = ~torch.eye(N, dtype=torch.bool, device=vision_embs.device)

        within_vision = vision_sim[mask].mean().item()
        within_text = text_sim[mask].mean().item()

        # Cross-modality similarity (positive pairs)
        cross_sim = torch.sum(vision_embs * text_embs, dim=1).mean().item()

        return {
            'within_vision': within_vision,
            'within_text': within_text,
            'cross_modal': cross_sim,
            'vision_text_gap': abs(within_vision - cross_sim),
        }

    def generate_explainability_report(
        self,
        vision_embs: torch.Tensor,
        text_embs: torch.Tensor,
        save_dir: Optional[Path] = None,
    ) -> Dict:
        """Generate comprehensive explainability report.

        Args:
            vision_embs: Vision embeddings (N, D)
            text_embs: Text embeddings (N, D)
            save_dir: Optional directory to save report

        Returns:
            Dict with all analysis results
        """
        print("="*60)
        print("GENERATING EXPLAINABILITY REPORT")
        print("="*60)

        report = {}

        # 1. Feature norms
        print("\n1. Analyzing feature norms...")
        report['vision_norms'] = self.compute_feature_norms(vision_embs)
        report['text_norms'] = self.compute_feature_norms(text_embs)

        print(f"   Vision: {report['vision_norms']['mean']:.3f} ± {report['vision_norms']['std']:.3f}")
        print(f"   Text:   {report['text_norms']['mean']:.3f} ± {report['text_norms']['std']:.3f}")

        # 2. Dimension importance
        print("\n2. Analyzing dimension importance...")
        report['vision_dims'] = self.analyze_dimension_importance(vision_embs)
        report['text_dims'] = self.analyze_dimension_importance(text_embs)

        print(f"   Top-3 vision dims (by variance): {report['vision_dims']['top_variance_dims'][:3]}")
        print(f"   Top-3 text dims (by variance):   {report['text_dims']['top_variance_dims'][:3]}")

        # 3. Modality separation
        print("\n3. Analyzing modality separation...")
        report['separation'] = self.analyze_modality_separation(vision_embs, text_embs)

        print(f"   Within-vision sim:  {report['separation']['within_vision']:.3f}")
        print(f"   Within-text sim:    {report['separation']['within_text']:.3f}")
        print(f"   Cross-modal sim:    {report['separation']['cross_modal']:.3f}")

        # 4. Embedding space analysis
        print("\n4. Analyzing embedding space...")
        report['pca_projection'] = self.analyze_embedding_space(
            vision_embs.cpu().numpy(),
            text_embs.cpu().numpy(),
            method='pca',
        )

        print("   PCA projection computed")

        print("\n" + "="*60)
        print("REPORT COMPLETE")
        print("="*60)

        # Save if requested
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

            import json
            report_dict = {
                'vision_norms': report['vision_norms'],
                'text_norms': report['text_norms'],
                'separation': report['separation'],
            }

            with open(save_dir / 'explainability_report.json', 'w') as f:
                json.dump(report_dict, f, indent=2)

            print(f"\nReport saved to {save_dir}")

        return report
