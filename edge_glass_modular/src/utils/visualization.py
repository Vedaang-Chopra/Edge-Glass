"""Visualization utilities for training and explainability."""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


class TrainingVisualizer:
    """Visualizer for training metrics and explainability.

    Handles:
    - Training/validation loss curves
    - Learning rate schedules
    - Embedding space visualization
    - Retrieval metrics
    - Attention heatmaps
    """

    def __init__(self, save_dir: str | Path, style: str = "seaborn-v0_8-darkgrid"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        plt.style.use(style)

    def plot_training_curves(
        self,
        history: Dict[str, List[float]],
        save_name: str = "training_curves.png",
        dpi: int = 150,
    ):
        """Plot training and validation loss curves.

        Args:
            history: Dict with keys like 'train_loss', 'val_loss', 'train_loss_clip', etc.
            save_name: Filename to save the plot
            dpi: DPI for saved figure
        """
        # Determine number of subplots needed
        metrics = list(history.keys())
        loss_metrics = [k for k in metrics if 'loss' in k]
        other_metrics = [k for k in metrics if 'loss' not in k]

        n_plots = 1 if loss_metrics else 0
        if other_metrics:
            n_plots += 1

        fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 5))
        if n_plots == 1:
            axes = [axes]

        # Plot losses
        if loss_metrics:
            ax = axes[0]
            for metric in loss_metrics:
                if history[metric]:
                    ax.plot(history[metric], label=metric.replace('_', ' ').title(), marker='o', markersize=4)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title('Training & Validation Losses')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Plot other metrics (e.g., retrieval)
        if other_metrics and n_plots > 1:
            ax = axes[1]
            for metric in other_metrics:
                if history[metric]:
                    values = [v * 100 if v < 1.5 else v for v in history[metric]]  # Convert to %
                    ax.plot(values, label=metric.replace('_', ' ').title(), marker='s', markersize=4)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Recall (%)' if 'r@' in other_metrics[0].lower() else 'Metric')
            ax.set_title('Retrieval Metrics')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=dpi, bbox_inches='tight')
        plt.close()

    def plot_loss_components(
        self,
        history: Dict[str, List[float]],
        save_name: str = "loss_components.png",
        dpi: int = 150,
    ):
        """Plot individual loss components (CLIP, MRL, total).

        Args:
            history: Dict with loss components
            save_name: Filename to save
            dpi: DPI for saved figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Training losses
        ax = axes[0]
        if 'train_loss_total' in history and history['train_loss_total']:
            ax.plot(history['train_loss_total'], label='Total Loss', linewidth=2, marker='o')
        if 'train_loss_clip' in history and history['train_loss_clip']:
            ax.plot(history['train_loss_clip'], label='CLIP Loss', marker='s', alpha=0.8)
        if 'train_loss_mrl' in history and history['train_loss_mrl']:
            ax.plot(history['train_loss_mrl'], label='MRL Loss', marker='^', alpha=0.8)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss Components')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Validation losses
        ax = axes[1]
        if 'val_loss_total' in history and history['val_loss_total']:
            ax.plot(history['val_loss_total'], label='Total Loss', linewidth=2, marker='o')
        if 'val_loss_clip' in history and history['val_loss_clip']:
            ax.plot(history['val_loss_clip'], label='CLIP Loss', marker='s', alpha=0.8)
        if 'val_loss_mrl' in history and history['val_loss_mrl']:
            ax.plot(history['val_loss_mrl'], label='MRL Loss', marker='^', alpha=0.8)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Validation Loss Components')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=dpi, bbox_inches='tight')
        plt.close()

    def plot_lr_schedule(
        self,
        lr_history: List[float],
        save_name: str = "lr_schedule.png",
        dpi: int = 150,
    ):
        """Plot learning rate schedule.

        Args:
            lr_history: List of learning rates per step
            save_name: Filename to save
            dpi: DPI for saved figure
        """
        plt.figure(figsize=(10, 5))
        plt.plot(lr_history, linewidth=2)
        plt.xlabel('Training Step')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule (Warmup + Cosine Decay)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=dpi, bbox_inches='tight')
        plt.close()

    def plot_embedding_space(
        self,
        vision_embs: np.ndarray,
        text_embs: np.ndarray,
        method: str = "pca",
        n_samples: int = 500,
        save_name: str = "embedding_space.png",
        dpi: int = 150,
    ):
        """Visualize vision-text embedding alignment in 2D.

        Args:
            vision_embs: Vision embeddings (N, dim)
            text_embs: Text embeddings (N, dim)
            method: Dimensionality reduction method ("pca" or "tsne")
            n_samples: Number of samples to plot
            save_name: Filename to save
            dpi: DPI for saved figure
        """
        # Subsample if needed
        if len(vision_embs) > n_samples:
            indices = np.random.choice(len(vision_embs), n_samples, replace=False)
            vision_embs = vision_embs[indices]
            text_embs = text_embs[indices]

        # Combine and reduce dimensions
        all_embs = np.vstack([vision_embs, text_embs])

        if method == "pca":
            reducer = PCA(n_components=2)
        else:  # tsne
            reducer = TSNE(n_components=2, random_state=42)

        all_embs_2d = reducer.fit_transform(all_embs)

        vision_embs_2d = all_embs_2d[:len(vision_embs)]
        text_embs_2d = all_embs_2d[len(vision_embs):]

        # Plot
        plt.figure(figsize=(12, 10))
        plt.scatter(vision_embs_2d[:, 0], vision_embs_2d[:, 1],
                   c='blue', alpha=0.6, s=50, label='Vision', edgecolors='k', linewidths=0.5)
        plt.scatter(text_embs_2d[:, 0], text_embs_2d[:, 1],
                   c='red', alpha=0.6, s=50, label='Text', edgecolors='k', linewidths=0.5)

        # Draw lines connecting matching pairs (first 50)
        for i in range(min(50, len(vision_embs_2d))):
            plt.plot([vision_embs_2d[i, 0], text_embs_2d[i, 0]],
                    [vision_embs_2d[i, 1], text_embs_2d[i, 1]],
                    'gray', alpha=0.2, linewidth=0.5)

        method_name = "PCA" if method == "pca" else "t-SNE"
        plt.xlabel(f'{method_name} Component 1')
        plt.ylabel(f'{method_name} Component 2')
        plt.title(f'Vision-Text Embedding Space ({method_name} Projection)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(self.save_dir / save_name, dpi=dpi, bbox_inches='tight')
        plt.close()

    def plot_mrl_performance(
        self,
        mrl_dims: List[int],
        mrl_scores: Dict[int, float],
        metric_name: str = "Recall@1",
        save_name: str = "mrl_performance.png",
        dpi: int = 150,
    ):
        """Plot MRL performance at different dimensions.

        Args:
            mrl_dims: List of MRL dimensions
            mrl_scores: Dict mapping dim -> score
            metric_name: Name of the metric being plotted
            save_name: Filename to save
            dpi: DPI for saved figure
        """
        dims = sorted(mrl_dims)
        scores = [mrl_scores.get(d, 0) * 100 for d in dims]

        plt.figure(figsize=(10, 6))
        plt.plot(dims, scores, marker='o', linewidth=2, markersize=8)
        plt.xlabel('MRL Dimension')
        plt.ylabel(f'{metric_name} (%)')
        plt.title(f'{metric_name} vs MRL Embedding Dimension')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=dpi, bbox_inches='tight')
        plt.close()

    def plot_similarity_matrix(
        self,
        vision_embs: np.ndarray,
        text_embs: np.ndarray,
        n_samples: int = 50,
        save_name: str = "similarity_matrix.png",
        dpi: int = 150,
    ):
        """Plot vision-text similarity matrix heatmap.

        Args:
            vision_embs: Vision embeddings (N, dim)
            text_embs: Text embeddings (N, dim)
            n_samples: Number of samples to include
            save_name: Filename to save
            dpi: DPI for saved figure
        """
        # Subsample
        if len(vision_embs) > n_samples:
            indices = np.random.choice(len(vision_embs), n_samples, replace=False)
            vision_embs = vision_embs[indices]
            text_embs = text_embs[indices]

        # Compute similarity matrix
        similarity = np.matmul(vision_embs, text_embs.T)

        plt.figure(figsize=(10, 8))
        sns.heatmap(similarity, cmap='viridis', center=0,
                    xticklabels=False, yticklabels=False,
                    cbar_kws={'label': 'Cosine Similarity'})
        plt.xlabel('Text Embeddings')
        plt.ylabel('Vision Embeddings')
        plt.title('Vision-Text Similarity Matrix')
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=dpi, bbox_inches='tight')
        plt.close()

    def save_metrics_table(
        self,
        metrics: Dict[str, float],
        save_name: str = "metrics.csv",
    ):
        """Save metrics to CSV table.

        Args:
            metrics: Dict of metric_name -> value
            save_name: Filename to save
        """
        df = pd.DataFrame([metrics])
        df.to_csv(self.save_dir / save_name, index=False)

    def plot_rank_histogram(
        self,
        ranks: np.ndarray,
        title: str = "Retrieval Rank Histogram",
        max_rank: int = 100,
        save_name: str = "rank_histogram.png",
        dpi: int = 150,
    ):
        """Plot histogram of retrieval ranks.

        Args:
            ranks: Array of ranks (0-indexed)
            title: Plot title
            max_rank: Maximum rank to display
            save_name: Filename to save
            dpi: DPI for saved figure
        """
        ranks = np.clip(ranks, 0, max_rank)

        plt.figure(figsize=(10, 6))
        plt.hist(ranks + 1, bins=min(max_rank, 50), edgecolor='black')
        plt.xlabel('Rank (1-indexed)')
        plt.ylabel('Frequency (log scale)')
        plt.yscale('log')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=dpi, bbox_inches='tight')
        plt.close()

    def plot_rank_cdf(
        self,
        ranks: np.ndarray,
        title: str = "Retrieval Rank CDF",
        max_k: int = 50,
        save_name: str = "rank_cdf.png",
        dpi: int = 150,
    ):
        """Plot cumulative distribution of ranks.

        Args:
            ranks: Array of ranks (0-indexed)
            title: Plot title
            max_k: Maximum K to display
            save_name: Filename to save
            dpi: DPI for saved figure
        """
        ranks_1indexed = ranks + 1
        ks = np.arange(1, max_k + 1)
        cdf = [(ranks_1indexed <= k).mean() * 100.0 for k in ks]

        plt.figure(figsize=(10, 6))
        plt.plot(ks, cdf, marker='o', linewidth=2, markersize=4)
        plt.xlabel('K')
        plt.ylabel('P(rank ≤ K) [%]')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=dpi, bbox_inches='tight')
        plt.close()

    def plot_similarity_distributions(
        self,
        positive_sims: np.ndarray,
        negative_sims: np.ndarray,
        title: str = "Similarity Distributions",
        save_name: str = "similarity_distributions.png",
        dpi: int = 150,
    ):
        """Plot distributions of positive vs negative similarities.

        Args:
            positive_sims: Similarities for positive pairs
            negative_sims: Similarities for negative pairs
            title: Plot title
            save_name: Filename to save
            dpi: DPI for saved figure
        """
        plt.figure(figsize=(10, 6))
        plt.hist(negative_sims, bins=50, alpha=0.6, label='Negative Pairs', color='red')
        plt.hist(positive_sims, bins=50, alpha=0.6, label='Positive Pairs', color='green')
        plt.xlabel('Cosine Similarity')
        plt.ylabel('Frequency')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=dpi, bbox_inches='tight')
        plt.close()

    def plot_mrl_curves(
        self,
        mrl_results: Dict[int, Dict],
        metric: str = 'r_at_1',
        title: str = "MRL Performance vs Dimension",
        save_name: str = "mrl_curves.png",
        dpi: int = 150,
    ):
        """Plot MRL performance curves across dimensions.

        Args:
            mrl_results: Dict mapping dim -> {'i2t': metrics, 't2i': metrics}
            metric: Metric name to plot (e.g., 'r_at_1', 'r_at_5')
            title: Plot title
            save_name: Filename to save
            dpi: DPI for saved figure
        """
        dims = sorted(mrl_results.keys())
        i2t_scores = [getattr(mrl_results[d]['i2t'], metric) for d in dims]
        t2i_scores = [getattr(mrl_results[d]['t2i'], metric) for d in dims]

        plt.figure(figsize=(10, 6))
        plt.plot(dims, i2t_scores, marker='o', linewidth=2, label='Image→Text', markersize=8)
        plt.plot(dims, t2i_scores, marker='s', linewidth=2, label='Text→Image', markersize=8)
        plt.xlabel('MRL Dimension')
        plt.ylabel(f'{metric.replace("_", " ").title()} (%)')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xscale('log')
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=dpi, bbox_inches='tight')
        plt.close()

    def plot_recall_at_k(
        self,
        metrics: Dict[str, float],
        direction: str = 'i2t',
        title: Optional[str] = None,
        save_name: str = "recall_at_k.png",
        dpi: int = 150,
    ):
        """Plot Recall@K curve.

        Args:
            metrics: Metrics dict with r_at_1, r_at_5, etc.
            direction: 'i2t' or 't2i'
            title: Plot title (auto-generated if None)
            save_name: Filename to save
            dpi: DPI for saved figure
        """
        ks = [1, 5, 10, 50]
        recalls = [
            metrics.get(f'r_at_{k}', 0.0) if isinstance(metrics, dict)
            else getattr(metrics, f'r_at_{k}', 0.0)
            for k in ks
        ]

        if title is None:
            title = f'Recall@K ({direction.upper()})'

        plt.figure(figsize=(10, 6))
        plt.plot(ks, recalls, marker='o', linewidth=2, markersize=8)
        plt.xlabel('K')
        plt.ylabel('Recall (%)')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=dpi, bbox_inches='tight')
        plt.close()
