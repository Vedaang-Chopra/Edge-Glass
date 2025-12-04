"""Benchmark suite for alignment evaluation."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional, List
from tqdm.auto import tqdm
from pathlib import Path

from .metrics import (
    RetrievalMetrics,
    compute_retrieval_metrics,
    compute_mrl_performance,
    compute_similarity_statistics,
)


class AlignmentBenchmark:
    """Comprehensive benchmark for aligned models.

    Evaluates:
    - Image-to-text and text-to-image retrieval
    - MRL performance across dimensions
    - Similarity statistics
    - Cross-modal alignment quality
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        mrl_dims: Optional[List[int]] = None,
    ):
        """Initialize benchmark.

        Args:
            model: Aligned model with encode_vision() and encode_text() methods
            device: Device to run evaluation on
            mrl_dims: List of MRL dimensions to evaluate (if applicable)
        """
        self.model = model
        self.device = device
        self.mrl_dims = mrl_dims or []
        self.model.eval()

    @torch.no_grad()
    def collect_embeddings(
        self,
        dataloader: DataLoader,
        max_batches: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Collect vision and text embeddings from dataloader.

        Args:
            dataloader: DataLoader yielding {'image', 'text'} batches
            max_batches: Optional limit on number of batches

        Returns:
            (vision_embs, text_embs) tensors of shape (N, D)
        """
        vision_embs = []
        text_embs = []

        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Collecting embeddings")):
            if max_batches and batch_idx >= max_batches:
                break

            images = batch['image'].to(self.device)
            texts = batch['text']

            # Filter out dropped texts (from text_dropout)
            valid_indices = [i for i, t in enumerate(texts) if t]
            if not valid_indices:
                continue

            images = images[valid_indices]
            texts = [texts[i] for i in valid_indices]

            # Get embeddings
            outputs = self.model(images=images, texts=texts, return_embeddings=True)

            vision_embs.append(outputs.vision_emb.cpu())
            text_embs.append(outputs.text_emb.cpu())

        vision_embs = torch.cat(vision_embs, dim=0)
        text_embs = torch.cat(text_embs, dim=0)

        return vision_embs, text_embs

    def evaluate_retrieval(
        self,
        vision_embs: torch.Tensor,
        text_embs: torch.Tensor,
    ) -> Dict[str, RetrievalMetrics]:
        """Evaluate retrieval performance.

        Args:
            vision_embs: Vision embeddings (N, D)
            text_embs: Text embeddings (N, D)

        Returns:
            Dict with 'i2t' and 't2i' RetrievalMetrics
        """
        # Image-to-text retrieval
        i2t_metrics = compute_retrieval_metrics(vision_embs, text_embs)

        # Text-to-image retrieval
        t2i_metrics = compute_retrieval_metrics(text_embs, vision_embs)

        return {
            'i2t': i2t_metrics,
            't2i': t2i_metrics,
        }

    def evaluate_mrl(
        self,
        vision_embs: torch.Tensor,
        text_embs: torch.Tensor,
    ) -> Optional[Dict[int, Dict[str, RetrievalMetrics]]]:
        """Evaluate MRL performance across dimensions.

        Args:
            vision_embs: Vision embeddings (N, D)
            text_embs: Text embeddings (N, D)

        Returns:
            Dict mapping dimension -> {'i2t': metrics, 't2i': metrics}
            or None if no MRL dimensions specified
        """
        if not self.mrl_dims:
            return None

        return compute_mrl_performance(vision_embs, text_embs, self.mrl_dims)

    def evaluate_similarity(
        self,
        vision_embs: torch.Tensor,
        text_embs: torch.Tensor,
    ) -> Dict[str, float]:
        """Evaluate similarity statistics.

        Args:
            vision_embs: Vision embeddings (N, D)
            text_embs: Text embeddings (N, D)

        Returns:
            Dict with similarity statistics
        """
        return compute_similarity_statistics(vision_embs, text_embs)

    def run_full_evaluation(
        self,
        dataloader: DataLoader,
        max_batches: Optional[int] = None,
    ) -> Dict:
        """Run comprehensive evaluation.

        Args:
            dataloader: DataLoader for evaluation
            max_batches: Optional limit on batches

        Returns:
            Dict with all evaluation results
        """
        print("="*60)
        print("RUNNING ALIGNMENT EVALUATION")
        print("="*60)

        # Collect embeddings
        print("\n1. Collecting embeddings...")
        vision_embs, text_embs = self.collect_embeddings(dataloader, max_batches)
        print(f"   Collected {len(vision_embs)} samples")
        print(f"   Embedding dimension: {vision_embs.shape[1]}")

        # Retrieval evaluation
        print("\n2. Evaluating retrieval performance...")
        retrieval_metrics = self.evaluate_retrieval(vision_embs, text_embs)

        print(f"   Image-to-Text:")
        print(f"     R@1:  {retrieval_metrics['i2t'].r_at_1:.2f}%")
        print(f"     R@5:  {retrieval_metrics['i2t'].r_at_5:.2f}%")
        print(f"     R@10: {retrieval_metrics['i2t'].r_at_10:.2f}%")

        print(f"   Text-to-Image:")
        print(f"     R@1:  {retrieval_metrics['t2i'].r_at_1:.2f}%")
        print(f"     R@5:  {retrieval_metrics['t2i'].r_at_5:.2f}%")
        print(f"     R@10: {retrieval_metrics['t2i'].r_at_10:.2f}%")

        # MRL evaluation
        mrl_results = None
        if self.mrl_dims:
            print(f"\n3. Evaluating MRL performance across {len(self.mrl_dims)} dimensions...")
            mrl_results = self.evaluate_mrl(vision_embs, text_embs)

            for dim in sorted(self.mrl_dims):
                metrics = mrl_results[dim]['i2t']
                print(f"   Dim {dim:4d}: R@1 = {metrics.r_at_1:.2f}%")

        # Similarity statistics
        print("\n4. Computing similarity statistics...")
        sim_stats = self.evaluate_similarity(vision_embs, text_embs)
        print(f"   Positive pairs: {sim_stats['pos_mean']:.3f} ± {sim_stats['pos_std']:.3f}")
        print(f"   Negative pairs: {sim_stats['neg_mean']:.3f} ± {sim_stats['neg_std']:.3f}")
        print(f"   Separation: {sim_stats['separation']:.3f}")

        print("\n" + "="*60)
        print("EVALUATION COMPLETE")
        print("="*60)

        return {
            'embeddings': {
                'vision': vision_embs,
                'text': text_embs,
            },
            'retrieval': retrieval_metrics,
            'mrl': mrl_results,
            'similarity': sim_stats,
        }

    def save_results(
        self,
        results: Dict,
        save_dir: Path,
    ):
        """Save evaluation results.

        Args:
            results: Results from run_full_evaluation()
            save_dir: Directory to save results
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save retrieval metrics
        import json

        metrics_dict = {
            'i2t': results['retrieval']['i2t'].to_dict(),
            't2i': results['retrieval']['t2i'].to_dict(),
            'similarity': results['similarity'],
        }

        # Add MRL metrics if available
        if results['mrl']:
            metrics_dict['mrl'] = {}
            for dim, metrics in results['mrl'].items():
                metrics_dict['mrl'][dim] = {
                    'i2t': metrics['i2t'].to_dict(),
                    't2i': metrics['t2i'].to_dict(),
                }

        with open(save_dir / 'metrics.json', 'w') as f:
            json.dump(metrics_dict, f, indent=2)

        # Save embeddings
        torch.save({
            'vision': results['embeddings']['vision'],
            'text': results['embeddings']['text'],
        }, save_dir / 'embeddings.pt')

        print(f"\nResults saved to {save_dir}")
