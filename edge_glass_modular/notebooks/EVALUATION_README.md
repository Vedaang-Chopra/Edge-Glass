# Alignment Model Evaluation System

This directory contains a comprehensive, modular evaluation system for vision-text aligned models trained with the Edge Glass framework.

## Overview

The evaluation system provides:

1. **Retrieval Benchmarks**: Standard image↔text retrieval evaluation
2. **MRL Performance Analysis**: Performance across different embedding dimensions
3. **Explainability Tools**: Understanding model behavior and alignment quality
4. **Modular Architecture**: Easy to extend and customize

## Files

### Notebooks

- **[04_alignment_evaluation.ipynb](04_alignment_evaluation.ipynb)**: Main evaluation notebook
  - Comprehensive evaluation of aligned models
  - Works with both MLP and Perceiver architectures
  - Generates visualizations and reports
  - Supports model comparison

### Source Code

Located in `../src/evaluation/`:

- **metrics.py**: Retrieval metrics computation
  - `RetrievalMetrics`: Dataclass for metrics
  - `compute_retrieval_metrics()`: Compute R@K, mAP, NDCG, etc.
  - `compute_mrl_performance()`: Evaluate at different dimensions
  - `compute_similarity_statistics()`: Analyze similarity distributions

- **benchmark.py**: Benchmarking framework
  - `AlignmentBenchmark`: Main benchmark class
  - Automated evaluation pipeline
  - Results saving and logging

- **explainability.py**: Explainability analysis
  - `ExplainabilityAnalyzer`: Interpretability tools
  - Embedding space visualization
  - Dimension importance analysis
  - Modality separation metrics

### Configuration

- **[../configs/evaluation.yaml](../configs/evaluation.yaml)**: Evaluation configuration
  - Model checkpoints to evaluate
  - Dataset settings
  - Metrics configuration
  - Visualization options

## Quick Start

### 1. Basic Evaluation

```python
# Open 04_alignment_evaluation.ipynb
# Set MODEL_TYPE to "pixmo_mlp" or "perceiver_mrl"
# Run all cells
```

### 2. Custom Evaluation

```python
from evaluation import AlignmentBenchmark
from config import load_config

# Load model and config
model = MultimodalAlignmentModel(config)
# ... load checkpoint ...

# Create benchmark
benchmark = AlignmentBenchmark(
    model=model,
    device='cuda',
    mrl_dims=[128, 256, 512, 1024, 2048, 4096],
)

# Run evaluation
results = benchmark.run_full_evaluation(test_loader)

# Access results
print(results['retrieval']['i2t'].r_at_1)  # Image-to-text R@1
print(results['mrl'][512]['i2t'].r_at_5)   # R@5 at dim=512
```

## Metrics Explained

### Retrieval Metrics

1. **Recall@K (R@K)**: Percentage of queries where the correct match is in top-K results
   - R@1: Exact match accuracy
   - R@5, R@10, R@50: More relaxed matching

2. **Mean Rank (MR)**: Average rank of correct matches (lower is better)

3. **Median Rank (MedR)**: Median rank of correct matches (robust to outliers)

4. **Mean Average Precision@K (mAP@K)**: Quality of ranking within top-K
   - Formula: `1 / (rank + 1)` if rank < K, else 0

5. **Normalized Discounted Cumulative Gain@K (NDCG@K)**: Ranking quality with position discount
   - Formula: `1 / log2(rank + 2)` if rank < K, else 0

### Similarity Statistics

- **Positive mean/std**: Similarity for matching pairs (should be high)
- **Negative mean/std**: Similarity for non-matching pairs (should be low)
- **Separation**: Difference between positive and negative means (higher is better)

### MRL Metrics

- Performance at different embedding dimensions
- Trade-off between efficiency (lower dims) and accuracy (higher dims)
- Useful for deployment decisions

## Visualizations

The evaluation system generates:

### Retrieval Analysis
- **Rank Histograms**: Distribution of retrieval ranks
- **Rank CDFs**: Cumulative probability of finding match in top-K
- **Recall@K Curves**: Performance across different K values

### Similarity Analysis
- **Similarity Distributions**: Positive vs. negative pair similarities
- **Similarity Matrix**: Heatmap of pairwise similarities
- **Embedding Space**: 2D projection (PCA/t-SNE) of aligned embeddings

### MRL Analysis
- **MRL Curves**: Performance vs. embedding dimension
- **Dimension Trade-offs**: Accuracy vs. efficiency

## Explainability Features

### 1. Embedding Space Visualization

```python
from evaluation import ExplainabilityAnalyzer

analyzer = ExplainabilityAnalyzer(model)
projection = analyzer.analyze_embedding_space(
    vision_embs, text_embs, method='pca'
)
```

### 2. Dimension Importance

```python
importance = analyzer.analyze_dimension_importance(
    embeddings, top_k=20
)
print(importance['top_variance_dims'])  # Most important dimensions
```

### 3. Modality Separation

```python
separation = analyzer.analyze_modality_separation(
    vision_embs, text_embs
)
print(separation['cross_modal'])  # How well aligned
print(separation['within_vision'])  # Within-modality similarity
```

### 4. Retrieval Attention

```python
# Find top-K similar items for a query
indices, scores = analyzer.compute_retrieval_attention(
    query_emb=vision_embs[0],
    key_embs=text_embs,
    top_k=10,
)
```

## Customization

### Adding New Metrics

Edit `src/evaluation/metrics.py`:

```python
def compute_custom_metric(query_embs, key_embs):
    # Your metric computation
    return metric_value
```

### Adding New Visualizations

Edit `src/utils/visualization.py`:

```python
class TrainingVisualizer:
    def plot_custom_viz(self, data, **kwargs):
        # Your visualization code
        plt.savefig(self.save_dir / 'custom.png')
```

### Custom Benchmarks

Extend `AlignmentBenchmark`:

```python
class CustomBenchmark(AlignmentBenchmark):
    def evaluate_custom_task(self, dataloader):
        # Your evaluation logic
        return results
```

## Model Comparison

To compare multiple models:

1. Run evaluation for each model (change `MODEL_TYPE` in notebook)
2. Results are saved to separate directories
3. Use Section 19 of the notebook for automated comparison

Example output:
```
Metric               |        MLP |  Perceiver
---------------------------------------------
R@1                  |      45.20 |      52.30
R@5                  |      75.10 |      78.50
R@10                 |      85.40 |      88.20
mean_rank            |      12.50 |       9.80
```

## Integration with Training

The evaluation system integrates seamlessly with training:

```python
from training.improved_trainer import ImprovedMultimodalTrainer

# During training, validation metrics are computed
trainer = ImprovedMultimodalTrainer(...)
history = trainer.train()

# After training, run full evaluation
from evaluation import AlignmentBenchmark
benchmark = AlignmentBenchmark(model)
results = benchmark.run_full_evaluation(test_loader)
```

## Weights & Biases Integration

The system logs metrics to W&B:

```yaml
# In evaluation.yaml
logging:
  use_wandb: true
  wandb_project: "edge_glass_alignment"
  run_name: "alignment_eval"
```

Logged metrics:
- All retrieval metrics (R@K, mAP, NDCG)
- MRL performance per dimension
- Similarity statistics
- Visualizations (as W&B Images)

## Output Structure

```
results/evaluation/
├── pixmo_mlp/
│   ├── metrics.json          # All metrics
│   ├── embeddings.pt         # Saved embeddings
│   ├── summary.json          # Summary report
│   └── explainability_report.json
└── perceiver_mrl/
    └── ... (same structure)

outputs/evaluation/
├── pixmo_mlp/
│   ├── i2t_rank_histogram.png
│   ├── rank_cdf.png
│   ├── similarity_distributions.png
│   ├── embedding_space.png
│   ├── mrl_r_at_1.png
│   └── ... (more visualizations)
└── perceiver_mrl/
    └── ... (same structure)
```

## Tips & Best Practices

### 1. Memory Management
- For large datasets, set `max_samples` in config
- Use `max_batches` parameter in evaluation
- Clear GPU cache between runs: `torch.cuda.empty_cache()`

### 2. Reproducibility
- Set random seeds (done automatically in notebook)
- Use same test set across evaluations
- Save full config with results

### 3. Performance Optimization
- Use larger batch sizes for evaluation (no gradients)
- Enable `pin_memory=True` for faster data loading
- Use multiple workers in DataLoader

### 4. Debugging
- Start with small subset (`max_samples=100`)
- Check embedding norms (should be ~1.0 after normalization)
- Verify positive similarities > negative similarities

## Troubleshooting

### Low R@1 Performance
- Check if embeddings are normalized
- Verify positive/negative similarity separation
- Inspect embedding space visualization
- Check for training issues (look at training curves)

### Memory Errors
- Reduce `batch_size` in config
- Set `max_samples` to limit dataset size
- Use smaller `n_samples` for visualizations

### Slow Evaluation
- Increase `num_workers` in DataLoader
- Use larger `batch_size`
- Disable some visualizations
- Skip MRL evaluation if not needed

## Citation

If you use this evaluation system, please cite:

```bibtex
@software{edge_glass_eval,
  title={Edge Glass Alignment Evaluation System},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo}
}
```

## Next Steps

After evaluation:

1. **Analyze Results**: Review all visualizations and metrics
2. **Identify Issues**: Use explainability tools to understand failures
3. **Iterate**: Adjust model architecture or training based on insights
4. **Compare**: Benchmark against other models or published results
5. **Deploy**: Use MRL analysis to choose optimal dimension for deployment

## References

- CLIP: Learning Transferable Visual Models From Natural Language Supervision
- Matryoshka Representation Learning
- Freeze-Align: Vision-Language Alignment
- ImageBind: One Embedding Space To Bind Them All
- Unified-IO 2: Scaling Autoregressive Multimodal Models

## Support

For issues or questions:
- Check existing notebooks for examples
- Review source code documentation
- Open an issue on GitHub
