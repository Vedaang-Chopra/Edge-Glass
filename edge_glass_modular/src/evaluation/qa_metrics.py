
import string
from collections import Counter
from typing import Dict, List, Tuple

def normalize_answer(s: str) -> str:
    """Normalize answer for evaluation."""
    s = ''.join(ch for ch in s if ch not in string.punctuation)
    s = s.lower().strip()
    s = ' '.join([w for w in s.split() if w not in {'a', 'an', 'the'}])
    return s

def compute_exact_match(pred: str, target: str) -> float:
    """Compute exact match score."""
    return float(normalize_answer(pred) == normalize_answer(target))

def compute_f1(pred: str, target: str) -> float:
    """Compute F1 score."""
    pred_tokens = normalize_answer(pred).split()
    target_tokens = normalize_answer(target).split()
    
    if len(pred_tokens) == 0 or len(target_tokens) == 0:
        return float(pred_tokens == target_tokens)
    
    common = Counter(pred_tokens) & Counter(target_tokens)
    num_common = sum(common.values())
    
    if num_common == 0:
        return 0.0
    
    precision = num_common / len(pred_tokens)
    recall = num_common / len(target_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    
    return f1

def evaluate_qa_metrics(predictions: List[str], targets: List[str]) -> Dict[str, float]:
    """Compute aggregate QA metrics for a list of predictions and targets."""
    em_scores = [compute_exact_match(p, t) for p, t in zip(predictions, targets)]
    f1_scores = [compute_f1(p, t) for p, t in zip(predictions, targets)]
    
    return {
        "exact_match": sum(em_scores) / len(em_scores) * 100.0,
        "f1": sum(f1_scores) / len(f1_scores) * 100.0
    }

def compute_bleu(pred: str, target: str) -> float:
    """
    Compute a simplified BLEU score (BLEU-4) without external dependencies.
    Rough approximation suitable for monitoring.
    """
    pred_tokens = normalize_answer(pred).split()
    target_tokens = normalize_answer(target).split()
    
    if len(pred_tokens) == 0:
        return 0.0
    
    # Calculate precision for n-grams 1 to 4
    precisions = []
    for n in range(1, 5):
        if len(pred_tokens) < n:
            precisions.append(0.0)
            continue
            
        pred_ngrams = [tuple(pred_tokens[i:i+n]) for i in range(len(pred_tokens)-n+1)]
        target_ngrams = [tuple(target_tokens[i:i+n]) for i in range(len(target_tokens)-n+1)]
        
        pred_counts = Counter(pred_ngrams)
        target_counts = Counter(target_ngrams)
        
        overlap = sum((pred_counts & target_counts).values())
        total = sum(pred_counts.values())
        
        precisions.append(overlap / total if total > 0 else 0.0)
        
    if any(p == 0.0 for p in precisions):
        return 0.0
        
    import math
    geo_mean = math.exp(sum(math.log(p) for p in precisions) / 4)
    
    # Brevity penalty
    bp = 1.0
    if len(pred_tokens) < len(target_tokens):
        bp = math.exp(1 - len(target_tokens) / len(pred_tokens))
        
    return geo_mean * bp

def compute_rouge_l(pred: str, target: str) -> float:
    """
    Compute ROUGE-L score (Longest Common Subsequence) without external dependencies.
    """
    pred_tokens = normalize_answer(pred).split()
    target_tokens = normalize_answer(target).split()
    
    if len(pred_tokens) == 0 or len(target_tokens) == 0:
        return 0.0
        
    m = len(pred_tokens)
    n = len(target_tokens)
    
    # DP table for LCS
    # Optimize space: we only need previous row
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_tokens[i-1] == target_tokens[j-1]:
                curr[j] = prev[j-1] + 1
            else:
                curr[j] = max(prev[j], curr[j-1])
        prev = list(curr)
        
    lcs_len = curr[n]
    
    if lcs_len == 0:
        return 0.0
        
    precision = lcs_len / m
    recall = lcs_len / n
    
    f1 = 2 * precision * recall / (precision + recall)
    return f1

def evaluate_qa_metrics(predictions: List[str], targets: List[str]) -> Dict[str, float]:
    """Compute aggregate QA metrics for a list of predictions and targets."""
    em_scores = [compute_exact_match(p, t) for p, t in zip(predictions, targets)]
    f1_scores = [compute_f1(p, t) for p, t in zip(predictions, targets)]
    bleu_scores = [compute_bleu(p, t) for p, t in zip(predictions, targets)]
    rouge_scores = [compute_rouge_l(p, t) for p, t in zip(predictions, targets)]
    
    return {
        "exact_match": sum(em_scores) / len(em_scores) * 100.0,
        "f1": sum(f1_scores) / len(f1_scores) * 100.0,
        "bleu": sum(bleu_scores) / len(bleu_scores) * 100.0,
        "rouge_l": sum(rouge_scores) / len(rouge_scores) * 100.0
    }

