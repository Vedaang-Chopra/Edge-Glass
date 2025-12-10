
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
