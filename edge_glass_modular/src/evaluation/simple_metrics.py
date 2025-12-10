
import math
from collections import Counter

def compute_bleu(reference, candidate, n=4):
    """
    Simple implementation of BLEU score.
    """
    ref_tokens = reference.lower().split()
    cand_tokens = candidate.lower().split()
    
    if len(cand_tokens) == 0:
        return 0.0
    
    # Precision for n-grams
    precisions = []
    for i in range(1, n + 1):
        ref_ngrams = Counter(zip(*[ref_tokens[j:] for j in range(i)]))
        cand_ngrams = Counter(zip(*[cand_tokens[j:] for j in range(i)]))
        
        num = sum((cand_ngrams & ref_ngrams).values())
        den = max(1, sum(cand_ngrams.values()))
        precisions.append(num / den)
        
    # Geometric mean
    if min(precisions) > 0:
        p_log_sum = sum((1. / n) * math.log(p) for p in precisions)
        geo_mean = math.exp(p_log_sum)
    else:
        geo_mean = 0.0
        
    # Brevity penalty
    bp = 1.0
    if len(cand_tokens) < len(ref_tokens):
        bp = math.exp(1 - len(ref_tokens) / len(cand_tokens))
        
    return bp * geo_mean

def compute_rouge_l(reference, candidate):
    """
    Simple implementation of ROUGE-L (LCS-based).
    """
    ref_tokens = reference.lower().split()
    cand_tokens = candidate.lower().split()
    
    if not ref_tokens or not cand_tokens:
        return 0.0
        
    m = len(ref_tokens)
    n = len(cand_tokens)
    
    # LCS DP table
    lcs = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_tokens[i-1] == cand_tokens[j-1]:
                lcs[i][j] = lcs[i-1][j-1] + 1
            else:
                lcs[i][j] = max(lcs[i-1][j], lcs[i][j-1])
                
    lcs_len = lcs[m][n]
    
    if lcs_len == 0:
        return 0.0
        
    precision = lcs_len / n
    recall = lcs_len / m
    
    if precision + recall == 0:
        return 0.0
        
    f1 = 2 * precision * recall / (precision + recall)
    return f1

def compute_perplexity(loss):
    """
    Compute perplexity from cross-entropy loss.
    """
    try:
        return math.exp(loss)
    except OverflowError:
        return float('inf')
