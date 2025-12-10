
try:
    import evaluate
    print("evaluate: available")
except ImportError:
    print("evaluate: missing")

try:
    import nltk
    print("nltk: available")
except ImportError:
    print("nltk: missing")

try:
    import rouge_score
    print("rouge_score: available")
except ImportError:
    print("rouge_score: missing")

try:
    import sacrebleu
    print("sacrebleu: available")
except ImportError:
    print("sacrebleu: missing")
