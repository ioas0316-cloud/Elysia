import json
import math
from collections import Counter
from typing import Dict, List, Tuple

from tools.text_preprocessor import extract_content_tokens


def train_naive_bayes(corpus: Dict[str, List[str]], alpha: float = 1.0) -> Dict:
    """
    Trains a simple Multinomial Naive Bayes classifier.
    corpus: {label: [doc_text, ...]}
    returns a JSON‑serializable model dict.
    """
    label_doc_count = {lbl: len(docs) for lbl, docs in corpus.items()}
    total_docs = sum(label_doc_count.values()) or 1
    priors = {lbl: math.log((label_doc_count[lbl]) / total_docs) for lbl in corpus}

    vocab = set()
    label_word_counts: Dict[str, Counter] = {lbl: Counter() for lbl in corpus}
    label_total_tokens: Dict[str, int] = {lbl: 0 for lbl in corpus}

    for lbl, docs in corpus.items():
        for text in docs:
            toks = extract_content_tokens(text)
            vocab.update(toks)
            label_word_counts[lbl].update(toks)
            label_total_tokens[lbl] += len(toks)

    V = len(vocab) or 1
    # store as plain dicts
    model = {
        "priors": priors,
        "alpha": alpha,
        "vocab_size": V,
        "label_total_tokens": label_total_tokens,
        "label_word_counts": {lbl: dict(cnt) for lbl, cnt in label_word_counts.items()},
    }
    return model


def predict(model: Dict, text: str) -> Tuple[str, List[Tuple[str, float]]]:
    alpha = float(model.get("alpha", 1.0))
    V = int(model.get("vocab_size", 1))
    priors = model.get("priors", {})
    ltot = model.get("label_total_tokens", {})
    lwc = model.get("label_word_counts", {})
    toks = extract_content_tokens(text)
    scores: Dict[str, float] = {}
    for lbl in priors:
        logp = float(priors[lbl])
        denom = (ltot.get(lbl, 0) + alpha * V)
        for t in toks:
            count = lwc.get(lbl, {}).get(t, 0)
            logp += math.log((count + alpha) / denom)
        scores[lbl] = logp
    # normalize to probabilities (softmax)
    if not scores:
        return "", []
    maxlog = max(scores.values())
    exp = {k: math.exp(v - maxlog) for k, v in scores.items()}
    Z = sum(exp.values()) or 1.0
    probs = [(k, exp[k] / Z) for k in exp]
    probs.sort(key=lambda x: x[1], reverse=True)
    return probs[0][0], probs


def save_model(model: Dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(model, f, ensure_ascii=False, indent=2)


def load_model(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
