"""Metric implementations.

Retrieval metrics need only gold relevance labels. The RAGAS-style answer-quality
metrics are implemented locally with lexical/embedding proxies so they run offline
in CI; the same harness can be pointed at the real RAGAS package for LLM-judged
scores (see `harness.py --ragas`).
"""
from __future__ import annotations

from rag.text_utils import content_token_set


# ---------------- retrieval ----------------
def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    top = set(retrieved_ids[:k])
    return len(top & relevant_ids) / len(relevant_ids)


def hit_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    return 1.0 if set(retrieved_ids[:k]) & relevant_ids else 0.0


def mrr(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in relevant_ids:
            return 1.0 / rank
    return 0.0


def context_precision(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Fraction of the top-k retrieved contexts that are actually relevant."""
    top = retrieved_ids[:k]
    if not top:
        return 0.0
    return sum(1 for rid in top if rid in relevant_ids) / len(top)


# ---------------- answer quality (RAGAS-style, offline proxies) ----------------
def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def faithfulness(answer: str, contexts: list[str]) -> float:
    """Share of answer content tokens that are supported by the retrieved context.

    A lexical proxy for 'is every claim grounded?'. 1.0 means the answer introduces
    no tokens absent from its context (no hallucinated specifics)."""
    a = content_token_set(answer)
    if not a:
        return 1.0
    ctx = set()
    for c in contexts:
        ctx |= content_token_set(c)
    return len(a & ctx) / len(a)


def answer_relevancy(answer: str, question: str) -> float:
    """How well the answer's content overlaps the question's information need."""
    return _jaccard(content_token_set(answer), content_token_set(question))


def answer_correctness(answer: str, reference: str) -> float:
    """Token-level F1 between the answer and the reference answer."""
    a, r = content_token_set(answer), content_token_set(reference)
    if not a or not r:
        return 0.0
    inter = len(a & r)
    if inter == 0:
        return 0.0
    precision = inter / len(a)
    recall = inter / len(r)
    return 2 * precision * recall / (precision + recall)


# ---------------- classification (guardrails) ----------------
def precision_recall_f1(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true) if y_true else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy}
