"""Reranker backends.

`lexical`       -- token-overlap (Jaccard-ish) reranker, no download (CI/offline).
`cross-encoder` -- a real cross-encoder relevance model (default for local use).

Both return scores normalised to [0, 1] so the `MIN_RERANK_SCORE` refusal threshold
behaves consistently across backends.
"""
from __future__ import annotations

from typing import Protocol

from rag.config import Settings, get_settings
from rag.text_utils import content_token_set as _tokens


class Reranker(Protocol):
    def score(self, query: str, passages: list[str]) -> list[float]: ...


class LexicalReranker:
    """Overlap coefficient between query and passage tokens, in [0, 1]."""

    def score(self, query: str, passages: list[str]) -> list[float]:
        q = _tokens(query)
        if not q:
            return [0.0] * len(passages)
        out = []
        for p in passages:
            pt = _tokens(p)
            overlap = len(q & pt)
            out.append(overlap / len(q))
        return out


class CrossEncoderReranker:
    def __init__(self, model_name: str) -> None:
        from sentence_transformers import CrossEncoder  # lazy import

        self._model = CrossEncoder(model_name)

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        pairs = [(query, p) for p in passages]
        raw = self._model.predict(pairs)
        # Squash logits to [0, 1] for a stable, backend-agnostic threshold.
        import math

        return [1.0 / (1.0 + math.exp(-float(s))) for s in raw]


def get_reranker(settings: Settings | None = None) -> Reranker:
    settings = settings or get_settings()
    if settings.rerank_backend == "cross-encoder":
        return CrossEncoderReranker(settings.rerank_model)
    return LexicalReranker()
