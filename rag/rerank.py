"""Rerank a candidate set with a cross-encoder (or lexical fallback)."""
from __future__ import annotations

from rag.config import Settings, get_settings
from rag.providers.rerank import Reranker, get_reranker
from rag.types import Scored


def rerank(
    query: str,
    candidates: list[Scored],
    reranker: Reranker | None = None,
    settings: Settings | None = None,
) -> list[Scored]:
    settings = settings or get_settings()
    if not candidates:
        return []
    reranker = reranker or get_reranker(settings)
    scores = reranker.score(query, [c.chunk.text for c in candidates])

    reranked: list[Scored] = []
    for cand, s in zip(candidates, scores):
        comp = dict(cand.components)
        comp["rerank"] = float(s)
        reranked.append(Scored(chunk=cand.chunk, score=float(s), components=comp))

    reranked.sort(key=lambda c: c.score, reverse=True)
    return reranked[: settings.rerank_top_n]
