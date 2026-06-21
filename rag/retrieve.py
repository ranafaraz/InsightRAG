"""Hybrid retrieval: dense (vector) + sparse (BM25), fused by normalised score.

BM25 catches exact-term / rare-keyword matches that dense embeddings miss; dense
catches paraphrases and synonyms that BM25 misses. Fusing both with a tunable
``alpha`` consistently beats either signal alone (see the eval ablation).
"""
from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

from rag.config import Settings, get_settings
from rag.store import VectorStore
from rag.types import Chunk, Scored

_TOKEN = re.compile(r"[a-z0-9]+")


def _tok(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def _minmax(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    vals = list(scores.values())
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-12:
        return {k: 0.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


class HybridRetriever:
    def __init__(self, store: VectorStore, settings: Settings | None = None) -> None:
        self.store = store
        self.settings = settings or get_settings()
        self._build_bm25()

    def _build_bm25(self) -> None:
        self._chunks: list[Chunk] = self.store.all_chunks()
        self._by_id = {c.id: c for c in self._chunks}
        corpus = [_tok(c.text) for c in self._chunks]
        self._bm25 = BM25Okapi(corpus) if corpus else None

    def refresh(self) -> None:
        """Rebuild the BM25 index after new documents are ingested."""
        self._build_bm25()

    def retrieve(self, query: str, k: int | None = None) -> list[Scored]:
        k = k or self.settings.retrieve_k
        alpha = self.settings.hybrid_alpha
        if not self._chunks:
            return []

        # Dense leg
        dense_hits = self.store.query(query, k=max(k, self.settings.retrieve_k))
        dense_scores = {h.chunk.id: h.score for h in dense_hits}

        # Sparse leg (BM25 over the whole corpus)
        bm25_scores: dict[str, float] = {}
        if self._bm25 is not None:
            raw = self._bm25.get_scores(_tok(query))
            bm25_scores = {self._chunks[i].id: float(raw[i]) for i in range(len(self._chunks))}

        dn = _minmax(dense_scores)
        bn = _minmax(bm25_scores)

        fused: dict[str, float] = {}
        for cid in set(dn) | set(bn):
            fused[cid] = alpha * dn.get(cid, 0.0) + (1 - alpha) * bn.get(cid, 0.0)

        ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:k]
        out: list[Scored] = []
        for cid, score in ranked:
            out.append(
                Scored(
                    chunk=self._by_id[cid],
                    score=score,
                    components={
                        "dense": dn.get(cid, 0.0),
                        "bm25": bn.get(cid, 0.0),
                        "hybrid": score,
                    },
                )
            )
        return out
