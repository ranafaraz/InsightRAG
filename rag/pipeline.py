"""End-to-end RAG orchestration: guardrails -> retrieve -> rerank -> generate."""
from __future__ import annotations

import time

from guardrails import check_injection, redact_pii
from rag.config import Settings, get_settings
from rag.generate import generate_answer
from rag.ingest import ingest_path, ingest_texts
from rag.providers.llm import LLM, get_llm
from rag.providers.rerank import Reranker, get_reranker
from rag.rerank import rerank
from rag.retrieve import HybridRetriever
from rag.store import VectorStore, get_store
from rag.types import Answer

INJECTION_REFUSAL = (
    "Your request looks like a prompt-injection attempt and was blocked by the input "
    "guardrail. Please rephrase as a normal question about the documents."
)


class RAGPipeline:
    """Holds the retriever/reranker/LLM so a server can build it once and reuse it."""

    def __init__(
        self,
        store: VectorStore | None = None,
        settings: Settings | None = None,
        llm: LLM | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.store = store or get_store(self.settings)
        self.retriever = HybridRetriever(self.store, self.settings)
        self.llm = llm or get_llm(self.settings)
        self.reranker = reranker or get_reranker(self.settings)

    # ---- ingestion ---------------------------------------------------------
    def ingest_path(self, path: str) -> int:
        _, n = ingest_path(path, store=self.store, settings=self.settings)
        self.retriever.refresh()
        return n

    def ingest_texts(self, texts: list[str], sources: list[str] | None = None) -> int:
        before = self.store.count()
        ingest_texts(texts, sources, store=self.store, settings=self.settings)
        self.retriever.refresh()
        return self.store.count() - before

    # ---- query -------------------------------------------------------------
    def answer(self, query: str) -> Answer:
        start = time.perf_counter()

        verdict = check_injection(query)
        if verdict.is_injection:
            return Answer(
                query=query,
                text=INJECTION_REFUSAL,
                blocked=True,
                guardrail_flags=[f"prompt_injection:{m}" for m in verdict.matched],
                latency_ms=(time.perf_counter() - start) * 1000,
            )

        candidates = self.retriever.retrieve(query)
        reranked = rerank(query, candidates, reranker=self.reranker, settings=self.settings)
        ans = generate_answer(query, reranked, llm=self.llm, settings=self.settings)

        # Output guardrail: redact any PII that leaked into the answer text.
        redacted, found = redact_pii(ans.text)
        if found:
            ans.text = redacted
            ans.guardrail_flags.extend(f"pii_redacted:{f}" for f in found)

        ans.latency_ms = (time.perf_counter() - start) * 1000
        return ans
