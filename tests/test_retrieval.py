from rag.config import Settings
from rag.ingest import ingest_texts
from rag.providers.embeddings import HashingEmbedder
from rag.retrieve import HybridRetriever
from rag.store import InMemoryStore


def _store(settings):
    store = InMemoryStore(HashingEmbedder(settings.hash_embedding_dim))
    ingest_texts(
        [
            "Compound interest grows on principal and accumulated interest.",
            "Diversification spreads investments to reduce portfolio risk.",
            "Spaced repetition improves long-term memory recall.",
        ],
        ["a", "b", "c"],
        store=store,
        settings=settings,
    )
    return store


def test_hybrid_retrieves_relevant_doc_first():
    s = Settings(embedding_backend="hash", vector_store="memory")
    retriever = HybridRetriever(_store(s), s)
    hits = retriever.retrieve("How does compound interest work?", k=3)
    assert hits
    assert hits[0].chunk.source == "a"
    # component scores are exposed for transparency
    assert {"dense", "bm25", "hybrid"} <= set(hits[0].components)


def test_alpha_extremes_select_single_signal():
    s = Settings(embedding_backend="hash", vector_store="memory")
    store = _store(s)
    s_bm25 = Settings(embedding_backend="hash", vector_store="memory", hybrid_alpha=0.0)
    s_dense = Settings(embedding_backend="hash", vector_store="memory", hybrid_alpha=1.0)
    bm25 = HybridRetriever(store, s_bm25).retrieve("portfolio risk diversification", k=3)
    dense = HybridRetriever(store, s_dense).retrieve("portfolio risk diversification", k=3)
    assert bm25[0].chunk.source == "b"
    assert dense[0].chunk.source == "b"


def test_empty_store_returns_no_hits():
    s = Settings(embedding_backend="hash", vector_store="memory")
    retriever = HybridRetriever(InMemoryStore(HashingEmbedder(s.hash_embedding_dim)), s)
    assert retriever.retrieve("anything", k=5) == []
