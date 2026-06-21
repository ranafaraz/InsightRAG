"""Streamlit demo UI for InsightRAG.

Run: `streamlit run ui/streamlit_app.py`
Works fully offline with the default stub/hash/lexical backends — drop in your own
docs and ask away, no API key required.
"""
from __future__ import annotations

import streamlit as st

from rag.config import get_settings
from rag.pipeline import RAGPipeline

st.set_page_config(page_title="InsightRAG", page_icon="🔎", layout="wide")


@st.cache_resource
def load_pipeline() -> RAGPipeline:
    return RAGPipeline()


SAMPLE_DOCS = [
    "InsightRAG combines BM25 sparse retrieval with dense vector search, then reranks "
    "the candidates with a cross-encoder before generating a grounded answer.",
    "Citations in InsightRAG point back to the exact source passage, and the assistant "
    "refuses to answer when the retrieved context is insufficient.",
    "Guardrails screen incoming queries for prompt-injection attempts and redact PII "
    "such as emails and phone numbers from the model output.",
]

pipe = load_pipeline()
settings = get_settings()

st.title("🔎 InsightRAG")
st.caption(
    "Hybrid retrieval · cross-encoder reranking · grounded citations · guardrails — "
    f"running on `{settings.llm_backend}` LLM / `{settings.embedding_backend}` embeddings."
)

with st.sidebar:
    st.header("Documents")
    st.write(f"Chunks indexed: **{pipe.store.count()}**")
    if st.button("Load sample documents"):
        n = pipe.ingest_texts(SAMPLE_DOCS, [f"sample_{i}.md" for i in range(len(SAMPLE_DOCS))])
        st.success(f"Ingested {n} chunks.")
        st.rerun()

    uploaded = st.file_uploader(
        "Upload PDF / Markdown / text", type=["pdf", "md", "markdown", "txt"], accept_multiple_files=True
    )
    if uploaded and st.button("Ingest uploaded"):
        import tempfile
        from pathlib import Path

        total = 0
        for uf in uploaded:
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uf.name).suffix) as tmp:
                tmp.write(uf.read())
                path = tmp.name
            total += pipe.ingest_path(path)
            Path(path).unlink(missing_ok=True)
        st.success(f"Ingested {total} chunks.")
        st.rerun()

query = st.text_input("Ask a question about your documents:")
if query:
    if pipe.store.count() == 0:
        st.warning("No documents indexed yet — load the samples or upload a file first.")
    else:
        ans = pipe.answer(query)
        if ans.blocked:
            st.error(ans.text)
        else:
            st.markdown("### Answer")
            st.write(ans.text)
        if ans.citations:
            st.markdown("### Sources")
            for c in ans.citations:
                with st.expander(f"{c.marker} — {c.source}"):
                    st.write(c.snippet)
        cols = st.columns(3)
        cols[0].metric("Latency", f"{ans.latency_ms:.0f} ms")
        cols[1].metric("Refused", "yes" if ans.refused else "no")
        cols[2].metric("Guardrail flags", len(ans.guardrail_flags))
