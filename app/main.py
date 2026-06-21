"""FastAPI service exposing ingestion and grounded chat over the RAG pipeline."""
from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel

from rag import __version__
from rag.config import get_settings
from rag.pipeline import RAGPipeline
from rag.types import Answer

app = FastAPI(
    title="InsightRAG",
    version=__version__,
    description="Hybrid-retrieval RAG with reranking, citations and guardrails.",
)

_pipeline: RAGPipeline | None = None


def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline


class ChatRequest(BaseModel):
    query: str


class IngestTextRequest(BaseModel):
    texts: list[str]
    sources: list[str] | None = None


class IngestResponse(BaseModel):
    ingested_chunks: int
    total_chunks: int


@app.get("/health")
def health() -> dict:
    s = get_settings()
    return {
        "status": "ok",
        "version": __version__,
        "backends": {
            "embedding": s.embedding_backend,
            "rerank": s.rerank_backend,
            "llm": s.llm_backend,
            "vector_store": s.vector_store,
        },
        "chunks": get_pipeline().store.count(),
    }


@app.post("/ingest/text", response_model=IngestResponse)
def ingest_text(req: IngestTextRequest) -> IngestResponse:
    pipe = get_pipeline()
    n = pipe.ingest_texts(req.texts, req.sources)
    return IngestResponse(ingested_chunks=n, total_chunks=pipe.store.count())


@app.post("/ingest/file", response_model=IngestResponse)
async def ingest_file(file: UploadFile = File(...)) -> IngestResponse:
    pipe = get_pipeline()
    suffix = Path(file.filename or "doc.txt").suffix or ".txt"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        n = pipe.ingest_path(tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)
    return IngestResponse(ingested_chunks=n, total_chunks=pipe.store.count())


@app.post("/chat", response_model=Answer)
def chat(req: ChatRequest) -> Answer:
    return get_pipeline().answer(req.query)
