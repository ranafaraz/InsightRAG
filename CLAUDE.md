# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

InsightRAG is a production-grade RAG service: hybrid retrieval (BM25 + dense) → cross-encoder rerank → grounded answer with citations, wrapped in a FastAPI service, a Streamlit UI, guardrails, and an evaluation harness. It is the flagship repo of a larger public AI/ML portfolio (`ranafaraz`).

## Commands

Use the venv interpreter at `.venv/Scripts/python.exe` (Windows). Force the offline backends via env vars when running anything locally that should not download models.

```bash
pip install -e ".[dev]"          # minimal/offline stack (what CI uses)
pip install -e ".[local]"        # + sentence-transformers + chromadb (real models)
pip install -e ".[ui]"           # + streamlit

pytest -q                        # full suite (offline, no downloads)
pytest tests/test_pipeline.py -q                 # one file
pytest tests/test_pipeline.py::test_pipeline_refuses_unknown   # one test
ruff check .                     # lint (CI fails on lint errors)
ruff check . --fix               # autofix imports etc.

python -m eval_harness.harness   # regenerate eval_harness/RESULTS.md
python -m eval_harness.gate      # CI quality gate (exit 1 if metrics regress)

uvicorn app.main:app --reload    # API at /docs
streamlit run ui/streamlit_app.py
python -m rag.cli ask "..." --path docs/         # CLI
docker compose up --build        # API on :8000
```

CI (`.github/workflows/ci.yml`) runs ruff + pytest + the eval gate on Python 3.10–3.12 with `EMBEDDING_BACKEND=hash RERANK_BACKEND=lexical LLM_BACKEND=stub VECTOR_STORE=memory`.

## Architecture: the pluggable-backend design is the core idea

Every heavy component has **two backends** — a deterministic offline one (no downloads, no API key, no GPU) and a real one — selected at runtime from `rag/config.py` (`Settings`, env-driven, `.env`). This is why the entire pipeline *and its evaluation* run green in CI with nothing installed beyond the core deps. When adding or changing a component, preserve this property: the offline backend must keep tests and the eval gate passing.

| Concern | Factory | Offline default | Real backend |
|---|---|---|---|
| Embeddings | `rag/providers/embeddings.py::get_embedder` | `hash` (hashing BoW) | `sentence-transformers` |
| Rerank | `rag/providers/rerank.py::get_reranker` | `lexical` (token overlap) | `cross-encoder` |
| LLM | `rag/providers/llm.py::get_llm` | `stub` (extractive) | `ollama` / `openai` |
| Vector store | `rag/store.py::get_store` | `memory` (numpy) | `chroma` |

Each backend implements a small `Protocol`; callers depend on the protocol, never the concrete class.

### Request flow
`rag/pipeline.py::RAGPipeline.answer()` is the orchestrator and the place to understand the system end to end:
1. **Input guardrail** — `guardrails/injection.py` screens the query; injection → blocked answer, no retrieval.
2. **Hybrid retrieve** — `rag/retrieve.py::HybridRetriever` fuses dense (from the store) + BM25 (built over `store.all_chunks()`) with min-max normalisation and `hybrid_alpha` (0 = BM25 only, 1 = dense only).
3. **Rerank** — `rag/rerank.py` re-scores candidates and keeps `rerank_top_n`.
4. **Generate** — `rag/generate.py` refuses *before* calling the LLM if top score < `min_rerank_score`; otherwise builds a numbered CONTEXT prompt and extracts citations only from the `[n]` markers the model actually used.
5. **Output guardrail** — `guardrails/pii.py` redacts PII from the answer text.

### Two things that are easy to get wrong
- **The `StubLLM` parses the prompt.** `rag/generate.py::build_prompt` emits a specific `[n] (source: ...) text` CONTEXT format, and `StubLLM.generate` (in `rag/providers/llm.py`) parses that exact format to produce a deterministic extractive answer. Changing the prompt layout will break the offline LLM — keep them in sync.
- **Rebuild BM25 after ingest.** The BM25 index is built from the store's chunks at retriever construction. After ingesting, call `HybridRetriever.refresh()` (the pipeline does this in its `ingest_*` methods). New docs are invisible to sparse retrieval until you do.

### Evaluation is gated, not decorative
`eval_harness/` computes retrieval metrics (recall@k, MRR, context precision), RAGAS-style answer quality (lexical/embedding proxies that run offline), and guardrail precision/recall over a bundled benchmark (`eval_harness/data/`). `eval_harness/gate.py` enforces floors (recall@5, faithfulness, injection-F1, PII accuracy) and is run in CI — a refactor that degrades quality fails the build. Update thresholds there if you intentionally change behaviour.

## Conventions
- Shared lexical scoring (with stopword filtering) lives in `rag/text_utils.py`; reuse it rather than re-tokenising, so the stub LLM, lexical reranker, and eval metrics stay consistent.
- Windows console is cp1252 — do **not** `print()` non-ASCII (≈, em-dash) from scripts; write UTF-8 to files or set `PYTHONIOENCODING=utf-8`.
- The Python eval package is `eval_harness` (not `eval`, which shadows the builtin).
