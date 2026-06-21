# InsightRAG API image. Defaults to the offline/free stack so it runs anywhere;
# set EMBEDDING_BACKEND / RERANK_BACKEND / LLM_BACKEND at runtime to use real models.
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    EMBEDDING_BACKEND=hash \
    RERANK_BACKEND=lexical \
    LLM_BACKEND=stub \
    VECTOR_STORE=memory

WORKDIR /app

# Install dependencies first for better layer caching.
COPY pyproject.toml README.md ./
COPY rag ./rag
COPY app ./app
COPY guardrails ./guardrails
COPY eval_harness ./eval_harness

RUN pip install --upgrade pip && pip install ".[ui]"

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
