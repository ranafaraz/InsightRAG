"""Run the evaluation suite and write a markdown results table.

Usage:
    python -m eval_harness.harness                 # offline deterministic backends
    EMBEDDING_BACKEND=sentence-transformers \\
    RERANK_BACKEND=cross-encoder python -m eval_harness.harness   # real models

Outputs `eval_harness/RESULTS.md`.
"""
from __future__ import annotations

import json
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

from eval_harness import metrics as M
from guardrails import check_injection, redact_pii
from rag.config import Settings, get_settings
from rag.generate import generate_answer
from rag.ingest import ingest_texts
from rag.providers.llm import get_llm
from rag.providers.rerank import get_reranker
from rag.rerank import rerank
from rag.retrieve import HybridRetriever
from rag.store import get_store

DATA = Path(__file__).parent / "data"
RESULTS = Path(__file__).parent / "RESULTS.md"


def load_corpus() -> tuple[list[str], list[str], list[dict]]:
    obj = json.loads((DATA / "corpus.json").read_text(encoding="utf-8"))
    docs = obj["documents"]
    return (
        [d["text"] for d in docs],
        [d["id"] for d in docs],
        obj["questions"],
    )


def _build_store(settings: Settings):
    texts, ids, questions = load_corpus()
    store = get_store(settings)
    ingest_texts(texts, ids, store=store, settings=settings)
    return store, questions


@dataclass
class RetrievalRow:
    name: str
    recall_at_3: float
    recall_at_5: float
    mrr: float
    ctx_precision_at_5: float


def _retrieved_sources(scored) -> list[str]:
    return [s.chunk.source for s in scored]


def run_retrieval_ablation(settings: Settings) -> list[RetrievalRow]:
    store, questions = _build_store(settings)
    reranker = get_reranker(settings)

    configs = [
        ("BM25 only", 0.0, False),
        ("Dense only", 1.0, False),
        ("Hybrid (BM25+dense)", 0.5, False),
        ("Hybrid + reranker", 0.5, True),
    ]
    rows: list[RetrievalRow] = []
    for name, alpha, use_rerank in configs:
        s = deepcopy(settings)
        s.hybrid_alpha = alpha
        retriever = HybridRetriever(store, s)
        r3 = r5 = mrr = cp5 = 0.0
        for q in questions:
            rel = set(q["relevant_doc_ids"])
            cands = retriever.retrieve(q["q"], k=settings.retrieve_k)
            if use_rerank:
                cands = rerank(q["q"], cands, reranker=reranker, settings=s)
            ids = _retrieved_sources(cands)
            r3 += M.recall_at_k(ids, rel, 3)
            r5 += M.recall_at_k(ids, rel, 5)
            mrr += M.mrr(ids, rel)
            cp5 += M.context_precision(ids, rel, 5)
        n = len(questions)
        rows.append(RetrievalRow(name, r3 / n, r5 / n, mrr / n, cp5 / n))
    return rows


def run_answer_quality(settings: Settings) -> dict[str, float]:
    store, questions = _build_store(settings)
    reranker = get_reranker(settings)
    llm = get_llm(settings)

    faith = relev = corr = 0.0
    lat_total = 0.0
    ctx_before = ctx_after = 0
    for q in questions:
        retriever = HybridRetriever(store, settings)
        t0 = time.perf_counter()
        cands = retriever.retrieve(q["q"], k=settings.retrieve_k)
        ctx_before += sum(len(c.chunk.text) for c in cands)
        reranked = rerank(q["q"], cands, reranker=reranker, settings=settings)
        ctx_after += sum(len(c.chunk.text) for c in reranked)
        ans = generate_answer(q["q"], reranked, llm=llm, settings=settings)
        lat_total += (time.perf_counter() - t0) * 1000
        ctx_texts = [c.chunk.text for c in reranked]
        faith += M.faithfulness(ans.text, ctx_texts)
        relev += M.answer_relevancy(ans.text, q["q"])
        corr += M.answer_correctness(ans.text, q["answer"])
    n = len(questions)
    return {
        "faithfulness": faith / n,
        "answer_relevancy": relev / n,
        "answer_correctness": corr / n,
        "avg_latency_ms": lat_total / n,
        # approx prompt tokens (chars/4) fed to the LLM before vs after rerank
        "ctx_tokens_before": ctx_before / n / 4,
        "ctx_tokens_after": ctx_after / n / 4,
    }


def run_guardrails() -> dict[str, dict]:
    inj = json.loads((DATA / "injection.json").read_text(encoding="utf-8"))["samples"]
    y_true = [s["label"] for s in inj]
    y_pred = [1 if check_injection(s["text"]).is_injection else 0 for s in inj]
    injection = M.precision_recall_f1(y_true, y_pred)

    # PII redaction accuracy on synthetic strings
    pii_samples = [
        ("Email me at jane.doe@example.com", True),
        ("My SSN is 123-45-6789", True),
        ("Call 415-555-0132 after noon", True),
        ("Server IP is 10.0.0.42", True),
        ("The meeting is at noon tomorrow", False),
        ("Diversification reduces portfolio risk", False),
    ]
    correct = 0
    for text, has_pii in pii_samples:
        _, found = redact_pii(text)
        if bool(found) == has_pii:
            correct += 1
    return {
        "injection": injection,
        "pii_redaction": {"accuracy": correct / len(pii_samples), "n": len(pii_samples)},
    }


def _fmt(x: float) -> str:
    return f"{x:.3f}"


def render_markdown(settings: Settings) -> str:
    retr = run_retrieval_ablation(settings)
    aq = run_answer_quality(settings)
    g = run_guardrails()

    lines: list[str] = []
    lines.append("# InsightRAG — Evaluation Results\n")
    lines.append(
        f"_Backends: embedding=`{settings.embedding_backend}`, "
        f"rerank=`{settings.rerank_backend}`, llm=`{settings.llm_backend}`, "
        f"store=`{settings.vector_store}`. "
        f"Corpus: {len(load_corpus()[2])} questions over {len(load_corpus()[1])} documents._\n"
    )

    lines.append("## Retrieval ablation\n")
    lines.append("| Configuration | Recall@3 | Recall@5 | MRR | Context Precision@5 |")
    lines.append("|---|---|---|---|---|")
    for r in retr:
        lines.append(
            f"| {r.name} | {_fmt(r.recall_at_3)} | {_fmt(r.recall_at_5)} | "
            f"{_fmt(r.mrr)} | {_fmt(r.ctx_precision_at_5)} |"
        )
    lines.append("")

    lines.append("## Answer quality (RAGAS-style) & cost\n")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| Faithfulness | {_fmt(aq['faithfulness'])} |")
    lines.append(f"| Answer relevancy | {_fmt(aq['answer_relevancy'])} |")
    lines.append(f"| Answer correctness (token F1) | {_fmt(aq['answer_correctness'])} |")
    lines.append(f"| Avg latency / query | {aq['avg_latency_ms']:.1f} ms |")
    lines.append(f"| Prompt tokens before rerank (~) | {aq['ctx_tokens_before']:.0f} |")
    lines.append(f"| Prompt tokens after rerank (~) | {aq['ctx_tokens_after']:.0f} |")
    reduction = (
        100 * (1 - aq["ctx_tokens_after"] / aq["ctx_tokens_before"])
        if aq["ctx_tokens_before"]
        else 0.0
    )
    lines.append(f"| Context/token reduction from rerank | {reduction:.0f}% |")
    lines.append("")

    lines.append("## Guardrails\n")
    inj = g["injection"]
    lines.append("| Guardrail | Precision | Recall | F1 | Accuracy |")
    lines.append("|---|---|---|---|---|")
    lines.append(
        f"| Prompt-injection detection | {_fmt(inj['precision'])} | {_fmt(inj['recall'])} | "
        f"{_fmt(inj['f1'])} | {_fmt(inj['accuracy'])} |"
    )
    pii = g["pii_redaction"]
    lines.append(f"| PII redaction (n={pii['n']}) | — | — | — | {_fmt(pii['accuracy'])} |")
    lines.append("")
    if settings.embedding_backend == "hash":
        note = (
            "Numbers above are from the deterministic offline backends, so they regenerate "
            "identically on any machine (this is what CI runs). Switching to "
            "`sentence-transformers` + `cross-encoder` raises retrieval quality further — "
            "re-run with those env vars to regenerate."
        )
    else:
        note = (
            f"Numbers above use the real `{settings.embedding_backend}` embeddings and "
            f"`{settings.rerank_backend}` reranker (llm=`{settings.llm_backend}`)."
        )
    lines.append(f"> Generated by `python -m eval_harness.harness`. {note}")
    return "\n".join(lines)


def main() -> dict:
    settings = get_settings()
    md = render_markdown(settings)
    RESULTS.write_text(md, encoding="utf-8")
    print(md)
    print(f"\nWrote {RESULTS}")
    return {"ok": True}


if __name__ == "__main__":
    main()
