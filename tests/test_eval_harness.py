from eval_harness import metrics as M
from eval_harness.harness import run_guardrails, run_retrieval_ablation
from rag.config import Settings


def test_retrieval_metrics_math():
    assert M.recall_at_k(["a", "b", "c"], {"a"}, 3) == 1.0
    assert M.recall_at_k(["x", "y"], {"a"}, 3) == 0.0
    assert M.mrr(["x", "a"], {"a"}) == 0.5
    assert M.context_precision(["a", "x"], {"a"}, 2) == 0.5


def test_prf1():
    out = M.precision_recall_f1([1, 1, 0, 0], [1, 0, 0, 0])
    assert out["precision"] == 1.0
    assert out["recall"] == 0.5
    assert 0 < out["f1"] < 1


def test_faithfulness_grounded_vs_hallucinated():
    ctx = ["Paris is the capital of France."]
    assert M.faithfulness("The capital of France is Paris.", ctx) == 1.0
    assert M.faithfulness("Berlin quantum spaceship velocity", ctx) < 0.5


def test_retrieval_ablation_runs_offline():
    s = Settings(embedding_backend="hash", rerank_backend="lexical", vector_store="memory")
    rows = run_retrieval_ablation(s)
    names = {r.name for r in rows}
    assert "Hybrid (BM25+dense)" in names
    # retrieval should clearly work on the bundled corpus
    assert all(r.recall_at_5 > 0.5 for r in rows)


def test_guardrail_eval_high_quality():
    g = run_guardrails()
    assert g["injection"]["f1"] >= 0.8
    assert g["pii_redaction"]["accuracy"] >= 0.8
