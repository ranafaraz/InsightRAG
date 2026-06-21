"""CI eval gate: fail the build if core quality metrics regress.

This turns evaluation into a guardrail on the *codebase*, not just a report — the
kind of check that stops a refactor from silently degrading faithfulness.
"""
from __future__ import annotations

import sys

from eval_harness.harness import (
    run_answer_quality,
    run_guardrails,
    run_retrieval_ablation,
)
from rag.config import get_settings

# Thresholds calibrated against the deterministic offline backends with margin.
THRESHOLDS = {
    "recall_at_5": 0.80,
    "faithfulness": 0.80,
    "injection_f1": 0.90,
    "pii_accuracy": 0.80,
}


def main() -> int:
    settings = get_settings()
    rows = run_retrieval_ablation(settings)
    hybrid = next(r for r in rows if r.name.startswith("Hybrid (BM25"))
    aq = run_answer_quality(settings)
    g = run_guardrails()

    checks = {
        "recall_at_5": hybrid.recall_at_5,
        "faithfulness": aq["faithfulness"],
        "injection_f1": g["injection"]["f1"],
        "pii_accuracy": g["pii_redaction"]["accuracy"],
    }

    failed = []
    for name, value in checks.items():
        floor = THRESHOLDS[name]
        ok = value >= floor
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {value:.3f} (>= {floor})")
        if not ok:
            failed.append(name)

    if failed:
        print(f"\nEval gate FAILED on: {', '.join(failed)}")
        return 1
    print("\nEval gate PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
