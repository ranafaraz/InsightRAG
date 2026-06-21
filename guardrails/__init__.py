"""Lightweight, dependency-free safety guardrails for the RAG pipeline.

These are intentionally small and rule-based so they run in CI with no models.
The standalone `GuardrAIl` repo (next in the portfolio) is the heavyweight version;
InsightRAG ships a focused subset: prompt-injection screening, PII redaction, and a
context-size cap.
"""
from guardrails.injection import InjectionVerdict, check_injection
from guardrails.pii import redact_pii

__all__ = ["check_injection", "InjectionVerdict", "redact_pii"]
