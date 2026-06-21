"""Command-line entry point: ingest documents and ask questions."""
from __future__ import annotations

import argparse
import sys

from rag.config import get_settings
from rag.pipeline import RAGPipeline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="insightrag", description="InsightRAG CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ing = sub.add_parser("ingest", help="ingest a file or directory")
    p_ing.add_argument("path")

    p_ask = sub.add_parser("ask", help="ask a question (ingest first)")
    p_ask.add_argument("question")
    p_ask.add_argument("--path", help="ingest this path before asking", default=None)

    args = parser.parse_args(argv)
    settings = get_settings()
    pipe = RAGPipeline(settings=settings)

    if args.cmd == "ingest":
        n = pipe.ingest_path(args.path)
        print(f"Ingested {n} chunks from {args.path} into '{settings.vector_store}' store.")
        return 0

    if args.cmd == "ask":
        if args.path:
            pipe.ingest_path(args.path)
        ans = pipe.answer(args.question)
        print(f"\nAnswer:\n{ans.text}\n")
        if ans.citations:
            print("Sources:")
            for c in ans.citations:
                print(f"  {c.marker} {c.source}")
        if ans.guardrail_flags:
            print(f"\n[guardrails] {', '.join(ans.guardrail_flags)}")
        print(f"\nlatency: {ans.latency_ms:.0f} ms | backend: {settings.llm_backend}")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
