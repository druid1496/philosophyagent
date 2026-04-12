#!/usr/bin/env python3
"""
Dev-only: inspect philosopher RAG retrieval quality (scores + chunk text).

Not used by the debate pipeline. Run from the project root:

  python scripts/test_rag.py --philosopher Kant --query "What is the categorical imperative?"
  python scripts/test_rag.py --philosopher Plato --interactive

Requires OPENAI_API_KEY (same .env as main.py). Uses the same persisted indexes as main.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

from config import EMBEDDING_MODEL, PHILOSOPHERS
from rag.retriever import load_philosopher_vectorstore, setup_philosopher_rag


def _print_hits(query: str, philosopher: str, k: int, use_retriever: bool, force_rebuild: bool) -> None:
    cfg = PHILOSOPHERS[philosopher]
    text_file = cfg["text_file"]

    if use_retriever:
        retriever = setup_philosopher_rag(
            philosopher,
            text_file,
            EMBEDDING_MODEL,
            k=k,
            force_rebuild=force_rebuild,
        )
        docs = retriever.invoke(query)
        print(f"\nQuery: {query!r}\nPhilosopher: {philosopher}  (production-style retriever.invoke, k={k})\n")
        for i, doc in enumerate(docs, 1):
            prev = doc.page_content.replace("\n", " ")[:500]
            more = "…" if len(doc.page_content) > 500 else ""
            print(f"--- Hit {i} (len={len(doc.page_content)} chars) ---\n{prev}{more}\n")
        return

    vs = load_philosopher_vectorstore(
        philosopher,
        text_file,
        EMBEDDING_MODEL,
        force_rebuild=force_rebuild,
    )
    pairs = vs.similarity_search_with_score(query, k=k)
    print(
        f"\nQuery: {query!r}\nPhilosopher: {philosopher}  "
        f"(FAISS L2 distance; lower = closer, k={k})\n",
    )
    for i, (doc, score) in enumerate(pairs, 1):
        prev = doc.page_content.replace("\n", " ")[:500]
        more = "…" if len(doc.page_content) > 500 else ""
        print(f"--- Hit {i}  score={score:.4f}  (len={len(doc.page_content)} chars) ---\n{prev}{more}\n")


def main() -> None:
    load_dotenv(ROOT / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set. Copy .env.example to .env in the project root.", file=sys.stderr)
        sys.exit(1)

    names = list(PHILOSOPHERS.keys())
    p = argparse.ArgumentParser(description="Test philosopher RAG retrieval (dev tool).")
    p.add_argument("--philosopher", choices=names, default=names[0], help="Corpus to search.")
    p.add_argument("--query", type=str, default="", help="Search query.")
    p.add_argument("-k", type=int, default=5, help="Number of chunks to retrieve.")
    p.add_argument(
        "--retriever",
        action="store_true",
        help="Use the same retriever path as main.py (no similarity scores).",
    )
    p.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Force rebuild of the FAISS index for this philosopher.",
    )
    p.add_argument(
        "--interactive",
        action="store_true",
        help="After running --query (if any), prompt for more queries until EOF or 'quit'.",
    )
    args = p.parse_args()

    if not args.query and not args.interactive:
        p.print_help()
        print("\nProvide --query and/or --interactive.", file=sys.stderr)
        sys.exit(1)

    queries: list[str] = []
    if args.query:
        queries.append(args.query)

    if args.interactive:
        print("Interactive mode. Enter queries (blank line to skip if first). Type quit or exit to stop.\n")

    for q in queries:
        _print_hits(q, args.philosopher, args.k, args.retriever, args.rebuild_index)

    if not args.interactive:
        return

    while True:
        try:
            line = input("query> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line or line.lower() in ("quit", "exit", "q"):
            break
        _print_hits(line, args.philosopher, args.k, args.retriever, False)


if __name__ == "__main__":
    main()
