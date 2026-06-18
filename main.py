"""Section B entry point.

The autograder imports this module and calls `run(queries)` once with the full
batch of evaluation queries. It never rebuilds the index -- `run` loads the
prebuilt artifacts from disk (see index.py / README).
"""
from __future__ import annotations

from typing import List

from index import build_index
from retrieve import search_batch


def run(queries: List[str]) -> List[List[int]]:
    """Return one ranked list of page_id per query (most relevant first)."""
    return search_batch(queries)


def build_offline_index() -> None:
    """Build artifacts/ from the corpus. Run once locally; not timed at grading."""
    build_index()


if __name__ == "__main__":
    build_offline_index()
    print("Index built under artifacts/. Run: python scripts/eval_public.py")
