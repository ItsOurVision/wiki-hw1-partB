"""Grader entry point for the retrieval system.

At grading time only :func:`run` is called, once, with the full list of queries;
it loads the prebuilt artifacts and returns a ranked page-id list per query.
:func:`build_offline_index` is the untimed offline step used to create those
artifacts on our own machine (invoked by ``scripts/build_index.py``).
"""
from __future__ import annotations

import runtime  # noqa: F401  - installs the OpenMP guard before faiss/torch import

from typing import List


def run(queries: List[str]) -> List[List[int]]:
    """Return a best-first ranked list of page_id for each query string."""
    from retrieve import search_batch
    return search_batch(queries)


def build_offline_index() -> None:
    """Construct ``artifacts/`` from the corpus (run locally; not timed at grading)."""
    from index import build_index
    build_index()


if __name__ == "__main__":
    build_offline_index()
    print("artifacts/ ready — run: python scripts/eval_public.py")
