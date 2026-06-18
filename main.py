"""Grader entry point.

At grading time only :func:`run` is called, once, with the full query batch; it
loads the prebuilt artifacts and returns a ranked page-id list per query.
:func:`build_offline_index` is the untimed offline step that creates those
artifacts (invoked by ``scripts/build_index.py``); it is never run at grading.
"""
import runtime  # noqa: F401  - OpenMP guard, must precede faiss/torch


def run(queries: list[str]) -> list[list[int]]:
    """Return a best-first ranked list of page_id for each query."""
    from retrieve import rank_queries
    return rank_queries(queries)


def build_offline_index() -> None:
    """Build artifacts/ from the corpus (run locally; not timed at grading)."""
    from index import build_index
    build_index()


if __name__ == "__main__":
    build_offline_index()
    print("artifacts/ ready - run: python scripts/eval_public.py")
