"""Dev-only: test compressing the chunk index with Product Quantization.

Reconstructs the vectors from the existing flat index (no re-embedding), trains
PQ indexes of varying byte-budgets, and re-scores the full pipeline so we can see
whether a <100 MB (LFS-free) chunk index keeps the chunk channel's NDCG benefit.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import faiss
import numpy as np

import retrieve
from eval import load_query_file, mean_ndcg_at_k
from index import ARTIFACTS_DIR, _read_faiss, _write_faiss, CHUNK_PAGES
from utils import PUBLIC_QUERIES_PATH

ROWS = load_query_file(PUBLIC_QUERIES_PATH)
QUERIES = [r["query"] for r in ROWS]
GT = [r["relevant_page_ids"] for r in ROWS]
CHUNK_PAGE_IDS = np.load(ARTIFACTS_DIR / CHUNK_PAGES)


def eval_with(index) -> float:
    retrieve._CHUNK = (index, CHUNK_PAGE_IDS)   # inject the index under test
    return mean_ndcg_at_k(retrieve.search_batch(QUERIES), GT)


def main() -> None:
    flat = _read_faiss(ARTIFACTS_DIR / "chunk.faiss")
    n, d = flat.ntotal, flat.d
    print(f"flat index: {n} x {d}, {n*d*4/1e6:.0f} MB")
    print(f"flat NDCG@10 = {eval_with(flat):.4f}\n")

    vecs = flat.reconstruct_n(0, n)             # recover the raw vectors
    for m in (64, 96, 128):                     # bytes per vector after PQ
        t = time.time()
        pq = faiss.IndexPQ(d, m, 8, faiss.METRIC_INNER_PRODUCT)
        pq.train(vecs)
        pq.add(vecs)
        size_mb = m * n / 1e6
        ndcg = eval_with(pq)
        print(f"PQ m={m:3d}  ~{size_mb:5.0f} MB  NDCG@10={ndcg:.4f}  (train+eval {time.time()-t:.0f}s)")
        if m == 96:
            _write_faiss(pq, ARTIFACTS_DIR / "chunk_pq96.faiss")


if __name__ == "__main__":
    main()
