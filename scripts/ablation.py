"""Development-only: per-stage ablation on the public queries.

Measures the marginal contribution of each retrieval channel by disabling one at
a time (via retrieve's module-level knobs). Drives two decisions: whether the
chunk channel earns its on-disk cost, and the empirical story for the video.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import retrieve
from eval import load_query_file, mean_ndcg_at_k
from utils import PUBLIC_QUERIES_PATH

ROWS = load_query_file(PUBLIC_QUERIES_PATH)
QUERIES = [r["query"] for r in ROWS]
GT = [r["relevant_page_ids"] for r in ROWS]

# snapshot defaults
BASE = {k: getattr(retrieve, k) for k in
        ["W_DENSE", "W_BM25", "W_CHUNK", "CHUNK_TOPN", "CE_TOPK", "CAND_M", "W_CE"]}


def measure(label: str, **overrides) -> None:
    for k, v in BASE.items():
        setattr(retrieve, k, v)
    for k, v in overrides.items():
        setattr(retrieve, k, v)
    ndcg = mean_ndcg_at_k(retrieve.search_batch(QUERIES), GT)
    print(f"{label:34s} NDCG@10 = {ndcg:.4f}")


if __name__ == "__main__":
    print(f"defaults: {BASE}\n")
    measure("full (dense+bm25+chunk+CE)")
    measure("no cross-encoder", CE_TOPK=0)
    measure("no chunk channel", CHUNK_TOPN=0, W_CHUNK=0.0)
    measure("no chunk + no CE", CHUNK_TOPN=0, W_CHUNK=0.0, CE_TOPK=0)
    measure("dense only (no bm25,chunk,CE)", W_BM25=0.0, W_CHUNK=0.0, CHUNK_TOPN=0, CE_TOPK=0)
    measure("no dense (bm25+chunk+CE)", W_DENSE=0.0)
