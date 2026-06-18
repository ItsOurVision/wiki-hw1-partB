"""Development-only: sweep retrieval hyper-parameters on the public queries.

Not part of the graded path. The dense/BM25/chunk artifacts are fixed, so this
only varies the *retrieval* knobs (fusion weights, candidate count, CE blend)
and re-scores -- no rebuild needed. Results are printed and saved to
artifacts-dev/sweep_results.json for the video's empirical plots.
"""
from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import retrieve
from eval import load_query_file, mean_ndcg_at_k
from utils import PUBLIC_QUERIES_PATH

OUT = ROOT / "artifacts-dev"
OUT.mkdir(exist_ok=True)


def score() -> float:
    rows = load_query_file(PUBLIC_QUERIES_PATH)
    queries = [r["query"] for r in rows]
    gt = [r["relevant_page_ids"] for r in rows]
    ranked = retrieve.search_batch(queries)
    return mean_ndcg_at_k(ranked, gt)


def run_grid(grid: dict) -> list:
    keys = list(grid)
    results = []
    for combo in itertools.product(*grid.values()):
        cfg = dict(zip(keys, combo))
        for k, v in cfg.items():
            setattr(retrieve, k, v)
        t = time.time()
        ndcg = score()
        results.append({**cfg, "ndcg": round(ndcg, 4), "sec": round(time.time() - t, 1)})
        print(f"{cfg} -> NDCG@10={ndcg:.4f}  ({results[-1]['sec']}s)")
    return results


if __name__ == "__main__":
    grid = {
        "W_BM25": [0.4, 0.5, 0.6],
        "W_DENSE": [0.3],
        "W_CHUNK": [0.2],
        "CAND_M": [60, 100],
        "CE_TOPK": [10, 15],
    }
    res = sorted(run_grid(grid), key=lambda r: -r["ndcg"])
    (OUT / "sweep_results.json").write_text(json.dumps(res, indent=2))
    print("\nTOP 5:")
    for r in res[:5]:
        print(r)
