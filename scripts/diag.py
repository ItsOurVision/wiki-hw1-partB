"""Development-only diagnostic: where do we lose NDCG on the public queries?

Breaks results down by single- vs multi-relevant queries and reports recall at
the top-10 (scored) and at the full returned depth, so we can tell candidate-
recall failures (relevant page never retrieved) apart from ranking failures
(retrieved but below rank 10).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import retrieve
from eval import load_query_file, ndcg_at_k
from utils import PUBLIC_QUERIES_PATH


def main() -> None:
    rows = load_query_file(PUBLIC_QUERIES_PATH)
    queries = [r["query"] for r in rows]
    gts = [r["relevant_page_ids"] for r in rows]
    ranked = retrieve.search_batch(queries)

    groups = {"all": [], "single |rel|=1": [], "multi |rel|>1": []}
    for q, rel, res in zip(queries, gts, ranked):
        ndcg = ndcg_at_k(res, rel)
        top10 = set(res[:10])
        full = set(res)
        rec10 = len(top10 & rel) / len(rel)
        recfull = len(full & rel) / len(rel)
        rec_but_low = len((full & rel) - top10)   # retrieved but rank > 10
        row = (ndcg, rec10, recfull, rec_but_low, len(rel))
        groups["all"].append(row)
        groups["single |rel|=1" if len(rel) == 1 else "multi |rel|>1"].append(row)

    print(f"{'group':16s} {'n':>3} {'NDCG':>7} {'Rec@10':>7} {'RecFull':>8} {'rank>10':>8}")
    for name, rowsg in groups.items():
        if not rowsg:
            continue
        n = len(rowsg)
        avg = lambda i: sum(r[i] for r in rowsg) / n
        print(f"{name:16s} {n:>3} {avg(0):>7.4f} {avg(1):>7.4f} {avg(2):>8.4f} {sum(r[3] for r in rowsg):>8.0f}")

    print("\nworst queries (NDCG, |rel|, recFull):")
    worst = sorted(zip(queries, gts, ranked), key=lambda t: ndcg_at_k(t[2], t[1]))[:8]
    for q, rel, res in worst:
        recfull = len(set(res) & rel) / len(rel)
        print(f"  ndcg={ndcg_at_k(res, rel):.3f} |rel|={len(rel)} recFull={recfull:.2f}  {q[:70]}")


if __name__ == "__main__":
    main()
