"""Query-time retrieval (timed path; includes query embedding).

For each query:
  1. take the top `CAND_M` pages by BM25 (fall back to dense if no lexical hit),
     widened with the best pages from the chunk index;
  2. fuse three min-max-normalized signals over those candidates --
     page-dense cosine, BM25, and chunk-dense (max-pooled to page);
  3. rerank the top `CE_TOPK` with a cross-encoder, blended with the fusion
     score so a noisy CE cannot bury an already well-ranked page.

The cross-encoder is optional at runtime: if it cannot be loaded (e.g. no hub
access on the grading box) retrieval falls back to the fusion ranking instead of
failing. Artifacts and models load once and are cached across the batch.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import runtime  # noqa: F401  # sets OpenMP guard before faiss/torch load
import numpy as np

from embed import embed_queries
from index import load_chunk_index, load_hybrid, tokenize

# --- tunable config (selected by the offline sweep; see README) --------------
CAND_M = 100                             # BM25 candidate pages fed to fusion
CHUNK_TOPN = 256                         # chunks pulled per query before max-pool
RETURN_K = 50                            # pages returned (only first 10 scored)
W_DENSE, W_BM25, W_CHUNK = 0.2, 0.5, 0.3  # linear fusion weights (offline sweep)
CE_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CE_TOPK = 12                             # candidates reranked by the cross-encoder
W_CE = 0.85                              # CE share in the rerank blend

_H: Optional[Dict] = None
_CHUNK = None
_PID2ROW: Optional[Dict[int, int]] = None
_CE = None
_CE_FAILED = False


def _hybrid(artifacts_dir: Optional[Path]) -> Dict:
    global _H, _PID2ROW
    if _H is None:
        _H = load_hybrid(artifacts_dir)
        _PID2ROW = {pid: i for i, pid in enumerate(_H["page_ids"])}
    return _H


def _chunk(artifacts_dir: Optional[Path]):
    global _CHUNK
    if _CHUNK is None:
        _CHUNK = load_chunk_index(artifacts_dir)
    return _CHUNK


def _cross_encoder():
    """Load the CE once. Returns None (permanently) if it cannot be loaded."""
    global _CE, _CE_FAILED
    if _CE is None and not _CE_FAILED:
        try:
            from sentence_transformers import CrossEncoder
            _CE = CrossEncoder(CE_MODEL)
        except Exception as exc:  # pragma: no cover - environment dependent
            print(f"[retrieve] cross-encoder unavailable, using fusion only: {exc}")
            _CE_FAILED = True
    return _CE


def _minmax(x: np.ndarray) -> np.ndarray:
    """Scale to [0, 1]; a flat vector maps to zeros."""
    if x.size == 0:
        return x
    lo, hi = float(x.min()), float(x.max())
    return np.zeros_like(x) if hi - lo < 1e-12 else (x - lo) / (hi - lo)


def _bm25_scores(tokens: List[str], h: Dict) -> np.ndarray:
    """Page BM25 via gather + scatter-add over precomputed posting weights."""
    scores = np.zeros(len(h["page_ids"]), dtype=np.float32)
    vocab, ptr, pages, wts = h["vocab"], h["term_ptr"], h["post_pages"], h["post_weights"]
    for tok in tokens:
        tid = vocab.get(tok)
        if tid is None:
            continue
        a, b = int(ptr[tid]), int(ptr[tid + 1])
        if b > a:
            np.add.at(scores, pages[a:b], wts[a:b])
    return scores


def _chunk_page_scores(qvec: np.ndarray, artifacts_dir: Optional[Path]) -> Dict[int, float]:
    """Best chunk-dense cosine per page for one query."""
    index, chunk_pages = _chunk(artifacts_dir)
    n = min(CHUNK_TOPN, index.ntotal)
    if n == 0:
        return {}
    sims, rows = index.search(qvec[None, :], n)
    best: Dict[int, float] = {}
    for s, row in zip(sims[0], rows[0]):
        if row < 0:
            continue
        pid = int(chunk_pages[int(row)])
        if s > best.get(pid, -1e9):
            best[pid] = float(s)
    return best


def _rank_one(query: str, qvec: np.ndarray, h: Dict, artifacts_dir: Optional[Path]) -> List[int]:
    page_ids = h["page_ids"]
    # The matmul triggers spurious FP warnings on macOS Accelerate (numpy 2.x);
    # scores are correct, so we silence them here.
    with np.errstate(all="ignore"):
        dense_all = h["page_vecs"] @ qvec
    bm25_all = _bm25_scores(tokenize(query), h)
    chunk_best = _chunk_page_scores(qvec, artifacts_dir)

    # candidate set: BM25 top-M (or dense fallback), widened by chunk winners
    cand = np.argsort(-bm25_all)[:CAND_M]
    if int((bm25_all[cand] > 0).sum()) == 0:
        cand = np.argsort(-dense_all)[:CAND_M]
    crows = [_PID2ROW[p] for p in chunk_best if p in _PID2ROW]
    if crows:
        cand = np.union1d(cand, np.array(crows, dtype=cand.dtype))

    chunk_vec = np.array([chunk_best.get(page_ids[int(r)], 0.0) for r in cand], dtype=np.float32)
    fused = (W_DENSE * _minmax(dense_all[cand])
             + W_BM25 * _minmax(bm25_all[cand])
             + W_CHUNK * _minmax(chunk_vec))
    order = cand[np.argsort(-fused)]
    fused_sorted = np.sort(fused)[::-1]

    # cross-encoder rerank of the head, blended with fusion
    ce = _cross_encoder()
    if ce is not None and len(order):
        k = min(CE_TOPK, len(order))
        head = order[:k]
        pairs = [(query, h["page_texts"][int(r)]) for r in head]
        try:
            ce_scores = np.asarray(ce.predict(pairs, show_progress_bar=False))
            blended = W_CE * _minmax(ce_scores) + (1.0 - W_CE) * _minmax(fused_sorted[:k])
            order = np.concatenate([head[np.argsort(-blended)], order[k:]])
        except Exception as exc:  # pragma: no cover
            print(f"[retrieve] CE predict failed, fusion only: {exc}")

    out, seen = [], set()
    for r in order:
        pid = page_ids[int(r)]
        if pid not in seen:
            seen.add(pid)
            out.append(pid)
        if len(out) >= RETURN_K:
            break
    return out


def search_batch(queries: List[str], *, artifacts_dir: Optional[Path] = None) -> List[List[int]]:
    """One ranked list of page_id (best first) per query."""
    h = _hybrid(artifacts_dir)
    qvecs = embed_queries(queries)
    if qvecs.size == 0:
        return [[] for _ in queries]
    qvecs = np.ascontiguousarray(qvecs, dtype=np.float32)
    return [_rank_one(queries[i], qvecs[i], h, artifacts_dir) for i in range(len(queries))]
