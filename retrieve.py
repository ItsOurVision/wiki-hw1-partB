"""Query-time search — the only timed code path.

A :class:`Searcher` wires the loaded signal scorers together: it asks each signal
for its view of the corpus, narrows to a candidate set, merges the views with
reciprocal rank fusion, and refines the very top with a cross-encoder. The bundle
and the reranker are built once and reused for the whole batch; if the reranker
cannot be loaded, the fusion order is returned instead of raising.
"""
from __future__ import annotations

import runtime  # noqa: F401  - OpenMP guard, must precede faiss/torch

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from embed import encode_queries
from index import IndexBundle, load_index
from signals import rank_fusion, rescale, tokenize

# --- knobs (fixed by offline experiments) ------------------------------------
BM25_SHORTLIST = 100         # BM25 leaders entering the candidate set
PASSAGE_PROBE = 256          # passages pulled from FAISS before pooling to pages
OUTPUT_LIMIT = 50            # ranked pages returned per query (top 10 are scored)
RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_TOP = 12              # how deep the cross-encoder re-scores
RERANK_MIX = 0.85            # cross-encoder share when mixed with fusion


class Searcher:
    """Composes the bundle's three signals into one ranking per query."""

    def __init__(self, bundle: IndexBundle):
        self.b = bundle
        self._ce = None
        self._ce_dead = False

    def _reranker(self):
        if self._ce is None and not self._ce_dead:
            try:
                from sentence_transformers import CrossEncoder
                self._ce = CrossEncoder(RERANKER)
            except Exception as err:  # pragma: no cover - environment dependent
                print(f"[retrieve] reranker unavailable ({err}); fusion-only ranking")
                self._ce_dead = True
        return self._ce

    def _candidates(self, page_cos: np.ndarray, keyword: np.ndarray,
                    passage: Dict[int, float]) -> np.ndarray:
        picked = np.argsort(-keyword)[:BM25_SHORTLIST]
        if not bool((keyword[picked] > 0).any()):      # query had no lexical overlap
            picked = np.argsort(-page_cos)[:BM25_SHORTLIST]
        owners = [self.b.row_of[p] for p in passage if p in self.b.row_of]
        if owners:
            picked = np.union1d(picked, np.asarray(owners, dtype=picked.dtype))
        return picked

    def _one(self, text: str, qvec: np.ndarray) -> List[int]:
        b = self.b
        page_cos = b.dense.score(qvec)
        keyword = b.lexical.score(tokenize(text))
        passage = b.passages.best_per_page(qvec, PASSAGE_PROBE)

        cand = self._candidates(page_cos, keyword, passage)
        passage_col = np.fromiter(
            (passage.get(b.page_ids[int(r)], 0.0) for r in cand),
            dtype=np.float32, count=cand.size)
        fused = rank_fusion([page_cos[cand], keyword[cand], passage_col])
        order = np.argsort(-fused)
        sequence = cand[order]
        fused_desc = fused[order]

        ce = self._reranker()
        if ce is not None and sequence.size:
            depth = min(RERANK_TOP, sequence.size)
            front = sequence[:depth]
            pairs = [(text, b.page_text[int(r)]) for r in front]
            try:
                ce_raw = np.asarray(ce.predict(pairs, show_progress_bar=False))
                blend = RERANK_MIX * rescale(ce_raw) + (1.0 - RERANK_MIX) * rescale(fused_desc[:depth])
                sequence = np.concatenate([front[np.argsort(-blend)], sequence[depth:]])
            except Exception as err:  # pragma: no cover
                print(f"[retrieve] rerank skipped ({err}); keeping fusion order")

        ordered_ids = (b.page_ids[int(r)] for r in sequence)
        return list(dict.fromkeys(ordered_ids))[:OUTPUT_LIMIT]   # de-dup, keep order

    def all(self, queries: List[str]) -> List[List[int]]:
        qvecs = encode_queries(queries)
        if qvecs.size == 0:
            return [[] for _ in queries]
        qvecs = np.ascontiguousarray(qvecs, dtype=np.float32)
        return [self._one(q, qvecs[i]) for i, q in enumerate(queries)]


_SEARCHER: Optional[Searcher] = None


def rank_queries(queries: List[str], *, artifacts_dir: Optional[Path] = None) -> List[List[int]]:
    """Rank every query; returns one best-first list of page_id per query."""
    global _SEARCHER
    if _SEARCHER is None:
        _SEARCHER = Searcher(load_index(artifacts_dir))
    return _SEARCHER.all(queries)
