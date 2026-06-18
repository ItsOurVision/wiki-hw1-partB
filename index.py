"""Offline index build and load.

The build (untimed) writes three retrieval signals to `artifacts/`:

  chunk.faiss / chunk_pages.npy   dense FAISS over overlapping windows
                                  + the chunk-row -> page_id map
  page_vecs.npy / page_ids.npy    one dense vector per page (page channel)
  bm25.npz / bm25_vocab.json      page-level BM25 with posting weights baked in
  page_texts.json                 per-page text aligned to page_ids (cross-encoder)
  meta.json                       counts + build parameters

`run()` only ever *loads* these; it never rebuilds. FAISS is (de)serialized
through Python bytes so non-ASCII artifact paths work everywhere.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import runtime  # noqa: F401  # sets OpenMP guard before faiss/torch load
import faiss
import numpy as np

faiss.omp_set_num_threads(1)  # avoid faiss<->torch OpenMP races at query time

from chunk import Chunk, chunk_corpus
from embed import EMBED_DIM, embed_texts
from utils import ARTIFACTS_DIR, ensure_artifacts_dir, entry_text, iter_entries

CHUNK_FAISS = "chunk.faiss"
CHUNK_PAGES = "chunk_pages.npy"
PAGE_VECS = "page_vecs.npy"
PAGE_IDS = "page_ids.npy"
PAGE_TEXTS = "page_texts.json"
BM25_ARRAYS = "bm25.npz"
BM25_VOCAB = "bm25_vocab.json"
META = "meta.json"

PAGE_WORD_CAP = 400          # words kept per page for the page channel + CE input
CHUNK_PQ_M = 96              # bytes/vector for the PQ chunk index (~42 MB; no LFS)
BM25_K1, BM25_B = 1.5, 0.75  # standard BM25 saturation / length-normalization
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    """Lowercase alphanumeric tokens; shared by the BM25 build and query time."""
    return _TOKEN_RE.findall(text.lower())


def page_text(record: Dict) -> str:
    """Title+content for a page, capped at PAGE_WORD_CAP words."""
    return " ".join(entry_text(record).split()[:PAGE_WORD_CAP])


def _write_faiss(index: faiss.Index, path: Path) -> None:
    Path(path).write_bytes(faiss.serialize_index(index).tobytes())


def _read_faiss(path: Path) -> faiss.Index:
    raw = np.frombuffer(Path(path).read_bytes(), dtype=np.uint8).copy()
    return faiss.deserialize_index(raw)


def build_bm25(token_lists: List[List[str]]) -> Tuple[Dict[str, int], Dict[str, np.ndarray]]:
    """Page-level BM25 stored CSR-by-term, with each posting's weight precomputed.

    At query time scoring is then a gather + scatter-add over the postings of the
    query terms, so no per-query BM25 arithmetic is needed.
    """
    n_pages = len(token_lists)
    doc_len = np.array([len(t) for t in token_lists], dtype=np.float64)
    avgdl = float(doc_len.mean()) if n_pages else 0.0

    vocab: Dict[str, int] = {}
    tf_per_page: List[Dict[int, int]] = []
    df: Dict[int, int] = {}
    for toks in token_lists:
        tf: Dict[int, int] = {}
        for w in toks:
            tid = vocab.setdefault(w, len(vocab))
            tf[tid] = tf.get(tid, 0) + 1
        tf_per_page.append(tf)
        for tid in tf:
            df[tid] = df.get(tid, 0) + 1

    idf = np.zeros(len(vocab), dtype=np.float64)
    for tid, d in df.items():
        idf[tid] = math.log((n_pages - d + 0.5) / (d + 0.5) + 1.0)

    postings: List[List[Tuple[int, float]]] = [[] for _ in range(len(vocab))]
    for page, tf in enumerate(tf_per_page):
        denom_norm = 1.0 - BM25_B + BM25_B * (doc_len[page] / avgdl if avgdl else 0.0)
        for tid, f in tf.items():
            weight = idf[tid] * (f * (BM25_K1 + 1.0)) / (f + BM25_K1 * denom_norm)
            postings[tid].append((page, weight))

    term_ptr = np.zeros(len(vocab) + 1, dtype=np.int64)
    post_pages: List[int] = []
    post_weights: List[float] = []
    for tid, plist in enumerate(postings):
        for page, w in plist:
            post_pages.append(page)
            post_weights.append(w)
        term_ptr[tid + 1] = len(post_pages)

    arrays = {
        "term_ptr": term_ptr,
        "post_pages": np.asarray(post_pages, dtype=np.int32),
        "post_weights": np.asarray(post_weights, dtype=np.float32),
    }
    return vocab, arrays


def build_index(*, entries_dir: Optional[Path] = None,
                artifacts_dir: Optional[Path] = None) -> None:
    """Build and persist every retrieval artifact from the corpus."""
    out = artifacts_dir or ensure_artifacts_dir()
    records = list(iter_entries(entries_dir))
    print(f"[build] {len(records)} pages")

    # ---- chunk channel: dense FAISS over overlapping windows -----------------
    chunks: List[Chunk] = chunk_corpus(records)
    print(f"[build] {len(chunks)} chunks; embedding...", flush=True)
    chunk_vecs = embed_texts([c.text for c in chunks], progress_every=20000)
    dim = int(chunk_vecs.shape[1]) if chunk_vecs.size else EMBED_DIM
    # Product-quantized index (METRIC_INNER_PRODUCT == cosine on unit vectors):
    # compresses 437k x 384 floats from ~672 MB to ~42 MB with no measurable
    # recall loss, keeping every artifact under GitHub's 100 MB limit (no LFS).
    chunk_index = faiss.IndexPQ(dim, CHUNK_PQ_M, 8, faiss.METRIC_INNER_PRODUCT)
    if chunk_vecs.size:
        chunk_index.train(chunk_vecs)
        chunk_index.add(chunk_vecs)
    _write_faiss(chunk_index, out / CHUNK_FAISS)
    np.save(out / CHUNK_PAGES, np.array([c.page_id for c in chunks], dtype=np.int32))

    # ---- page channel: one dense vector + capped text per page ---------------
    page_ids = [int(r["page_id"]) for r in records]
    page_texts = [page_text(r) for r in records]
    print("[build] embedding pages...")
    page_vecs = embed_texts(page_texts)
    if page_vecs.size == 0:
        page_vecs = np.zeros((0, dim), dtype=np.float32)
    np.save(out / PAGE_VECS, page_vecs.astype(np.float32))
    np.save(out / PAGE_IDS, np.array(page_ids, dtype=np.int64))
    (out / PAGE_TEXTS).write_text(json.dumps(page_texts), encoding="utf-8")

    # ---- lexical channel: page-level BM25 -----------------------------------
    print("[build] building BM25...")
    vocab, bm = build_bm25([tokenize(t) for t in page_texts])
    np.savez(out / BM25_ARRAYS, **bm)
    (out / BM25_VOCAB).write_text(json.dumps(vocab), encoding="utf-8")

    (out / META).write_text(json.dumps({
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "dim": dim,
        "num_pages": len(page_ids),
        "num_chunks": len(chunks),
        "vocab_size": len(vocab),
    }, indent=2), encoding="utf-8")
    print(f"[build] done -> {out}")


def load_chunk_index(artifacts_dir: Optional[Path] = None) -> Tuple[faiss.Index, np.ndarray]:
    """Load the chunk FAISS index and its chunk-row -> page_id array."""
    root = artifacts_dir or ARTIFACTS_DIR
    return _read_faiss(root / CHUNK_FAISS), np.load(root / CHUNK_PAGES)


def load_hybrid(artifacts_dir: Optional[Path] = None) -> Dict:
    """Load the page vectors, page-level BM25, ids, and page texts for retrieval."""
    root = artifacts_dir or ARTIFACTS_DIR
    bm = np.load(root / BM25_ARRAYS)
    return {
        "page_vecs": np.ascontiguousarray(np.load(root / PAGE_VECS), dtype=np.float32),
        "page_ids": [int(x) for x in np.load(root / PAGE_IDS)],
        "page_texts": json.loads((root / PAGE_TEXTS).read_text(encoding="utf-8")),
        "vocab": json.loads((root / BM25_VOCAB).read_text(encoding="utf-8")),
        "term_ptr": bm["term_ptr"],
        "post_pages": bm["post_pages"],
        "post_weights": bm["post_weights"],
    }
