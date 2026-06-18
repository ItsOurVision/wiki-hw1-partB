"""Offline build and loading of the on-disk search structures.

The build runs once and writes the files `run()` needs into ``artifacts/``; the
load step assembles those files into the three signal scorers (see ``signals.py``)
bundled together as an :class:`IndexBundle`. The passage index is product
quantized so the folder stays small enough to ship in a plain git repo (no LFS).
"""
from __future__ import annotations

import runtime  # noqa: F401  - OpenMP guard, must precede faiss/torch

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import faiss
import numpy as np

from chunk import Passage, passages_of
from embed import VECTOR_WIDTH, encode_texts
from signals import DenseSignal, LexicalSignal, PassageSignal, tokenize
from utils import ARTIFACTS_DIR, ensure_artifacts_dir, entry_text, iter_entries

faiss.omp_set_num_threads(1)  # keep faiss off the threads torch/OpenMP also use

PASSAGE_FILE = "chunk.faiss"
OWNER_FILE = "chunk_pages.npy"
PAGEVEC_FILE = "page_vecs.npy"
PAGEID_FILE = "page_ids.npy"
PAGETXT_FILE = "page_texts.json"
BM25_FILE = "bm25.npz"
VOCAB_FILE = "bm25_vocab.json"
META_FILE = "meta.json"

PQ_CODE_BYTES = 96           # PQ sub-quantizers -> ~bytes per passage vector
PAGE_TEXT_WORDS = 400        # words kept per page for the page/rerank signals


def _page_blurb(record: Dict) -> str:
    """Title+body for a page, trimmed to PAGE_TEXT_WORDS words."""
    return " ".join(entry_text(record).split()[:PAGE_TEXT_WORDS])


@dataclass
class IndexBundle:
    """Everything the searcher needs: page metadata + the three signal scorers."""
    page_ids: List[int]
    page_text: List[str]
    dense: DenseSignal
    lexical: LexicalSignal
    passages: PassageSignal
    row_of: Dict[int, int] = field(init=False)

    def __post_init__(self) -> None:
        self.row_of = {pid: i for i, pid in enumerate(self.page_ids)}


def build_index(*, entries_dir: Optional[Path] = None,
                artifacts_dir: Optional[Path] = None) -> None:
    """Build and persist every artifact from the corpus."""
    out = artifacts_dir or ensure_artifacts_dir()
    records = list(iter_entries(entries_dir))
    print(f"[build] {len(records)} pages", flush=True)

    # passage channel -> PQ FAISS
    units: List[Passage] = passages_of(records)
    print(f"[build] {len(units)} passages; encoding", flush=True)
    unit_vecs = encode_texts([u.text for u in units], report_every=20000)
    width = int(unit_vecs.shape[1]) if unit_vecs.size else VECTOR_WIDTH
    pq = faiss.IndexPQ(width, PQ_CODE_BYTES, 8, faiss.METRIC_INNER_PRODUCT)
    if unit_vecs.size:
        pq.train(unit_vecs)
        pq.add(unit_vecs)
    faiss.write_index(pq, str(out / PASSAGE_FILE))
    np.save(out / OWNER_FILE,
            np.fromiter((u.page_id for u in units), dtype=np.int32, count=len(units)))

    # page channel -> dense vectors + capped text
    page_ids = [int(r["page_id"]) for r in records]
    page_text = [_page_blurb(r) for r in records]
    print("[build] encoding pages", flush=True)
    page_matrix = encode_texts(page_text)
    if page_matrix.size == 0:
        page_matrix = np.zeros((0, width), dtype=np.float32)
    np.save(out / PAGEVEC_FILE, page_matrix.astype(np.float32))
    np.save(out / PAGEID_FILE, np.asarray(page_ids, dtype=np.int64))
    (out / PAGETXT_FILE).write_text(json.dumps(page_text), encoding="utf-8")

    # lexical channel -> BM25 CSR (fit lives with the signal)
    print("[build] fitting BM25", flush=True)
    vocab, ptr, post_pages, post_weights = LexicalSignal.fit([tokenize(t) for t in page_text])
    np.savez(out / BM25_FILE, term_ptr=ptr, post_pages=post_pages, post_weights=post_weights)
    (out / VOCAB_FILE).write_text(json.dumps(vocab), encoding="utf-8")

    (out / META_FILE).write_text(json.dumps({
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "dim": width, "num_pages": len(page_ids),
        "num_chunks": len(units), "vocab_size": len(vocab), "pq_bytes": PQ_CODE_BYTES,
    }, indent=2), encoding="utf-8")
    print(f"[build] artifacts written to {out}", flush=True)


def load_index(artifacts_dir: Optional[Path] = None) -> IndexBundle:
    """Read the artifacts and assemble the signal scorers into an IndexBundle."""
    root = artifacts_dir or ARTIFACTS_DIR
    page_ids = [int(x) for x in np.load(root / PAGEID_FILE)]
    page_matrix = np.ascontiguousarray(np.load(root / PAGEVEC_FILE), dtype=np.float32)
    lex = np.load(root / BM25_FILE)
    vocab = json.loads((root / VOCAB_FILE).read_text(encoding="utf-8"))
    return IndexBundle(
        page_ids=page_ids,
        page_text=json.loads((root / PAGETXT_FILE).read_text(encoding="utf-8")),
        dense=DenseSignal(page_matrix),
        lexical=LexicalSignal(vocab, lex["term_ptr"], lex["post_pages"], lex["post_weights"], len(page_ids)),
        passages=PassageSignal(faiss.read_index(str(root / PASSAGE_FILE)), np.load(root / OWNER_FILE)),
    )
