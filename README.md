# Hybrid retrieval over a Wikipedia corpus

**Authors:** Daniel Kats (207730854) · Noga Nagel (322586082)

This project answers a search query by returning the `page_id`s of the Wikipedia
entries most likely to be relevant, ordered best-first. Quality is measured with
mean **NDCG@10** over a held-out query set. The corpus has ~27,000 pages; a query
may have several correct answers.

The system is split in two: an **offline build** that turns the corpus into a few
compact files, and a **fast online lookup** that the grader calls. Only the lookup
is timed, and it never rebuilds anything — it just reads what the build produced.

---

## Running it

Dependencies (NumPy, sentence-transformers, FAISS):

```bash
pip install -r requirements.txt
```

Then evaluate on the public queries — the prebuilt index is already in the repo,
so no rebuild is needed:

```bash
python scripts/eval_public.py
```

This prints mean NDCG@10. (MiniLM and the reranker weights download from the
Hugging Face hub the first time they are used.)

---

## What the timed lookup does (`main.run` → `retrieve.py`)

For one batch of queries:

1. **Encode** each query with MiniLM into the shared 384-d space.
2. **Three relevance scores per page** are computed: a whole-page cosine, a
   keyword **BM25** score, and the cosine of the page's single best-matching
   *passage* (pulled from the passage index).
3. **A short candidate list** is formed from the strongest BM25 pages (or the
   strongest dense pages when a query has no keyword overlap), plus any page that
   owns a high-scoring passage.
4. **The three scores are merged with reciprocal rank fusion** — each score votes
   for a page by its *rank*, and the votes are summed. Because it works on ranks,
   there are no blending weights to tune (and therefore none to overfit).
5. **A cross-encoder re-reads the top 12** query-page pairs jointly and its score
   is mixed 0.85 / 0.15 with the fusion score. If the reranker can't be loaded,
   the fusion order is used as-is rather than failing.
6. The de-duplicated ranking is returned (the grader reads the first 10).

## What the offline build does (`scripts/build_index.py` → `index.py`)

Run once on a full machine; the resulting `artifacts/` are committed.

- **Passages.** Each page is sliced into overlapping ~180-word windows (40-word
  overlap) with the page title repeated on every window, so a query can match a
  specific paragraph instead of an averaged-out whole page (`chunk.py`).
- **Vectors.** Passages and pages are encoded with
  `sentence-transformers/all-MiniLM-L6-v2`, L2-normalized so a dot product is a
  cosine (`embed.py`).
- **Three stored signals.** A **product-quantized** FAISS index over the passage
  vectors, one dense vector per page, and a page-level BM25 model whose posting
  weights are pre-computed so query scoring is a plain sum.

The passage index is product-quantized on purpose: a flat float32 index is
~670 MB, but the quantized one is ~42 MB at the same retrieval quality. That keeps
**every file under 100 MB, so the repository needs no Git LFS** and clones ready
to run.

---

## Files under `artifacts/`

| file | holds | used for |
|------|-------|----------|
| `chunk.faiss` | passage vectors, product-quantized (`IndexPQ`, ~42 MB) | passage cosine signal |
| `chunk_pages.npy` | passage-row → page_id | mapping passages back to pages |
| `page_vecs.npy` | one 384-d vector per page | whole-page cosine signal |
| `page_ids.npy` | page_ids aligned to the rows above | output ids / BM25 rows |
| `bm25.npz` | BM25 postings with weights baked in | keyword signal |
| `bm25_vocab.json` | term → id | BM25 lookup |
| `page_texts.json` | page title+text (≤400 words) | cross-encoder input |
| `meta.json` | sizes and build parameters | reference |

---

## Rebuilding from the raw corpus (optional)

The corpus folder `data/Wikipedia Entries/` is **not** in the repo — it is course
input data and the lookup never reads it (it reads `artifacts/`). To regenerate
the artifacts, drop the corpus into that folder and run:

```bash
python scripts/build_index.py
```

---

## Why these choices (decided by measurement)

On the public set, mean NDCG@10 grew as:

| version | NDCG@10 |
|---|---|
| dense vectors only | 0.369 |
| hybrid without the reranker | 0.383 |
| hybrid without the passage signal | 0.381 |
| full system, weight-tuned fusion | 0.445 |
| **full system, reciprocal rank fusion** | **0.448** |

Dropping either the reranker or the passage signal costs about 0.05 each, so both
stay. Rank fusion beat a hand-tuned weighted blend while removing the tunable
weights, which should transfer better to unseen queries. Single-answer queries
score ~0.77; multi-answer queries are the hard part (~0.22), where the limit is
fitting many correct pages into the top 10. The whole batch runs in a few seconds,
far inside the 60-second budget.

## Files in this repo

```
main.py            run(queries) entry point + offline build hook
chunk.py           page → overlapping passages
embed.py           MiniLM encoder (shared by build and query)
index.py           build + load of the on-disk signals
retrieve.py        the timed lookup (fusion + rerank)
runtime.py         OpenMP guard so FAISS and torch coexist safely
utils.py           paths and corpus/query helpers
eval.py            NDCG@10 scoring (course file, unmodified)
scripts/           eval_public.py, build_index.py (course files)
artifacts/         the committed prebuilt index
```

## Presentation

Video (≤ 3 min): https://youtu.be/zI6FRaDmKSE
