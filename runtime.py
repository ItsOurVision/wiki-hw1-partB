"""Process-wide runtime guards. Import this *before* faiss or torch.

faiss (LLVM ``libomp``) and torch (Intel ``libiomp5``) each bundle their own
OpenMP runtime. Loading both into one process aborts with a duplicate-runtime
error on some platforms (notably macOS). Setting this guard before either
library is imported lets them coexist; faiss is additionally pinned to a single
OpenMP thread (see index.py) so its parallel search cannot race the torch
threadpool. The retrieval workload is tiny, so single-threaded faiss costs
nothing measurable.
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
