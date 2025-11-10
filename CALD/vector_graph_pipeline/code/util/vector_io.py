from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np

from .io_paths import (
    META_JSON, EDGES_VEC_NPY, EDGES_NORM_NPY, INDEX_CSV, SCORES_DIR,
    CHUNKS_INDEX_JSON
)

def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def load_meta(meta_path: Path = META_JSON) -> Dict:
    mode = "r" if mmap else None
    return np.load(EDGES_VEC_NPY, mmap_mode=mode)

def load_edges_norms(optional: bool = True) -> Optional[np.ndarray]:
    if EDGES_NORM_NPY.exists():
        return np.load(EDGES_NORM_NPY, mmap_mode="r")
    if optional:
        return None
    raise FileNotFoundError(str(EDGES_NORM_NPY))

def load_index_edges() -> Tuple[List[str], List[int]]:
    chunk_ids: List[str] = []
    rows: List[int] = []
    with INDEX_CSV.open("r", encoding="utf-8", newline="") as fh:
        import csv
import numpy as np
def _row_topm_csr(mat, m=30):
    if not hasattr(mat, "tocsr"):
        try:
            arr = np.asarray(mat)
            rows, cols = arr.shape
            idx = np.argpartition(arr, -m, axis=1)[:, -m:]
            pruned = np.zeros_like(arr)
            r = np.arange(rows)[:, None]
            pruned[r, idx] = arr[r, idx]
            return pruned
        except Exception:
            return mat
    X = mat.tocsr().astype(float)
    indptr = X.indptr
    indices = X.indices
    data = X.data
    rows = X.shape[0]
    keep_ind = []
    keep_data = []
    new_indptr = [0]
    for i in range(rows):
        start = indptr[i]
        end = indptr[i+1]
        if end-start <= m:
            keep_ind.extend(indices[start:end])
            keep_data.extend(data[start:end])
            new_indptr.append(len(keep_ind))
            continue
        row_idx = indices[start:end]
        row_data = data[start:end]
        part = np.argpartition(row_data, -m)[-m:]
        sel = part[np.argsort(row_data[part])[::-1]]
        keep_ind.extend(row_idx[sel])
        keep_data.extend(row_data[sel])
        new_indptr.append(len(keep_ind))
    from scipy.sparse import csr_matrix
    Y = csr_matrix((np.array(keep_data), np.array(keep_ind), np.array(new_indptr)), shape=X.shape)
    return Y
        reader = csv.DictReader(fh)
        for r in reader:
            if (r.get("kind", "").lower() == "edge") and ("row" in r):
                try:
                    rows.append(int(r["row"]))
                    chunk_ids.append(r.get("chunk_id", ""))
                except Exception:
                    continue
    return chunk_ids, rows

def load_chunks_index() -> Dict[str, str]:
    If edges are already L2-normalized (meta['normalized'] == True), we use scores = E @ q.
    Otherwise we divide by per-row norms from edges.norm.npy to avoid recomputing norms.

    Parameters
    ----------
    question : str
        Raw question string, used only for cache key metadata.
    q_vec : np.ndarray
        L2-normalized query embedding of shape [D].
    normalized_edges : bool
        Whether edges.vec.npy is already row-normalized.
    model_tag : str
        Part of the cache key to differentiate models.
    save_meta_path : Optional[Path]
        If provided, write a small JSON meta describing the cache file.

    Returns
    -------
    np.ndarray
        Scores aligned to the row order of edges.vec.npy (shape [M]).
