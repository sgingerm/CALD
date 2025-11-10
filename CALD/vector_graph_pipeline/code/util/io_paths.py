from __future__ import annotations
from pathlib import Path
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

KG_OUT = Path(r"D:\kg_out2")
VECTOR_DIR = KG_OUT / "vector_graph"

GLOBAL_GRAPH_JSON = KG_OUT / "global_graph.json"
INDEX_CSV         = VECTOR_DIR / "index.csv"
EDGES_VEC_NPY     = VECTOR_DIR / "edges.vec.npy"
EDGES_NORM_NPY    = VECTOR_DIR / "edges.norm.npy"
NODES_VEC_NPY     = VECTOR_DIR / "nodes.vec.npy"
META_JSON         = VECTOR_DIR / "meta.json"
CHUNKS_INDEX_JSON = KG_OUT / "chunks_index.json"

SCORES_DIR              = VECTOR_DIR / "scores"
OUTPUT_DIR              = KG_OUT
SEED_SUBGRAPH_JSON      = OUTPUT_DIR / "seed_subgraph.json"
APPNP_SUBGRAPH_JSON     = OUTPUT_DIR / "appnp_subgraph.json"

ALL_CHUNK_SIMS_NPY      = OUTPUT_DIR / "all_chunk_similarities.npy"
ALL_CHUNK_SIMS_META     = OUTPUT_DIR / "all_chunk_similarities.meta.json"
SEED_CHUNK_SCORES_JSON  = OUTPUT_DIR / "seed_chunk_scores.json"
APPNP_NODE_SCORES_JSON  = OUTPUT_DIR / "appnp_node_scores.json"

RESULT_JSON             = OUTPUT_DIR / "vector_pipeline_result.json"

EMBED_MODEL_DIR   = Path(r"D:\score\weitiaomoxing2(hao!)")
RERANKER_DIR = r"D:\models\bge_reranker_large_ce"



def ensure_dirs() -> None:
    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
