from __future__ import annotations

import json
from typing import Dict, List, Tuple, Optional
from pathlib import Path

import numpy as np
import networkx as nx
from networkx.readwrite import json_graph
from sentence_transformers import SentenceTransformer

from .io_paths import (
    GLOBAL_GRAPH_JSON, SEED_SUBGRAPH_JSON, ALL_CHUNK_SIMS_NPY, ALL_CHUNK_SIMS_META,
    SEED_CHUNK_SCORES_JSON
)
from .io_paths import CHUNKS_INDEX_JSON
from .vector_io import (
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
    load_meta, load_edges_matrix, load_index_edges, compute_or_load_all_similarities,
    normalize_vector
)

def load_global_graph(path: Path = GLOBAL_GRAPH_JSON) -> nx.MultiGraph:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return json_graph.node_link_graph(payload, multigraph=True)

def save_graph(graph: nx.MultiGraph, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json_graph.node_link_data(graph, edges="links")
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

def compute_query_vector(question: str, model_path: Path, device: Optional[str] = None) -> np.ndarray:
    model = SentenceTransformer(str(model_path))
    if device:
        model = model.to(device)
    q = model.encode([question], convert_to_numpy=True, normalize_embeddings=False)[0]
    return normalize_vector(q)

def build_seed_subgraph(
    question: str,
    model_path: Path,
    top_k: int,
    device: Optional[str] = None,
    save_all_scores: bool = True,
) -> Tuple[nx.MultiGraph, List[str], Dict[str, float]]:
    meta = load_meta()
    normalized_edges = bool(meta.get("normalized", True))

    q_vec = compute_query_vector(question, model_path, device=device)

    scores_vec = compute_or_load_all_similarities(
        question, q_vec, normalized_edges, model_tag=str(model_path),
        save_meta_path=(ALL_CHUNK_SIMS_META if save_all_scores else None)
    )

    if save_all_scores:
        np.save(ALL_CHUNK_SIMS_NPY, scores_vec.astype(np.float32))

    chunk_ids, rows = load_index_edges()
    scores_by_chunk: Dict[str, float] = {}
    for cid, r in zip(chunk_ids, rows):
        if 0 <= r < len(scores_vec):
            scores_by_chunk[cid] = float(scores_vec[r])

    sorted_items = sorted(scores_by_chunk.items(), key=lambda x: x[1], reverse=True)
    seed_chunk_ids = [cid for cid, _ in sorted_items[:max(1, top_k)]]

    seed_scores = {cid: scores_by_chunk[cid] for cid in seed_chunk_ids}
    SEED_CHUNK_SCORES_JSON.write_text(json.dumps(seed_scores, ensure_ascii=False, indent=2), encoding="utf-8")

    GG = load_global_graph(GLOBAL_GRAPH_JSON)

    for u, v, key, data in GG.edges(keys=True, data=True):
        cid = data.get("chunk_id")
        if cid in scores_by_chunk:
            data["weight"] = scores_by_chunk[cid]
        else:
            data["weight"] = 0.0

    seed_edges = [
        (u, v, key)
        for u, v, key, data in GG.edges(keys=True, data=True)
        if data.get("chunk_id") in seed_chunk_ids
    ]
    if not seed_edges:
        raise ValueError("No edges found for seed chunks. Check index.csv <-> global_graph chunk_id consistency.")

    G0 = GG.edge_subgraph(seed_edges).copy()
    save_graph(G0, SEED_SUBGRAPH_JSON)
    return G0, seed_chunk_ids, seed_scores
