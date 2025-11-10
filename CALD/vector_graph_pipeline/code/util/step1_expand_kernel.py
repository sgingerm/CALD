from __future__ import annotations
import os, json, math, random
from typing import Dict, Iterable, List, Optional, Tuple
import numpy as np
import scipy.sparse as sp
import networkx as nx
from networkx.readwrite import json_graph
from .io_paths import APPNP_SUBGRAPH_JSON, APPNP_NODE_SCORES_JSON
from .seed_retrieval import save_graph

def _load_comm_params(G: nx.MultiDiGraph) -> Tuple[Optional[dict], bool]:
    meta = G.graph.get("meta") if isinstance(G.graph, dict) else None
    cp = meta.get("comm_params") if isinstance(meta, dict) else None
    if isinstance(cp, dict) and cp.get("schema") == "per-community-v1":
        return cp, True
    path = os.environ.get("APPNP_COMM_PARAMS", "").strip()
    if path and os.path.exists(path):
        try:
            cp2 = json.load(open(path, "r", encoding="utf-8"))
            if isinstance(cp2, dict) and cp2.get("schema") == "per-community-v1":
                return cp2, True
        except Exception:
            pass
    return None, False

def _node_comm(G: nx.MultiDiGraph, u: str) -> Optional[str]:
    c = G.nodes[u].get("community", None)
    if c is None:
        return None
    try:
        return str(int(c))
    except Exception:
        return str(c)

class _Embedder:
    def __init__(self, model_dir: Optional[str] = None, batch_size: int = 64, allow_fallback: bool = True, device: Optional[str] = None):
        self.ok = False
        self.model = None
        self.dim = 512
        self.batch = int(batch_size)
        self.allow = allow_fallback
        self.device = device or ("cuda" if os.environ.get("APPNP_DEVICE", "cuda") == "cuda" else "cpu")
        if model_dir:
            try:
                from sentence_transformers import SentenceTransformer
import numpy as np
def _row_topm_csr(mat, m=30):
    from scipy.sparse import csr_matrix
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
    Y = csr_matrix((np.array(keep_data), np.array(keep_ind), np.array(new_indptr)), shape=X.shape)
    return Y
self.model = SentenceTransformer(model_dir, device=self.device)
                z = self.model.encode(["test"], convert_to_numpy=True, normalize_embeddings=True)
                self.dim = int(z.shape[-1])
                self.ok = True
            except Exception:
                raise
    def fit(self, texts: List[str]):
        if self.ok:
            return
        from collections import defaultdict
        df = defaultdict(int)
        for t in texts:
            toks = [w.lower() for w in t.split()]
            for w in set(toks):
                df[w] += 1
        N = max(1, len(texts))
        top = sorted(df.items(), key=lambda x: -x[1])[: self.dim]
        self.vocab = {w: i for i, (w, _) in enumerate(top)}
        for w, c in df.items():
            self.idf[w] = math.log((N + 1) / (c + 1)) + 1.0
        self.dim = max(16, len(self.vocab))
    def encode(self, texts: List[str]) -> np.ndarray:
        if self.ok:
            out = []
            for s in range(0, len(texts), self.batch):
                arr = self.model.encode(texts[s:s+self.batch], convert_to_numpy=True, normalize_embeddings=True)
                out.append(arr.astype(np.float32))
            return np.concatenate(out, axis=0) if out else np.zeros((0, self.dim), dtype=np.float32)
        X = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            toks = [w.lower() for w in t.split()]
            for w in toks:
                j = self.vocab.get(w)
                if j is not None:
                    X[i, j] += self.idf.get(w, 0.0)
            n = float(np.linalg.norm(X[i]) + 1e-12)
            if n > 0:
                X[i] /= n
        return X

def _build_rels_by_ent(G: nx.MultiDiGraph) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for u in G.nodes():
        out[u] = []
    for u, v, k, d in G.edges(keys=True, data=True):
        r = d.get("rel") or d.get("relation") or d.get("r") or ""
        if r:
            out[u].append(str(r)); out[v].append(str(r))
    return out

def _row_stochastic(W: sp.csr_matrix) -> sp.csr_matrix:
    deg = np.array(W.sum(axis=1)).ravel()
    deg[deg == 0.0] = 1.0
    Dinv = sp.diags(1.0 / deg, dtype=np.float32)
    return Dinv @ W

def induce_khop_region(G_full: nx.MultiDiGraph, seed_nodes: Iterable[str], k_hop: int = 1, max_nodes: Optional[int] = None, max_edges: Optional[int] = None) -> nx.MultiDiGraph:
    visited = set(seed_nodes)
    frontier = list(seed_nodes)
    for _ in range(k_hop):
        new_frontier = []
        for u in frontier:
            for _, v, _key in G_full.edges(u, keys=True):
                if v not in visited:
                    visited.add(v)
                    new_frontier.append(v)
        frontier = new_frontier
        if not frontier:
            break
    H = G_full.subgraph(visited).copy()
    if max_nodes is not None and H.number_of_nodes() > max_nodes:
        deg_sorted = sorted(H.degree(), key=lambda x: x[1], reverse=True)[:max_nodes]
        keep = set(n for n, _ in deg_sorted)
        H = H.subgraph(keep).copy()
    if max_edges is not None and H.number_of_edges() > max_edges:
        edges = list(H.edges(keys=True, data=True))
        edges.sort(key=lambda e: float(e[3].get("weight", 0.0)), reverse=True)
        keep_edges = edges[:max_edges]
        H = H.edge_subgraph([(u, v, k) for (u, v, k, _) in keep_edges]).copy()
    return H

def _aggregate_seed_node_scores(G_seed: nx.MultiDiGraph, seed_chunk_scores: Dict[str, float]) -> Dict[str, float]:
    node_scores: Dict[str, float] = {}
    for u, v, key, data in G_seed.edges(keys=True, data=True):
        cid = data.get("chunk_id")
        if cid is None:
            continue
        s = float(seed_chunk_scores.get(cid, 0.0))
        node_scores[u] = node_scores.get(u, 0.0) + s
        node_scores[v] = node_scores.get(v, 0.0) + s
    return node_scores

def _precompute_weights(G: nx.MultiDiGraph, cp: dict, embedder: _Embedder) -> Tuple[sp.csr_matrix, List[str], np.ndarray]:
    nodes = list(G.nodes())
    idx = {u: i for i, u in enumerate(nodes)}
    rels_by_ent = _build_rels_by_ent(G)
    ent_texts = nodes
    embedder.fit(ent_texts)
    E = embedder.encode(ent_texts)
    if E.shape[0] != len(nodes):
        E = np.zeros((len(nodes), embedder.dim), dtype=np.float32)
    nrm = np.linalg.norm(E, axis=1, keepdims=True) + 1e-12
    E = E / nrm
    rel_vocab = sorted({r for lst in rels_by_ent.values() for r in lst})
    if rel_vocab:
        Rv = embedder.encode(rel_vocab)
        Rv = Rv / (np.linalg.norm(Rv, axis=1, keepdims=True) + 1e-12)
        r2i = {r: i for i, r in enumerate(rel_vocab)}
        d = Rv.shape[1]
        R = np.zeros((len(nodes), d), dtype=np.float32)
        has = np.zeros((len(nodes),), dtype=np.int8)
        for u in nodes:
            i = idx[u]
            rels = [r for r in rels_by_ent.get(u, []) if r in r2i]
            if rels:
                M = Rv[[r2i[r] for r in rels]]
                m = M.mean(axis=0)
                m = m / (np.linalg.norm(m) + 1e-12)
                R[i] = m.astype(np.float32)
                has[i] = 1
    else:
        R = np.zeros((len(nodes), 1), dtype=np.float32)
        has = np.zeros((len(nodes),), dtype=np.int8)
    deg = dict(G.degree())
    ldeg = np.log1p(np.array([float(deg.get(u, 0)) for u in nodes], dtype=np.float32))
    byc = cp.get("by_community", {}) if isinstance(cp, dict) else {}
    inter = cp.get("inter", {}) if isinstance(cp, dict) else {}
    rows, cols, data = [], [], []
    for u, v, k, d in G.edges(keys=True, data=True):
        i, j = idx[u], idx[v]
        ui, vj = E[i], E[j]
        gsyn = float(np.dot(ui, vj))
        if gsyn < 0.0:
            gsyn = 0.0
        ri, rj = R[i], R[j]
        if ri.shape[0] == rj.shape[0] and ri.shape[0] > 1:
            grel = float(np.dot(ri, rj))
            grel = 0.5 * (1.0 + grel)
        else:
            grel = 0.5 if (has[i] == 0 and has[j] == 0) else 0.5
        cu = _node_comm(G, u); cv = _node_comm(G, v)
        same = (cu is not None and cv is not None and cu == cv)
        params = byc.get(str(cu), inter) if same else inter
        alpha_ij = float(params.get("alpha", 0.5))
        kappa = float(params.get("kappa", 1.5))
        beta_syn = float(params.get("beta_syn", 0.2))
        beta_rel = float(params.get("beta_rel", 0.2))
        beta_deg1 = float(params.get("beta_deg1", 0.1))
        beta_deg2 = float(params.get("beta_deg2", 0.1))
        gamma0 = float(params.get("gamma0", 0.05))
        s = gamma0 + beta_syn * gsyn + beta_rel * grel + beta_deg1 * float(ldeg[i]) + beta_deg2 * float(ldeg[j])
        z = 1.0 / (1.0 + math.exp(-max(1e-4, kappa) * s))
        w = float(max(0.0, z))
        wij = alpha_ij * w
        rows.append(i); cols.append(j); data.append(wij)
        rows.append(j); cols.append(i); data.append(wij)
    for i in range(len(nodes)):
        rows.append(i); cols.append(i); data.append(1.0)
    A = sp.csr_matrix((data, (rows, cols)), shape=(len(nodes), len(nodes)), dtype=np.float32)
A = _row_topm_csr(A, m=30)
    P = _row_stochastic(A)
    alpha_self = np.zeros((len(nodes),), dtype=np.float32)
    for u in nodes:
        i = idx[u]
        cu = _node_comm(G, u)
        params = byc.get(str(cu), inter)
        a = float(params.get("alpha", 0.5))
        a = float(min(0.9999, max(1e-4, a)))
        alpha_self[i] = a
    return P, nodes, alpha_self

def _build_sparse_row_stochastic_adj_fallback(G: nx.MultiDiGraph) -> Tuple[sp.csr_matrix, List[str], np.ndarray]:
    nodes = list(G.nodes())
    idx = {u: i for i, u in enumerate(nodes)}
    rows, cols, data = [], [], []
    deg_map = dict(G.degree())
    for u, v, k, d in G.edges(keys=True, data=True):
        i, j = idx[u], idx[v]
        rows.append(i); cols.append(j); data.append(1.0)
        rows.append(j); cols.append(i); data.append(1.0)
    for i in range(len(nodes)):
        rows.append(i); cols.append(i); data.append(1.0)
    A = sp.csr_matrix((data, (rows, cols)), shape=(len(nodes), len(nodes)), dtype=np.float32)
A = _row_topm_csr(A, m=30)
    P = _row_stochastic(A)
    alpha_self = np.full((len(nodes),), 0.5, dtype=np.float32)
    return P, nodes, alpha_self

def run_appnp_local(G_full: nx.MultiDiGraph, G_seed: nx.MultiDiGraph, seed_chunk_scores: Dict[str, float], k_hop_region: int = 1, top_nodes: int = 300, alpha: float = 0.5, iterations: int = 10, min_score: float = 0.0, edge_policy: str = "either") -> Tuple[nx.MultiDiGraph, Dict[str, float], List[str]]:
    cp, ok = _load_comm_params(G_full)
    R = induce_khop_region(G_full, G_seed.nodes(), k_hop=k_hop_region, max_nodes=None, max_edges=None)
    p_scores = _aggregate_seed_node_scores(G_seed, seed_chunk_scores)
    if not p_scores:
        raise ValueError("Empty personalization scores from seeds; check seed_chunk_scores.")
    model_dir = os.environ.get("APPNP_EMBEDDER", "").strip()
    bs = int(os.environ.get("APPNP_EMBED_BATCH", "64"))
    allow_fb = os.environ.get("APPNP_NO_FALLBACK", "0") != "1"
    emb = _Embedder(model_dir=model_dir if ok else None, batch_size=bs, allow_fallback=allow_fb)
    if ok:
        try:
            P, nodes, alpha_self = _precompute_weights(R, cp, emb)
        except Exception:
            P, nodes, alpha_self = _build_sparse_row_stochastic_adj_fallback(R)
    else:
        P, nodes, alpha_self = _build_sparse_row_stochastic_adj_fallback(R)
    idx = {u: i for i, u in enumerate(nodes)}
    h0 = np.zeros((len(nodes),), dtype=np.float32)
    for u, s in p_scores.items():
        if u in idx:
            h0[idx[u]] = float(s)
    a_vec = alpha_self if ok else np.full((len(nodes),), float(alpha), dtype=np.float32)
    h = h0.copy()
    for _ in range(int(iterations)):
        h = (1.0 - a_vec) * (P @ h) + a_vec * h0
    order = np.argsort(-h)
    selected: List[str] = []
    for i in order:
        if h[i] < min_score and selected:
            break
        selected.append(nodes[i])
        if len(selected) >= int(top_nodes):
            break
    if not selected and len(order) > 0:
        selected = [nodes[int(order[0])]]
    if edge_policy == "both":
        edges = [(u, v, k) for u, v, k, d in R.edges(keys=True, data=True) if (u in selected and v in selected)]
    else:
        sel = set(selected)
        edges = [(u, v, k) for u, v, k, d in R.edges(keys=True, data=True) if (u in sel or v in sel)]
    if not edges:
        raise ValueError("No edges selected after APPNP; try increasing top_nodes or lowering min_score.")
    G_exp = R.edge_subgraph(edges).copy()
    node_scores = {nodes[i]: float(h[i]) for i in range(len(nodes))}
    with open(APPNP_NODE_SCORES_JSON, "w", encoding="utf-8") as fh:
        json.dump(node_scores, fh, ensure_ascii=False, indent=2)
    save_graph(G_exp, APPNP_SUBGRAPH_JSON)
    return G_exp, node_scores, selected
