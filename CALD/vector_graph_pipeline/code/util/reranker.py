from __future__ import annotations
from typing import List, Tuple, Optional, Protocol, Any, Iterable
from pathlib import Path

def _to_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, (list, tuple)):
        return " ".join(map(_to_str, x))
    if isinstance(x, dict):
        for k in ("text", "content", "passage", "body"):
            v = x.get(k)
            if isinstance(v, str):
                return v
        try:
            import json as _json
            return _json.dumps(x, ensure_ascii=False)
        except Exception:
            return str(x)
    return str(x)

class Reranker(Protocol):
    def compute_score(self, pairs: List[Tuple[str, str]]) -> List[float]: ...

class FlagRerankerWrapper:
    def __init__(self, model_name_or_path: str, use_fp16: Optional[bool] = None, device: Optional[str] = None):
        from FlagEmbedding import FlagReranker
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
        kwargs = {"model_name_or_path": model_name_or_path}
        if use_fp16 is not None:
            kwargs["use_fp16"] = use_fp16
        self._rr = FlagReranker(**kwargs)
        self._device = device

    def compute_score(self, pairs: List[Tuple[str, str]]) -> List[float]:
        if not pairs:
            return []
        cleaned = [( _to_str(q).strip(), _to_str(p).strip() ) for q, p in pairs]
        cleaned = [ (q, p) for q, p in cleaned if q and p ]
        if not cleaned:
            return []
        return list(self._rr.compute_score(cleaned))

class STBiEncoderReranker:
    def __init__(self, model_path: str, device: Optional[str] = None):
        from sentence_transformers import SentenceTransformer
        from torch import device as _dev
        self._st = SentenceTransformer(model_path)
        if device:
            try:
                self._st = self._st.to(device)
            except Exception:
                pass

    def compute_score(self, pairs: List[Tuple[str, str]]) -> List[float]:
        if not pairs:
            return []
        from sentence_transformers import util
        qs = [ _to_str(q).strip() for q, _ in pairs ]
        ps = [ _to_str(p).strip() for _, p in pairs ]
        idx = [ i for i,(q,p) in enumerate(zip(qs, ps)) if q and p ]
        if not idx:
            return []
        qs2 = [qs[i] for i in idx]
        ps2 = [ps[i] for i in idx]
        qv = self._st.encode(qs, convert_to_tensor=True, normalize_embeddings=True)
        pv = self._st.encode(ps, convert_to_tensor=True, normalize_embeddings=True)
        scores = util.cos_sim(qv, pv).diagonal().cpu().tolist()
        out = [0.0] * len(pairs)
        for k, i in enumerate(idx):
            out[i] = float(scores[k])
        return out

def create_reranker(model_name_or_path: str, prefer: Optional[str] = None, device: Optional[str] = None) -> Reranker:
