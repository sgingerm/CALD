# CALD: Community‑Aware Learnable Diffusion (Minimal)

This repository strictly follows the paper’s core pipeline and hyperparameters. It keeps only the essential method: a **query‑conditioned expectation diffusion kernel** with **APPNP‑style propagation (K=1)**, **row‑wise Top‑m pruning (m=30)**, and **pairwise re‑ranking based evaluation**. The upstream graph‑construction stage is intentionally excluded. All paths are relative.

## Structure
- `vector_graph_pipeline/code/util/step1_expand_kernel.py`: loads the graph and index, builds the expectation diffusion kernel, applies row‑wise Top‑30 pruning, runs K=1 propagation, and writes candidates.
- `eval/step2_rerank_eval.py`: performs pairwise re‑ranking and evaluation and exports metrics.
- `data/`: placeholder directory for user‑provided graph and index files (relative paths).

## Usage
```bash
python vector_graph_pipeline/code/util/step1_expand_kernel.py
python eval/step2_rerank_eval.py
```

## Default Settings (paper‑aligned)
- `K = 1`
- `topm = 30` (keep top 30 entries per row before normalization/propagation)
- `alpha = 0.5`
- Other retrieval / re‑ranking hyperparameters follow the paper.
