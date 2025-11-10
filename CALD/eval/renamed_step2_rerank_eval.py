from __future__ import annotations
import subprocess, sys
from pathlib import Path

def main():
    DATASET = Path(r"D:\datanew\question-answer-passages_test.filtered.strict——yuanshi.jsonl")
    OUT_DIR = Path(r"D:\kg_out2\eval")
    GRAPH   = Path(r"D:\kg_out2\global_graph.json")

    KS = ["10","50"]
    RERANK_TOP_N = 220
    BFS_MAX_CHUNKS =280
    DEVICE = "cuda"
    RERANKER_MODEL = None
    MAX_SAMPLES = None
    FILTER_GOLD_IN_GRAPH = True

    repo_root = Path(__file__).resolve().parents[1]
    ks_token = ",".join(str(k) for k in KS)

    cmd = [
        sys.executable, "-m", "eval.evaluate_dataset_multi",
        "--dataset", str(DATASET),
        "--out-dir", str(OUT_DIR),
        "--graph", str(GRAPH),
        "--ks", ks_token,
        "--rerank-top-n", str(RERANK_TOP_N),
        "--bfs-max-chunks", str(BFS_MAX_CHUNKS),
        "--device", DEVICE,
    ]
    if FILTER_GOLD_IN_GRAPH:
        cmd.append("--filter-gold-in-graph")
    if MAX_SAMPLES is not None:
        cmd += ["--max-samples", str(MAX_SAMPLES)]
    if RERANKER_MODEL:
        cmd += ["--reranker-model", str(RERANKER_MODEL)]

    print("\n=== Running evaluation ===")
    print("CWD:", repo_root)
    print("CMD:", " ".join(cmd), "\n")
    subprocess.run(cmd, check=True, cwd=str(repo_root))

if __name__ == "__main__":
    main()
