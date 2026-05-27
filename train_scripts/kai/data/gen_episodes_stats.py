#!/usr/bin/env python3
"""Generate episodes_stats.jsonl for a v2 lerobot dataset (compute per-episode stats from parquet).

Usage:
  python gen_episodes_stats.py <dataset_root>
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

STAT_KEYS = ["observation.state", "action", "timestamp", "frame_index", "episode_index", "index", "task_index"]


def episode_stats(parquet_path: Path) -> dict:
    t = pq.read_table(parquet_path)
    out = {}
    for k in STAT_KEYS:
        if k in t.column_names:
            arr = np.array(t[k].to_pylist())
            if arr.ndim == 1:
                arr = arr[:, None]
            arr = arr.astype(np.float64)
            out[k] = {
                "mean": np.mean(arr, axis=0).tolist(),
                "std": np.std(arr, axis=0).tolist(),
                "min": np.min(arr, axis=0).tolist(),
                "max": np.max(arr, axis=0).tolist(),
                "count": [int(arr.shape[0])],
            }
            # Flatten 1d -> scalars per stat key (lerobot expects list)
            if all(len(v) == 1 for v in out[k].values() if isinstance(v, list)):
                pass
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root")
    args = p.parse_args()
    root = Path(args.root)
    out_path = root / "meta" / "episodes_stats.jsonl"
    if out_path.exists():
        print(f"already exists: {out_path}", file=sys.stderr)
        return

    # Read episodes.jsonl to get the indices
    indices = []
    for line in (root / "meta" / "episodes.jsonl").open():
        d = json.loads(line)
        indices.append(d["episode_index"])

    # Iterate over data files (chunk-000 only, gaps OK)
    chunk_dir = root / "data" / "chunk-000"
    n = 0
    with out_path.open("w") as f:
        for idx in sorted(indices):
            pq_path = chunk_dir / f"episode_{idx:06d}.parquet"
            if not pq_path.exists():
                print(f"skip missing {pq_path}", file=sys.stderr)
                continue
            stats = episode_stats(pq_path)
            f.write(json.dumps({"episode_index": idx, "stats": stats}) + "\n")
            n += 1
    print(f"wrote {n} entries → {out_path}")


if __name__ == "__main__":
    main()
