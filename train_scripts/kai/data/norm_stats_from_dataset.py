"""Config-free norm_stats computation directly from a built lerobot dataset.

Replicates `kai0/scripts/compute_norm_states_fast.py` **exactly** (same RunningStats, same
pad-to-action_dim, same [-pi,pi] filter, same batch=32 accumulation, same openpi serializer)
but WITHOUT needing a registered config — takes the dataset dir + action_dim directly.

Purpose: let dataset-build scripts auto-(re)compute norm_stats from the *just-built* dataset,
so norm_stats is never forgotten or accidentally reused from the source (the #1 submit pitfall,
see docs/.../submission/training_pitfalls_common.md §1).

Usage:
  - import:    from norm_stats_from_dataset import compute_norm_stats
               compute_norm_stats("/path/to/self_built/MyDataset", action_dim=32)
  - CLI:       python norm_stats_from_dataset.py /path/to/dataset [--action-dim 32]

Writes `<dataset_dir>/norm_stats.json` (openpi format). For pi0/pi05, action_dim=32.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


def _ensure_openpi_on_path():
    """Add kai0/src to sys.path so `openpi` imports without installing."""
    src = Path(__file__).resolve().parents[3] / "kai0" / "src"
    if src.is_dir() and str(src) not in sys.path:
        sys.path.insert(0, str(src))


def _pad_to_dim(data, target_dim):
    data = np.asarray(data)
    if data.shape[-1] >= target_dim:
        return data[..., :target_dim]
    padding = np.zeros((*data.shape[:-1], target_dim - data.shape[-1]))
    return np.concatenate([data, padding], axis=-1)


def _process(arr, action_dim):
    """FakeInputs logic: pad to action_dim, zero out abnormal (|x|>pi) values."""
    arr = _pad_to_dim(arr, action_dim)
    arr = np.where(arr > np.pi, 0, arr)
    arr = np.where(arr < -np.pi, 0, arr)
    return arr


def compute_norm_stats(dataset_dir, action_dim: int = 32,
                       state_col: str = "observation.state", action_col: str = "action",
                       max_frames: int | None = None):
    """Compute + write norm_stats.json for a built dataset. Returns the stats dict.

    Identical numerics to compute_norm_states_fast.py: sorted parquet order, per-row pad/filter,
    concatenate-then-feed RunningStats in batches of 32, openpi serializer.
    """
    _ensure_openpi_on_path()
    import openpi.shared.normalize as normalize  # lazy: clear error if env lacks openpi

    base = Path(dataset_dir)
    parquet_files = sorted(str(p) for p in base.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"no .parquet under {base} — nothing to compute norm_stats from")

    collected = {"state": [], "actions": []}
    total = 0
    for fp in tqdm(parquet_files, desc="norm_stats: reading parquet"):
        try:
            df = pd.read_parquet(fp)
        except Exception as e:
            print(f"  [norm_stats] skip unreadable {fp}: {e}")
            continue
        if state_col not in df.columns or action_col not in df.columns:
            continue
        states, actions = [], []
        for i in range(len(df)):
            try:
                states.append(_process(np.array(df[state_col].iloc[i]), action_dim))
                actions.append(_process(np.array(df[action_col].iloc[i]), action_dim))
                total += 1
                if max_frames is not None and total >= max_frames:
                    break
            except Exception:
                continue
        if states:
            collected["state"].append(np.stack(states))
        if actions:
            collected["actions"].append(np.stack(actions))
        if max_frames is not None and total >= max_frames:
            break

    stats = {}
    for key in ("state", "actions"):
        if not collected[key]:
            raise ValueError(
                f"[norm_stats] no data collected for '{key}' — check columns "
                f"(expected '{state_col}' / '{action_col}' in parquet)")
        data = np.concatenate(collected[key], axis=0)
        rs = normalize.RunningStats()
        for i in range(0, len(data), 32):              # batch=32 → identical FP accumulation
            rs.update(data[i:i + 32])
        stats[key] = rs.get_statistics()

    normalize.save(base, stats)                        # writes <base>/norm_stats.json
    print(f"✅ [norm_stats] written → {base}/norm_stats.json  (frames={total}, action_dim={action_dim})")
    return stats


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Compute norm_stats.json directly from a built dataset (no config).")
    ap.add_argument("dataset_dir", help="dataset root (contains data/, meta/, videos/)")
    ap.add_argument("--action-dim", type=int, default=32, help="model action_dim to pad to (pi0/pi05=32)")
    ap.add_argument("--max-frames", type=int, default=None)
    a = ap.parse_args()
    compute_norm_stats(a.dataset_dir, action_dim=a.action_dim, max_frames=a.max_frames)
