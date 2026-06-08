#!/usr/bin/env python3
"""Recompute wam_fold action normalization stats on DELTA actions (matches the delta switch
in wam_fold_dataset.py). For each window anchor f and offset o in [0, chunk), the delta target
is  action[f+o] - state[f]  on arm-joint channels (mask True) and absolute action on grippers
(mask False) — identical to WamFoldLeRobotDataset.__getitem__. We sweep all offsets to get the
exact target distribution, then compute per-channel q01/q99/mean/std/min/max.

The `observation.state` block stays the absolute-state stats (state is the eval anchor, not a
model input). Writes the SAME schema as compute_action_stats.py so nothing downstream changes.

usage: compute_delta_action_stats.py [N_EP] [ROOT] [OUT_JSON] [CHUNK]
"""
import json, glob, sys, numpy as np, pandas as pd
from pathlib import Path

N_EP = int(sys.argv[1]) if len(sys.argv) > 1 else 300
ROOT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/mnt/pfs/p46h4f/cosmos/deepdive_kai0/kai0/data/wam_fold_v1/visrobot01_train")
OUT = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("/mnt/pfs/p46h4f/cosmos/deepdive_kai0/cosmos/wam_fold_policy/data/stats/visrobot01.json")
CHUNK = int(sys.argv[4]) if len(sys.argv) > 4 else 16

MASK = np.array([True] * 6 + [False] + [True] * 6 + [False], dtype=bool)  # joints delta, grippers abs

parquets = sorted(glob.glob(str(ROOT / "data" / "chunk-*" / "episode_*.parquet")))
if not parquets:
    print("NO PARQUETS FOUND at", ROOT); sys.exit(1)
idx = np.linspace(0, len(parquets) - 1, min(N_EP, len(parquets))).astype(int)
sel = [parquets[i] for i in idx]
print(f"sampling {len(sel)} / {len(parquets)} episodes; chunk={CHUNK}")

deltas, states = [], []
for i, p in enumerate(sel):
    try:
        df = pd.read_parquet(p, columns=["action", "observation.state"])
    except Exception as e:
        print("skip", p, e); continue
    A = np.stack(df["action"].to_numpy()).astype(np.float64)            # [T,14] absolute action
    S = np.stack(df["observation.state"].to_numpy()).astype(np.float64) # [T,14] absolute state
    states.append(S)
    T = A.shape[0]
    if T <= 1:
        continue
    # exact dataset targets: for every anchor f, offsets o in [0,chunk): action[f+o]-state[f] (masked)
    for o in range(min(CHUNK, T)):
        a = A[o:]              # action at f+o
        s = S[: T - o]         # state at anchor f
        d = a.copy()
        d[:, MASK] = a[:, MASK] - s[:, MASK]   # joints: delta; grippers: absolute (unchanged)
        deltas.append(d)
    if (i + 1) % 50 == 0:
        print(f"  {i+1}/{len(sel)}")

D = np.concatenate(deltas, 0)    # [N,14] delta targets
S = np.concatenate(states, 0)    # [N,14] absolute state
print("delta target rows:", D.shape, "state rows:", S.shape)


def stats(X):
    return dict(
        mean=X.mean(0).tolist(), std=(X.std(0) + 1e-8).tolist(),
        min=X.min(0).tolist(), max=X.max(0).tolist(),
        q01=np.quantile(X, 0.01, 0).tolist(), q99=np.quantile(X, 0.99, 0).tolist(),
    )


out = {
    "global": {"action": stats(D), "observation.state": stats(S)},
    "n_frames": int(D.shape[0]), "n_episodes": len(sel), "action_dim": D.shape[1],
    "action_repr": "delta", "delta_mask": MASK.tolist(), "delta_chunk": CHUNK,
}
OUT.write_text(json.dumps(out, indent=2))
print("WROTE", OUT)
print("delta action q01:", [round(x, 4) for x in out["global"]["action"]["q01"]])
print("delta action q99:", [round(x, 4) for x in out["global"]["action"]["q99"]])
print("STATS_DONE")
