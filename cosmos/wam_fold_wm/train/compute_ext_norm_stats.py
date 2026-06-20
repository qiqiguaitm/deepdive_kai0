#!/usr/bin/env python3
"""Compute per-dim quantile norm stats for external cloth datasets.

Output format (same as visrobot01.json):
  {"global": {"action": {"mean": [...], "std": [...], "min": [...], "max": [...], "q01": [...], "q99": [...]}}}

Usage:
  python compute_ext_norm_stats.py [--rig NAME [NAME...]] [--out-dir /path/to/stats]

By default writes one JSON per rig to:
  /mnt/pfs/p46h4f/cosmos/deepdive_kai0/cosmos/wam_fold_policy/data/stats/<domain_name>.json
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

# Add cosmos_framework to path
_REPO = Path(__file__).resolve().parents[3] / "packages" / "cosmos3"
sys.path.insert(0, str(_REPO))

ALOHA_ACTION_INDICES = [0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17, 18, 25]

_EXT_BASE = "/mnt/pfs/p46h4f/cosmos/deepdive_kai0/kai0/data/external_cloth"

RIG_CONFIGS = {
    "robocoin_fold": {
        "root": f"{_EXT_BASE}/robocoin_fold_clothes",
        "domain_name": "robocoin_fold",
        "action_col": "action",
        "action_indices": None,
    },
    "robocoin_r1lite": {
        "root": f"{_EXT_BASE}/robocoin_r1lite_fold_clothes",
        "domain_name": "robocoin_r1lite",
        "action_col": "action",
        "action_indices": None,
    },
    # AgiBot a2d humanoid, 14-D arm joints — all 7 tasks share one stats file
    "agibot_362": {
        "root": f"{_EXT_BASE}/agibot_task362/task_362",
        "domain_name": "agibotworld_fold",
        "action_col": "actions.joint.position",
        "action_indices": None,
    },
    "agibot_444": {
        "root": f"{_EXT_BASE}/agibot_task444/task_444",
        "domain_name": "agibotworld_fold",
        "action_col": "actions.joint.position",
        "action_indices": None,
    },
    "agibot_477": {
        "root": f"{_EXT_BASE}/agibot_task477/task_477",
        "domain_name": "agibotworld_fold",
        "action_col": "actions.joint.position",
        "action_indices": None,
    },
    "agibot_509": {
        "root": f"{_EXT_BASE}/agibot_task509/task_509",
        "domain_name": "agibotworld_fold",
        "action_col": "actions.joint.position",
        "action_indices": None,
    },
    "agibot_520": {
        "root": f"{_EXT_BASE}/agibot_task520/task_520",
        "domain_name": "agibotworld_fold",
        "action_col": "actions.joint.position",
        "action_indices": None,
    },
    "agibot_555": {
        "root": f"{_EXT_BASE}/agibot_task555/task_555",
        "domain_name": "agibotworld_fold",
        "action_col": "actions.joint.position",
        "action_indices": None,
    },
    "agibot_561": {
        "root": f"{_EXT_BASE}/agibot_task561/task_561",
        "domain_name": "agibotworld_fold",
        "action_col": "actions.joint.position",
        "action_indices": None,
    },
    # ALOHA 26-D → 14-D: group under single stats file (same embodiment domain)
    "robocoin_towel_blue": {
        "root": f"{_EXT_BASE}/robocoin_fold_towel_blue",
        "domain_name": "robocoin_aloha",
        "action_col": "action",
        "action_indices": ALOHA_ACTION_INDICES,
    },
    "robocoin_towel_brown": {
        "root": f"{_EXT_BASE}/robocoin_fold_towel_brown",
        "domain_name": "robocoin_aloha",
        "action_col": "action",
        "action_indices": ALOHA_ACTION_INDICES,
    },
    "robocoin_short_sleeve": {
        "root": f"{_EXT_BASE}/robocoin_fold_short_sleeve_white",
        "domain_name": "robocoin_aloha",
        "action_col": "action",
        "action_indices": ALOHA_ACTION_INDICES,
    },
    "robocoin_tray_twice": {
        "root": f"{_EXT_BASE}/robocoin_fold_towel_tray_twice",
        "domain_name": "robocoin_aloha",
        "action_col": "action",
        "action_indices": ALOHA_ACTION_INDICES,
    },
}


def _load_all_actions(cfg: dict) -> np.ndarray:
    """Load and optionally index-extract all actions from a dataset root."""
    parquets = sorted(glob.glob(f"{cfg['root']}/data/chunk-*/episode_*.parquet"))
    if not parquets:
        raise FileNotFoundError(f"No parquet files found in {cfg['root']}")
    chunks = []
    for pf in parquets:
        table = pq.read_table(pf, columns=[cfg["action_col"]])
        rows = table.column(cfg["action_col"]).to_pylist()
        arr = np.asarray(rows, dtype=np.float32)
        if cfg["action_indices"] is not None:
            arr = arr[:, cfg["action_indices"]]
        chunks.append(arr)
    return np.concatenate(chunks, axis=0)  # [N, 14]


def compute_stats(actions: np.ndarray) -> dict:
    return {
        "mean": actions.mean(axis=0).tolist(),
        "std": actions.std(axis=0).tolist(),
        "min": actions.min(axis=0).tolist(),
        "max": actions.max(axis=0).tolist(),
        "q01": np.percentile(actions, 1, axis=0).tolist(),
        "q99": np.percentile(actions, 99, axis=0).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rig", nargs="*", default=list(RIG_CONFIGS.keys()),
        help="Rig names to compute (default: all)"
    )
    parser.add_argument(
        "--out-dir",
        default="/mnt/pfs/p46h4f/cosmos/deepdive_kai0/cosmos/wam_fold_policy/data/stats",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Group rigs by domain_name so ALOHA sub-datasets share stats
    domain_to_rigs: dict[str, list[str]] = {}
    for rig in args.rig:
        if rig not in RIG_CONFIGS:
            print(f"[WARN] Unknown rig {rig!r}, skipping")
            continue
        dn = RIG_CONFIGS[rig]["domain_name"]
        domain_to_rigs.setdefault(dn, []).append(rig)

    for domain_name, rigs in domain_to_rigs.items():
        out_path = Path(args.out_dir) / f"{domain_name}.json"
        print(f"\n[{domain_name}] Collecting actions from {len(rigs)} rig(s): {rigs}")
        all_actions = []
        for rig in rigs:
            cfg = RIG_CONFIGS[rig]
            print(f"  Loading {rig} ({cfg['root']}) ...", end=" ", flush=True)
            acts = _load_all_actions(cfg)
            print(f"{acts.shape[0]} rows, dim={acts.shape[1]}")
            all_actions.append(acts)
        combined = np.concatenate(all_actions, axis=0)
        print(f"  Combined: {combined.shape[0]} rows → computing stats ...")
        stats = compute_stats(combined)
        out = {"global": {"action": stats}}
        out_path.write_text(json.dumps(out, indent=2))
        print(f"  Saved → {out_path}")
        for k in ("mean", "std", "q01", "q99"):
            print(f"    {k}: [{', '.join(f'{v:.4f}' for v in stats[k])}]")

    print("\nDone.")


if __name__ == "__main__":
    main()
