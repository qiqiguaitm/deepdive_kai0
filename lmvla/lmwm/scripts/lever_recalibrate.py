#!/usr/bin/env python
"""L5: honest temperature + fusion recalibration of the fused ensemble.

Fits temperature T (on the ensemble-averaged log-probs) and fusion weight lambda
on a held-out CALIBRATION half of the val set, then reports NLL/variance/top1 on
the untouched TEST half -- an honest recalibration gain, not tuned-on-test. Targets
the discrete NLL mean + tail (std / CVaR) without retraining anything.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lmwm.data import split_indices  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_mean_variance import load_model, forward_all, discrete_stats  # noqa: E402


def nll(probs, y):
    return float(-np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, 1)).mean())


def apply_T_lam(logprob_ens, logprior, T, lam):
    lp = (1 - lam) * (logprob_ens / T) + lam * np.log(np.clip(logprior, 1e-12, 1))
    lp -= lp.max(1, keepdims=True)
    p = np.exp(lp)
    return p / p.sum(1, keepdims=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="lmwm/data/crave_sequences/kai0base_dinov3h_frame2proto/pairs_next_unique_augin.npz")
    ap.add_argument("--out", default="lmwm/outputs/lever_recal/summary.json", type=Path)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")

    z = np.load(args.pairs)
    n = len(z["current_milestone"])
    _, vi = split_indices(z, n, 0.2, 2026, torch.device("cpu"), "episode")
    vi = vi.numpy()
    X = z["current"][vi].astype(np.float32)
    y = z["future_milestone"][vi].astype(np.int64)
    cur_m = z["current_milestone"][vi].astype(np.int64)
    ep = z["episode_id"][vi].astype(np.int64)

    # honest calib/test split of val, BY EPISODE (no leakage)
    uep = np.unique(ep)
    calib_ep = set(uep[::2].tolist())
    is_cal = np.array([e in calib_ep for e in ep])

    gnpz = np.load("lmwm/data/recurrence_graphs/kai0base_dinov3h/recurrence_graph.npz")
    trans = gnpz["transition_probs"].astype(np.float64)
    trans = trans / trans.sum(1, keepdims=True).clip(1e-12)
    prior = trans[cur_m]

    paths = (sorted(glob.glob("lmwm/checkpoints/stage3_augin_big/*/best.pt"))
             + sorted(glob.glob("lmwm/checkpoints/stage3_augin_ens/*/best.pt"))
             + sorted(glob.glob("lmwm/checkpoints/stage3_augin/*/best.pt"))[-1:]
             + sorted(glob.glob("lmwm/checkpoints/stage3_augin_tail/*cecvar*/best.pt")))
    models = [load_model(p, dev)[0] for p in paths]
    probs, _ = forward_all(models, X, dev)                 # ensemble-averaged probs
    logens = np.log(np.clip(probs, 1e-12, 1))

    yc, yt = y[is_cal], y[~is_cal]
    def slc(a): return a[is_cal], a[~is_cal]
    le_c, le_t = slc(logens); pr_c, pr_t = slc(prior)

    # grid search T, lambda on CALIB minimizing NLL
    best = (1e9, 1.0, 0.0)
    for T in [0.7, 0.85, 1.0, 1.15, 1.3, 1.5, 1.8, 2.2]:
        for lam in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
            p = apply_T_lam(le_c, pr_c, T, lam)
            v = nll(p, yc)
            if v < best[0]:
                best = (v, T, lam)
    _, T, lam = best

    # baseline (v0 recipe: T=1 implicit in fuse, lam=0.3) vs recalibrated, on TEST
    p_base = apply_T_lam(le_t, pr_t, 1.0, 0.3)
    p_recal = apply_T_lam(le_t, pr_t, T, lam)
    res = {"fitted": {"T": T, "lambda": lam}, "n_test": int((~is_cal).sum()),
           "baseline_fuse0.3": discrete_stats(p_base, yt),
           "recalibrated": discrete_stats(p_recal, yt)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
