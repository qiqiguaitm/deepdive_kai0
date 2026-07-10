#!/usr/bin/env python
"""Unified mean+variance evaluator for LMWM candidates (single models & ensembles).

For each candidate, on the held-out episode split, against the REAL next milestone
and the episode-medoid subgoal target, reports BOTH central tendency and spread:

  discrete head: top1/top3/top5, NLL mean / std / p90 / CVaR@10%
  subgoal head:  cos mean / std / p05 / frac<0.7

Ensembles average the post-softmax probabilities (discrete) and the L2-normalized
subgoal latents (then re-normalize).
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lmwm.data import split_indices  # noqa: E402
from lmwm.models import UnifiedLMWM  # noqa: E402


def load_model(ckpt_path: str, dev):
    ck = torch.load(ckpt_path, map_location="cpu")
    cfg, meta = ck["config"], ck["meta"]
    g = np.load(cfg["graph_npz"])
    num_m = len(g["prototype_table"]); ld = g["prototype_table"].shape[1]
    m = UnifiedLMWM(int(meta["input_dim"]), ld, num_m, int(cfg["model"]["hidden_dim"]), int(cfg["model"]["depth"])).to(dev)
    m.load_state_dict(ck["model"]); m.eval()
    return m, cfg


def forward_all(models, X, dev, bs=8192):
    probs, protos = None, None
    for m in models:
        ps, gs = [], []
        with torch.no_grad():
            for s in range(0, len(X), bs):
                out = m(torch.from_numpy(X[s:s + bs]).to(dev))
                ps.append(F.softmax(out["greedy_logits"], -1).cpu().numpy())
                gs.append(out["greedy_proto"].cpu().numpy())
        p = np.concatenate(ps); g = np.concatenate(gs)
        probs = p if probs is None else probs + p
        protos = g if protos is None else protos + g
    probs = probs / len(models)
    protos = protos / (np.linalg.norm(protos, axis=1, keepdims=True) + 1e-8)
    return probs, protos


def discrete_stats(probs, y):
    per = -np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, 1.0))
    order = np.argsort(-probs, axis=1)
    ranks = (order == y[:, None]).argmax(1)
    k10 = max(1, int(0.1 * len(per)))
    return {"top1": round(float((ranks == 0).mean()), 4), "top3": round(float((ranks < 3).mean()), 4),
            "top5": round(float((ranks < 5).mean()), 4), "nll": round(float(per.mean()), 4),
            "nll_std": round(float(per.std()), 4), "nll_p90": round(float(np.percentile(per, 90)), 4),
            "nll_cvar10": round(float(np.sort(per)[-k10:].mean()), 4)}


def proto_stats(protos, med):
    c = (protos * med).sum(1)
    return {"cos_mean": round(float(c.mean()), 4), "cos_std": round(float(c.std()), 4),
            "cos_p05": round(float(np.percentile(c, 5)), 4), "frac_lt_0.7": round(float((c < 0.7).mean()), 4)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="lmwm/data/crave_sequences/kai0base_dinov3h_frame2proto/pairs_next_unique_augin.npz")
    ap.add_argument("--out", default="lmwm/outputs/mean_variance/summary.json", type=Path)
    ap.add_argument("--device", default="cuda:1")
    args = ap.parse_args()
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")

    z = np.load(args.pairs)
    n = len(z["current_milestone"])
    _, vi = split_indices(z, n, 0.2, 2026, torch.device("cpu"), "episode")
    vi = vi.numpy()
    X = z["current"][vi].astype(np.float32)
    y = z["future_milestone"][vi].astype(np.int64)
    med = z["next_medoid"][vi].astype(np.float32)
    ok = np.linalg.norm(med, axis=1) > 1e-6
    med = med / (np.linalg.norm(med, axis=1, keepdims=True) + 1e-8)

    def latest(pat):
        c = sorted(glob.glob(pat))
        return c[-1] if c else None

    # graph prior for fusion (Phase-B recipe, lam=0.3)
    gnpz = np.load("lmwm/data/recurrence_graphs/kai0base_dinov3h/recurrence_graph.npz")
    trans = gnpz["transition_probs"].astype(np.float64)
    trans = trans / trans.sum(1, keepdims=True).clip(1e-12)
    p_prior = trans[z["current_milestone"][vi].astype(np.int64)]

    def fuse(probs, lam=0.3):
        lp = (1 - lam) * np.log(np.clip(probs, 1e-12, 1)) + lam * np.log(np.clip(p_prior, 1e-12, 1))
        lp -= lp.max(1, keepdims=True)
        pf = np.exp(lp)
        return pf / pf.sum(1, keepdims=True)

    candidates = {
        "augin_single": [latest("lmwm/checkpoints/stage3_augin/*/best.pt")],
        "augin_focal": [latest("lmwm/checkpoints/stage3_augin_tail/*focal/best.pt")],
        "augin_cecvar": [latest("lmwm/checkpoints/stage3_augin_tail/*cecvar/best.pt")],
        "ensemble_3": sorted(glob.glob("lmwm/checkpoints/stage3_augin_ens/*/best.pt")),
        "ensemble_4(all)": sorted(glob.glob("lmwm/checkpoints/stage3_augin_ens/*/best.pt")) + [latest("lmwm/checkpoints/stage3_augin/*/best.pt")],
        "cvar_ensemble_3": sorted(glob.glob("lmwm/checkpoints/stage3_augin_tail/*cecvar*/best.pt")),
        "mixed_ens_6(all)": sorted(glob.glob("lmwm/checkpoints/stage3_augin_ens/*/best.pt"))
                            + [latest("lmwm/checkpoints/stage3_augin/*/best.pt")]
                            + sorted(glob.glob("lmwm/checkpoints/stage3_augin_tail/*cecvar*/best.pt")),
        "big_ens_3": sorted(glob.glob("lmwm/checkpoints/stage3_augin_big/*/best.pt")),
        "big3+mixed6": sorted(glob.glob("lmwm/checkpoints/stage3_augin_big/*/best.pt"))
                       + sorted(glob.glob("lmwm/checkpoints/stage3_augin_ens/*/best.pt"))
                       + [latest("lmwm/checkpoints/stage3_augin/*/best.pt")]
                       + sorted(glob.glob("lmwm/checkpoints/stage3_augin_tail/*cecvar*/best.pt")),
        "hetero_12(all)": sorted(glob.glob("lmwm/checkpoints/stage3_augin_big/*/best.pt"))
                       + sorted(glob.glob("lmwm/checkpoints/stage3_augin_div/*/best.pt"))
                       + sorted(glob.glob("lmwm/checkpoints/stage3_augin_ens/*/best.pt"))
                       + [latest("lmwm/checkpoints/stage3_augin/*/best.pt")]
                       + sorted(glob.glob("lmwm/checkpoints/stage3_augin_tail/*cecvar*/best.pt")),
    }
    results = {}
    for name, paths in candidates.items():
        paths = [p for p in paths if p]
        if not paths:
            print(f"{name}: SKIP (no ckpt)")
            continue
        models = [load_model(p, dev)[0] for p in paths]
        probs, protos = forward_all(models, X, dev)
        for tag, pp in [(name, probs), (name + "+fuse0.3", fuse(probs))]:
            d = discrete_stats(pp, y)
            p = proto_stats(protos[ok], med[ok])
            results[tag] = {"n_members": len(paths), "discrete": d, "subgoal": p}
            print(f"{tag:24s} top1={d['top1']:.4f} nll={d['nll']:.3f}±{d['nll_std']:.3f} p90={d['nll_p90']:.3f} cvar10={d['nll_cvar10']:.3f} | "
                  f"cos={p['cos_mean']:.4f}±{p['cos_std']:.4f} p05={p['cos_p05']:.3f} <0.7={p['frac_lt_0.7']:.3f}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
