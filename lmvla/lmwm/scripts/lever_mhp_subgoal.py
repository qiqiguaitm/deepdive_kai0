#!/usr/bin/env python
"""L2: multiple-hypothesis (MCL) subgoal head vs single-regression baseline.

The future is multimodal (~13 branches). A single regression head averages the
modes -> blurry, regression-to-mean. A multiple-choice head predicts K candidate
subgoals + a weight over them, trained winner-take-all (only the candidate closest
to the true medoid is pulled), so candidates specialize on distinct modes. At eval
we take the top-weighted candidate (deployment) and also report the oracle-best
(upper bound). Full pooled augin data, no image encoding -> fast.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lmwm.data import split_indices  # noqa: E402


class Trunk(nn.Module):
    def __init__(self, din, hid=512, depth=2):
        super().__init__()
        layers = [nn.Linear(din, hid), nn.GELU(), nn.LayerNorm(hid)]
        for _ in range(depth - 1):
            layers += [nn.Linear(hid, hid), nn.GELU(), nn.LayerNorm(hid)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SingleHead(nn.Module):
    def __init__(self, din, ld=1280, hid=512):
        super().__init__()
        self.trunk = Trunk(din, hid)
        self.proto = nn.Sequential(nn.Linear(hid, hid), nn.GELU(), nn.LayerNorm(hid), nn.Linear(hid, ld))

    def forward(self, x):
        h = self.trunk(x)
        return F.normalize(self.proto(h), dim=-1).unsqueeze(1), None  # (B,1,ld), no weights


class MHPHead(nn.Module):
    def __init__(self, din, k=4, ld=1280, hid=512):
        super().__init__()
        self.k = k
        self.trunk = Trunk(din, hid)
        self.proto = nn.Sequential(nn.Linear(hid, hid), nn.GELU(), nn.LayerNorm(hid), nn.Linear(hid, k * ld))
        self.gate = nn.Linear(hid, k)
        self.ld = ld

    def forward(self, x):
        h = self.trunk(x)
        cand = self.proto(h).view(-1, self.k, self.ld)
        cand = F.normalize(cand, dim=-1)          # (B,K,ld)
        w = self.gate(h)                           # (B,K) logits
        return cand, w


def mcl_loss(cand, w, med, eps=0.05):
    # cand (B,K,ld), med (B,ld). winner = closest candidate; pull it + relaxed pull on others.
    d = 1.0 - (cand * med.unsqueeze(1)).sum(-1)    # (B,K) cosine distance
    win = d.argmin(1)                              # (B,)
    wta = d[torch.arange(len(d)), win].mean()      # pull winner hard
    relax = d.mean()                               # weak pull on all (avoid dead candidates)
    gate_ce = F.cross_entropy(w, win.detach())     # gate learns which candidate wins
    return wta + eps * relax + 0.5 * gate_ce


def eval_stats(cand, w, med):
    # deployment: top-weighted candidate; oracle: best candidate.
    top = w.argmax(1)
    dep = cand[torch.arange(len(cand)), top]
    c_dep = (dep * med).sum(-1).numpy()
    c_orc = (cand * med.unsqueeze(1)).sum(-1).max(1).values.numpy()
    def s(c):
        return {"cos": round(float(c.mean()), 4), "cos_std": round(float(c.std()), 4),
                "cos_p05": round(float(np.percentile(c, 5)), 4), "lt07": round(float((c < 0.7).mean()), 4)}
    return {"deploy_topweight": s(c_dep), "oracle_best": s(c_orc)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="lmwm/data/crave_sequences/kai0base_dinov3h_frame2proto/pairs_next_unique_augin.npz")
    ap.add_argument("--ks", default="2,4,8")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--out", default="lmwm/outputs/lever_mhp/summary.json", type=Path)
    ap.add_argument("--device", default="cuda:1")
    args = ap.parse_args()
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")

    z = np.load(args.pairs)
    n = len(z["current_milestone"])
    ti, vi = split_indices(z, n, 0.2, 2026, torch.device("cpu"), "episode")
    ti, vi = ti.numpy(), vi.numpy()
    X = z["current"].astype(np.float32); din = X.shape[1]
    med = z["next_medoid"].astype(np.float32)
    ok = np.linalg.norm(med, axis=1) > 1e-6
    med = med / (np.linalg.norm(med, axis=1, keepdims=True) + 1e-8)
    ti = ti[ok[ti]]; vi = vi[ok[vi]]
    Xt, Mt = torch.from_numpy(X[ti]), torch.from_numpy(med[ti])
    Xv, Mv = torch.from_numpy(X[vi]).to(dev), torch.from_numpy(med[vi])
    ntr = len(ti)

    def run(model, is_mhp):
        model = model.to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)
        sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 300))
        for s in range(args.steps):
            bi = torch.randint(0, ntr, (1024,))
            xb, mb = Xt[bi].to(dev), Mt[bi].to(dev)
            cand, w = model(xb)
            loss = mcl_loss(cand, w, mb) if is_mhp else (1.0 - (cand.squeeze(1) * mb).sum(-1)).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        model.eval()
        with torch.no_grad():
            cands, ws = [], []
            for s in range(0, len(Xv), 8192):
                cand, w = model(Xv[s:s + 8192])
                cands.append(cand.cpu()); ws.append(w.cpu() if w is not None else torch.zeros(cand.shape[0], cand.shape[1]))
            return torch.cat(cands), torch.cat(ws)

    results = {}
    print("baseline single-regression ...", flush=True)
    cand, w = run(SingleHead(din), is_mhp=False)
    c = (cand.squeeze(1) * Mv).sum(-1).numpy()
    results["single_regression"] = {"deploy_topweight": {"cos": round(float(c.mean()), 4), "cos_std": round(float(c.std()), 4),
                                    "cos_p05": round(float(np.percentile(c, 5)), 4), "lt07": round(float((c < 0.7).mean()), 4)}}
    print(f"  single cos={c.mean():.4f}±{c.std():.4f} <0.7={float((c<0.7).mean()):.3f}", flush=True)

    for k in [int(x) for x in args.ks.split(",")]:
        print(f"MHP K={k} ...", flush=True)
        cand, w = run(MHPHead(din, k=k), is_mhp=True)
        st = eval_stats(cand, w, Mv)
        results[f"mhp_k{k}"] = st
        print(f"  K={k} deploy cos={st['deploy_topweight']['cos']:.4f}±{st['deploy_topweight']['cos_std']:.4f} "
              f"<0.7={st['deploy_topweight']['lt07']:.3f} | oracle cos={st['oracle_best']['cos']:.4f}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
