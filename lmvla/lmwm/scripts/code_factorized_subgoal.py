#!/usr/bin/env python
"""LaWM-style code-factorized subgoal prediction (pooled space), two-stage like LaWAM:

  Stage A (teacher, = LAM):  u = Inv([x_t ; g_future])  (code_dim, LN)
                             g_hat = Fwd([x_frame_t ; u]) -> L2, target = next medoid
  Stage B (student, = policy-predicts-code):  u_hat = Stu(x_t) -> match u (frozen teacher)
  Deploy: subgoal = Fwd([x_frame_t ; u_hat])

Hypothesis: predicting a tiny code (32-D) is easier than a 1280-D latent; the
forward (conditioned on current) fills in the rest -> lower mean error AND lower
variance than direct regression (baseline: cos 0.864, std 0.069, frac<0.7=3.5%).

Inputs use the augin representation (frame+prev-milestone+state, 1332-D). Held-out
episode split identical to the LMWM trainers (seed 2026).
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


def mlp(sizes):
    layers = []
    for i in range(len(sizes) - 2):
        layers += [nn.Linear(sizes[i], sizes[i + 1]), nn.GELU(), nn.LayerNorm(sizes[i + 1])]
    layers.append(nn.Linear(sizes[-2], sizes[-1]))
    return nn.Sequential(*layers)


def cos_stats(pred, tgt):
    p = pred / (np.linalg.norm(pred, axis=1, keepdims=True) + 1e-8)
    c = (p * tgt).sum(1)
    return {"mean": round(float(c.mean()), 4), "std": round(float(c.std()), 4),
            "p05": round(float(np.percentile(c, 5)), 4),
            "frac_lt_0.7": round(float((c < 0.7).mean()), 4),
            "frac_lt_0.75": round(float((c < 0.75).mean()), 4)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="lmwm/data/crave_sequences/kai0base_dinov3h_frame2proto/pairs_next_unique_augin.npz")
    ap.add_argument("--code_dim", type=int, default=32)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--out", default="lmwm/outputs/code_factorized/summary.json", type=Path)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--tail_w", type=float, default=0.0)
    args = ap.parse_args()
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")

    z = np.load(args.pairs)
    n = len(z["current_milestone"])
    _, vi = split_indices(z, n, 0.2, 2026, torch.device("cpu"), "episode")
    vi = vi.numpy(); itr = np.setdiff1d(np.arange(n), vi)
    X = z["current"].astype(np.float32)                    # (n,1332) frame+prev+state
    G = z["next_medoid"].astype(np.float32)                # (n,1280)
    ok = np.linalg.norm(G, axis=1) > 1e-6
    G = G / (np.linalg.norm(G, axis=1, keepdims=True) + 1e-8)
    itr = itr[ok[itr]]; viv = vi[ok[vi]]
    Xt = torch.from_numpy(X); Gt = torch.from_numpy(G)
    din, dg, cd = X.shape[1], G.shape[1], args.code_dim
    frame_slice = slice(0, 1280)  # first 1280 dims = frame feature

    torch.manual_seed(0)
    Inv = mlp([din + dg, 512, cd]).to(dev)
    inv_ln = nn.LayerNorm(cd).to(dev)
    Fwd = mlp([1280 + cd, 512, 512, dg]).to(dev)
    optA = torch.optim.AdamW(list(Inv.parameters()) + list(inv_ln.parameters()) + list(Fwd.parameters()), lr=1e-3, weight_decay=1e-6)

    def teacher(xb, gb):
        u = inv_ln(Inv(torch.cat([xb, gb], 1)))
        gh = F.normalize(Fwd(torch.cat([xb[:, frame_slice], u], 1)), dim=-1)
        return u, gh

    print("Stage A: teacher inverse+forward ...", flush=True)
    itr_t = torch.from_numpy(itr)
    for s in range(args.steps):
        sel = itr_t[torch.randint(0, len(itr_t), (2048,))]
        xb, gb = Xt[sel].to(dev), Gt[sel].to(dev)
        _, gh = teacher(xb, gb)
        loss = F.smooth_l1_loss(gh, gb)
        optA.zero_grad(); loss.backward(); optA.step()

    # teacher reconstruction quality (oracle code) on val
    with torch.no_grad():
        rec = []
        for s in range(0, len(viv), 4096):
            xb, gb = Xt[viv[s:s+4096]].to(dev), Gt[viv[s:s+4096]].to(dev)
            rec.append(teacher(xb, gb)[1].cpu().numpy())
    teacher_stats = cos_stats(np.concatenate(rec), G[viv])
    print("teacher(oracle code):", teacher_stats, flush=True)

    print("Stage B: student predicts code (teacher frozen) ...", flush=True)
    for p in list(Inv.parameters()) + list(inv_ln.parameters()) + list(Fwd.parameters()):
        p.requires_grad_(False)
    Stu = mlp([din, 512, 512, cd]).to(dev)
    optB = torch.optim.AdamW(Stu.parameters(), lr=1e-3, weight_decay=1e-6)
    for s in range(args.steps):
        sel = itr_t[torch.randint(0, len(itr_t), (2048,))]
        xb, gb = Xt[sel].to(dev), Gt[sel].to(dev)
        with torch.no_grad():
            u, _ = teacher(xb, gb)
        u_hat = Stu(xb)
        # code match + end-to-end through frozen forward (+ optional variance tail)
        gh = F.normalize(Fwd(torch.cat([xb[:, frame_slice], u_hat], 1)), dim=-1)
        per = F.smooth_l1_loss(gh, gb, reduction="none").mean(dim=-1)
        e2e = per.mean() + args.tail_w * per.std()
        loss = F.smooth_l1_loss(u_hat, u) + 5.0 * e2e
        optB.zero_grad(); loss.backward(); optB.step()

    with torch.no_grad():
        dep = []
        for s in range(0, len(viv), 4096):
            xb = Xt[viv[s:s+4096]].to(dev)
            u_hat = Stu(xb)
            dep.append(F.normalize(Fwd(torch.cat([xb[:, frame_slice], u_hat], 1)), dim=-1).cpu().numpy())
    deploy_stats = cos_stats(np.concatenate(dep), G[viv])
    print("student deploy (predicted code):", deploy_stats, flush=True)

    summary = {
        "code_dim": cd, "n_train": int(len(itr)), "n_val": int(len(viv)),
        "teacher_oracle_code": teacher_stats,
        "student_deploy": deploy_stats,
        "baseline_direct_head": {"mean": 0.864, "std": 0.069, "frac_lt_0.7": 0.035},
        "baseline_variance_loss": {"mean": 0.858, "std": 0.059, "frac_lt_0.7": 0.025},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
