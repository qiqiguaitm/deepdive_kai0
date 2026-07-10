#!/usr/bin/env python
"""L3: trunk capacity sweep (hidden x depth) on augin data.

Baseline trunk is 512-wide, depth-2. Bigger/deeper trunk may squeeze a bit more
discrete mean (top1/NLL) which then compounds with ensembling. Trains a combined
discrete(CE)+subgoal(regress) model per (hidden,depth) on the SAME split; reports
top1/NLL(+std)/cos on held-out. Fast (pooled augin, no image encoding).
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


class Net(nn.Module):
    def __init__(self, din, num_m, hid, depth, ld=1280):
        super().__init__()
        layers = [nn.Linear(din, hid), nn.GELU(), nn.LayerNorm(hid)]
        for _ in range(depth - 1):
            layers += [nn.Linear(hid, hid), nn.GELU(), nn.LayerNorm(hid)]
        self.trunk = nn.Sequential(*layers)
        self.cls = nn.Linear(hid, num_m)
        self.proto = nn.Sequential(nn.Linear(hid, hid), nn.GELU(), nn.LayerNorm(hid), nn.Linear(hid, ld))

    def forward(self, x):
        h = self.trunk(x)
        return self.cls(h), F.normalize(self.proto(h), dim=-1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="lmwm/data/crave_sequences/kai0base_dinov3h_frame2proto/pairs_next_unique_augin.npz")
    ap.add_argument("--grid", default="512x2,768x2,768x3,1024x3,1024x4")
    ap.add_argument("--steps", type=int, default=9000)
    ap.add_argument("--out", default="lmwm/outputs/lever_size/summary.json", type=Path)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")

    z = np.load(args.pairs)
    n = len(z["current_milestone"])
    ti, vi = split_indices(z, n, 0.2, 2026, torch.device("cpu"), "episode")
    ti, vi = ti.numpy(), vi.numpy()
    X = z["current"].astype(np.float32); din = X.shape[1]
    y = z["future_milestone"].astype(np.int64); num_m = int(y.max()) + 1
    med = z["next_medoid"].astype(np.float32)
    ok = np.linalg.norm(med, axis=1) > 1e-6
    med = med / (np.linalg.norm(med, axis=1, keepdims=True) + 1e-8)
    Xt, Yt, Mt = torch.from_numpy(X[ti]), torch.from_numpy(y[ti]), torch.from_numpy(med[ti])
    tok = torch.from_numpy(ok[ti])
    Xv = torch.from_numpy(X[vi]).to(dev); yv = y[vi]; okv = ok[vi]
    Mv = med[vi]
    ntr = len(ti)

    def run(hid, depth):
        torch.manual_seed(0)
        model = Net(din, num_m, hid, depth).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)
        sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 300))
        for s in range(args.steps):
            bi = torch.randint(0, ntr, (1024,))
            lg, pr = model(Xt[bi].to(dev))
            m = tok[bi].to(dev)
            ce = F.cross_entropy(lg, Yt[bi].to(dev))
            pl = (1.0 - (pr * Mt[bi].to(dev)).sum(-1))[m].mean() if m.any() else 0.0
            loss = ce + 5.0 * pl
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        model.eval()
        ps, prs = [], []
        with torch.no_grad():
            for s in range(0, len(Xv), 8192):
                lg, pr = model(Xv[s:s + 8192])
                ps.append(F.softmax(lg, -1).cpu().numpy()); prs.append(pr.cpu().numpy())
        p = np.concatenate(ps); pr = np.concatenate(prs)
        per = -np.log(np.clip(p[np.arange(len(yv)), yv], 1e-12, 1))
        rank = (np.argsort(-p, 1) == yv[:, None]).argmax(1)
        c = (pr[okv] * Mv[okv]).sum(1)
        nparam = sum(pp.numel() for pp in model.parameters())
        return {"params_M": round(nparam / 1e6, 3), "top1": round(float((rank == 0).mean()), 4),
                "top5": round(float((rank < 5).mean()), 4), "nll": round(float(per.mean()), 4),
                "nll_std": round(float(per.std()), 4), "cos": round(float(c.mean()), 4),
                "cos_lt07": round(float((c < 0.7).mean()), 4)}

    results = {}
    for g in args.grid.split(","):
        hid, depth = (int(x) for x in g.split("x"))
        r = run(hid, depth)
        results[g] = r
        print(f"{g:8s} params={r['params_M']:.2f}M top1={r['top1']:.4f} nll={r['nll']:.3f}±{r['nll_std']:.3f} "
              f"cos={r['cos']:.4f} <0.7={r['cos_lt07']:.3f}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("saved", args.out)


if __name__ == "__main__":
    main()
